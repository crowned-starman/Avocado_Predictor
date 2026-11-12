from __future__ import annotations

import csv
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from shapely.geometry import shape
from .common import data_signature, digest, load_plan, read_json, weeks, work, write_json
from .geo import make_tiles, plot_plan


def create_plan(c, scenes=None):
    if scenes is None:
        from .copernicus import search
        scenes = search(c)
    d = c["data"]
    tiles = make_tiles(scenes, c["grid_epsg"], d["size"], d["resolution_m"], d["max_spatial_tiles"])
    p = {"schema": 1, "data_signature": data_signature(c), "aoi": c["aoi"], "epsg": c["grid_epsg"],
         "weeks": weeks(d["start"], d["end"]), "tiles": tiles, "scenes": scenes}
    p["fingerprint"] = digest(p)
    directory = work(c)
    old = directory / "plan.json"
    if old.exists() and read_json(old)["fingerprint"] != p["fingerprint"] and (directory / "weekly").exists():
        raise ValueError("Ya hay descargas de un plan distinto. Elige otra work_dir; no se mezclan planes.")
    write_json(old, p)
    plot_plan(p, directory / "cobertura.png")
    with (directory / "recuadros.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "selected", "lat", "lon", "footprint_fraction"])
        writer.writeheader()
        writer.writerows({k: t[k] for k in writer.fieldnames} for t in tiles)
    n = sum(t["selected"] for t in tiles)
    records = n * len(p["weeks"])
    gb = records * d["size"]**2 * (6*2+1) / 1e9
    print(f"Escenas: {len(scenes)}. Cobertura: {len(tiles)} recuadros; seleccionados: {n}.")
    print(f"Semanas: {len(p['weeks'])}; máximo {records} archivos semanales; ~{gb:.2f} GB sin comprimir.")
    print(f"Hasta {records * (d['candidates_per_week'] + 1)} peticiones de imagen (sin reintentos).")
    print(f"Revisa {directory / 'cobertura.png'} antes de descargar.")
    return p


def record_path(c, tile_id, week):
    return work(c) / "weekly" / tile_id / f"{week['start']}.npz"


def cached_record(path, fingerprint):
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return None
    meta = read_json(meta_path)
    if meta.get("plan_fingerprint") != fingerprint:
        raise ValueError(f"Caché de otro plan en {path}. Usa otra work_dir.")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"reflectance", "valid"} or z["reflectance"].shape[0] != 6:
            raise ValueError(f"Archivo semanal inválido: {path}")
        if z["reflectance"].shape[1:] != z["valid"].shape or not np.isfinite(z["reflectance"]).all():
            raise ValueError(f"Datos semanales corruptos: {path}")
    return meta


def download(c, max_records=None):
    from .copernicus import ImageClient, valid_mask
    p = load_plan(c)
    client, manifest, attempts = None, [], 0
    dates = {s["id"]: datetime.fromisoformat(s["datetime"].replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
             for s in p["scenes"]}
    try:
        for tile in (t for t in p["tiles"] if t["selected"]):
            polygon = shape(tile["geometry"])
            tile_scenes = [s for s in p["scenes"] if shape(s["geometry"]).intersects(polygon)]
            for week in p["weeks"]:
                path = record_path(c, tile["id"], week)
                base = {"tile": tile["id"], "week": week["index"], "week_start": week["start"]}
                meta = cached_record(path, p["fingerprint"])
                if meta:
                    manifest.append({**base, "status": "cached", "valid_fraction": meta["valid_fraction"]})
                    continue
                if max_records is not None and attempts >= max_records:
                    manifest.append({**base, "status": "pending"})
                    continue
                attempts += 1
                candidates = [s for s in tile_scenes if week["start"] <= dates[s["id"]] <= week["end"]]
                candidates.sort(key=lambda s: (s["cloud"], s["datetime"], s["id"]))
                # Deduplicar adquisiciones solapadas antes de gastar peticiones.
                unique = {}
                for s in candidates:
                    unique.setdefault(s["datetime"], s)
                candidates = list(unique.values())[:c["data"]["candidates_per_week"]]
                best, score = None, -1.0
                for scene in candidates:
                    if client is None:
                        client = ImageClient()
                    valid = valid_mask(client.request(tile, scene, c, quality=True))
                    fraction = float(valid.mean())
                    if fraction > score:
                        best, score = (scene, valid), fraction
                    if score >= .98:
                        break
                if best is None or score < c["data"]["min_valid_fraction"]:
                    status = "no_scene" if best is None else "clouds_or_no_data"
                    manifest.append({**base, "status": status, "valid_fraction": max(0, score)})
                    print(f"{tile['id']} {week['start']}: {status}", flush=True)
                    continue
                scene, valid = best
                image = client.request(tile, scene, c, quality=False).astype(np.float32)
                valid &= np.isfinite(image).all(axis=-1)
                if float(valid.mean()) < c["data"]["min_valid_fraction"]:
                    manifest.append({**base, "status": "nonfinite_pixels"})
                    continue
                image[~valid] = 0
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, reflectance=image.transpose(2, 0, 1).astype(np.float16), valid=valid)
                os.replace(temporary, path)
                meta = {**base, "schema": 1, "plan_fingerprint": p["fingerprint"],
                        "scene": scene, "acquisition": scene["datetime"], "week_end": week["end"],
                        "valid_fraction": float(valid.mean()), "lat": tile["lat"], "lon": tile["lon"],
                        "bbox": tile["bbox"], "epsg": c["grid_epsg"], "bands": c["data"]["bands"],
                        "units": "reflectance_0_1_not_DN", "resolution_m": 30}
                write_json(path.with_suffix(".json"), meta)
                manifest.append({**base, "status": "ok", "valid_fraction": meta["valid_fraction"]})
                print(f"{tile['id']} {week['start']}: {meta['valid_fraction']:.1%} válido", flush=True)
    finally:
        # Se guarda aun si un error de red interrumpe la ejecución. Las imágenes terminadas se reutilizan.
        write_json(work(c) / "download_status.json", {"plan_fingerprint": p["fingerprint"], "records": manifest})
    from collections import Counter
    print(dict(Counter(r["status"] for r in manifest)))


def split_indices(n, train_fraction, val_fraction):
    a, b = int(n * train_fraction), int(n * (train_fraction + val_fraction))
    return {"train": (0, a), "val": (a, b), "test": (b, n)}
