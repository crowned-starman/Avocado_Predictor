"""Parejas pasado -> futuro y caché de características; nunca codifica la imagen objetivo."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import digest, load_plan, read_json, work, write_json
from .data import cached_record, record_path, split_indices
from .model import WeeklyDataset, build_model


def settings(c):
    f = c.get("forecast")
    if not f:
        raise ValueError("La configuración necesita la sección forecast. Usa los JSON de esta versión.")
    if f["horizon_weeks"] != 1 or c["data"]["frames"] != 4:
        raise ValueError("Esta prueba implementa exactamente cuatro semanas -> una semana posterior.")
    if min(f[k] for k in ["epochs", "batch_size", "accumulate", "hidden_channels", "project_channels", "patience"]) < 1:
        raise ValueError("Los parámetros enteros de forecast deben ser positivos.")
    if f["lr"] <= 0:
        raise ValueError("forecast.lr debe ser positivo.")
    return f


def forecast_work(c):
    p = work(c) / "forecast"
    p.mkdir(parents=True, exist_ok=True)
    return p


def issue_time(week_start):
    d = date.fromisoformat(week_start) - timedelta(days=1)
    return d.isoformat() + "T23:59:59+00:00"


def make_sample(c, p, tile, context, target_week, target=None):
    sample = {"tile": tile["id"], "paths": [r[0] for r in context],
              "dates": [r[1]["acquisition"] for r in context],
              "lat": tile["lat"], "lon": tile["lon"], "bbox": tile["bbox"],
              "epsg": p["epsg"], "forecast_week": target_week,
              "issue_time": issue_time(target_week["start"])}
    origin = datetime.fromisoformat(sample["issue_time"])
    if any(datetime.fromisoformat(x.replace("Z", "+00:00")) > origin for x in sample["dates"]):
        raise ValueError("Una imagen de entrada es posterior a la fecha de emisión.")
    if len(set(sample["dates"])) != 4:
        raise ValueError("Las cuatro observaciones de entrada deben ser distintas.")
    # El identificador y la entrada no dependen de los píxeles ni de la fecha real del objetivo.
    sample["id"] = digest(sample)[:24]
    if target is not None:
        sample["target_path"] = target[0]
        sample["target_date"] = target[1]["acquisition"]
        td = datetime.fromisoformat(sample["target_date"].replace("Z", "+00:00")).date().isoformat()
        if not target_week["start"] <= td <= target_week["end"]:
            raise ValueError("La imagen objetivo no pertenece a la semana pronosticada.")
    return sample


def prepare_forecast(c):
    settings(c)
    p = load_plan(c)
    ranges = split_indices(len(p["weeks"]), c["data"]["train_fraction"], c["data"]["val_fraction"])
    manifest = {"schema": 1, "task": "four_observed_weeks_to_next_week_six_bands",
                "plan_fingerprint": p["fingerprint"], "ranges": ranges, "splits": {k: [] for k in ranges}}
    for tile in (t for t in p["tiles"] if t["selected"]):
        available = {}
        for week in p["weeks"]:
            path = record_path(c, tile["id"], week)
            meta = cached_record(path, p["fingerprint"])
            if meta:
                available[week["index"]] = (path.relative_to(work(c)).as_posix(), meta)
        for name, (start, stop) in ranges.items():
            for i in range(start, stop - 4):
                if not all(j in available for j in range(i, i+5)):
                    continue
                sample = make_sample(c, p, tile, [available[j] for j in range(i, i+4)],
                                     p["weeks"][i+4], available[i+4])
                sample["context_week_indices"] = list(range(i, i+4))
                sample["target_week_index"] = i+4
                manifest["splits"][name].append(sample)
    manifest["fingerprint"] = digest(manifest)
    destination = forecast_work(c) / "dataset.json"
    if destination.exists() and read_json(destination)["fingerprint"] != manifest["fingerprint"]:
        if (forecast_work(c)/"training/last.pt").exists():
            raise ValueError("Ya entrenaste con otras parejas. Conserva aparte forecast/training antes de preparar otro conjunto.")
    write_json(destination, manifest)
    for name, samples in manifest["splits"].items():
        print(f"{name}: {len(samples)} parejas; cuatro semanas pasadas -> semana siguiente.")
    if any(not rows for rows in manifest["splits"].values()):
        print("Faltan parejas en alguna partición. Revisa semanas vacías antes de extraer características.")
    return manifest


def load_forecast_manifest(c):
    settings(c)
    p = load_plan(c)
    manifest = read_json(forecast_work(c)/"dataset.json")
    if manifest["plan_fingerprint"] != p["fingerprint"] or manifest["fingerprint"] != digest({k:v for k,v in manifest.items() if k != "fingerprint"}):
        raise ValueError("Las parejas de pronóstico no corresponden al plan. Ejecuta pronostico.py prepare.")
    seen = set()
    for name, rows in manifest["splits"].items():
        paths = set()
        for s in rows:
            if s["target_path"] in s["paths"] or len(s["paths"]) != 4:
                raise ValueError("El objetivo está dentro de la entrada o faltan fechas.")
            paths.update(s["paths"] + [s["target_path"]])
        if seen & paths:
            raise ValueError("Hay imágenes compartidas entre particiones de pronóstico.")
        seen.update(paths)
    return manifest


def extract_spatial_features(encoder, batch):
    # Se invoca SOLO con las cuatro imágenes de contexto.
    latent, _, restore = encoder(batch["x"], batch["time"], batch["location"], mask_ratio=0.0)
    ordered = torch.gather(latent[:,1:], 1, restore.unsqueeze(-1).expand(-1,-1,latent.shape[-1]))
    pt, ph, pw = encoder.patch_embed.patch_size
    b, _, t, h, w = batch["x"].shape
    if pt != 1:
        raise ValueError("La prueba temporal requiere parches de una fecha.")
    return ordered.reshape(b,t,h//ph,w//pw,-1).permute(0,1,4,2,3).contiguous()


def feature_path(c, meta, sample):
    return forecast_work(c)/"features"/meta["encoder_signature"][:16]/f"{sample['id']}.npz"


def valid_feature(path, signature):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        if str(z["signature"].item()) != signature:
            raise ValueError(f"Características de otro codificador: {path}")
        if z["features"].ndim != 4 or not np.isfinite(z["features"]).all():
            raise ValueError(f"Caché de características inválido: {path}")
    return True


def save_features(path, features, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, features=features, signature=np.array(signature))
    os.replace(temp, path)


def build_feature_metadata(c, bundle, manifest):
    signature = digest({"model":c["model"], "pretrained_config":bundle.config,
                        "encoder_state":"official_pretrained_frozen", "schema":1})
    meta = {"schema":1, "model":c["model"], "encoder_signature":signature,
            "dataset_fingerprint":manifest["fingerprint"], "embedding_dim":bundle.model.encoder.embed_dim,
            "patch_size":list(bundle.model.encoder.patch_embed.patch_size),
            "mean":bundle.mean.flatten().tolist(), "std":bundle.std.flatten().tolist()}
    meta["fingerprint"] = digest(meta)
    return meta


def cache_features(c, allow_cpu=False, max_records=None):
    from .model import device_for, move
    manifest = load_forecast_manifest(c)
    if any(not rows for rows in manifest["splits"].values()):
        raise ValueError("Faltan parejas de entrenamiento, validación o prueba. Completa los datos y prepare.")
    device = device_for(allow_cpu)
    bundle = build_model(c)
    encoder = bundle.model.encoder.requires_grad_(False).to(device).eval()
    meta = build_feature_metadata(c, bundle, manifest)
    write_json(forecast_work(c)/"features.json", meta)
    saved = reused = pending = 0
    with torch.inference_mode():
        for name, rows in manifest["splits"].items():
            context_dataset = WeeklyDataset(rows, work(c), bundle.mean, bundle.std)
            for i, s in enumerate(rows):
                path = feature_path(c, meta, s)
                if valid_feature(path, meta["encoder_signature"]):
                    reused += 1
                    continue
                if max_records is not None and saved >= max_records:
                    pending += 1
                    continue
                raw = context_dataset[i]
                batch = move({k:v.unsqueeze(0) for k,v in raw.items() if torch.is_tensor(v)}, device)
                features = extract_spatial_features(encoder, batch)[0].float().cpu().numpy().astype(np.float16)
                save_features(path, features, meta["encoder_signature"])
                saved += 1
                print(f"{name}: contexto {i+1}/{len(rows)} guardado; sin leer la imagen objetivo.", flush=True)
    print(f"Características: {saved} nuevas, {reused} reutilizadas, {pending} pendientes.")
    return meta


def load_feature_metadata(c, manifest=None):
    meta = read_json(forecast_work(c)/"features.json")
    if meta["model"] != c["model"] or meta["fingerprint"] != digest({k:v for k,v in meta.items() if k != "fingerprint"}):
        raise ValueError("Características de otro modelo o metadatos modificados.")
    if manifest and meta["dataset_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Las características corresponden a otras parejas; ejecuta features.")
    return meta


def known_time_features(sample):
    # El día real de adquisición del objetivo NO entra en el predictor.
    nominal = date.fromisoformat(sample["forecast_week"]["end"])
    phase = 2*np.pi*(nominal.timetuple().tm_yday-1)/365.25
    result = []
    for v in sample["dates"]:
        d = datetime.fromisoformat(v.replace("Z", "+00:00")).date()
        result.append([(nominal-d).days/7.0, np.sin(phase), np.cos(phase)])
    return torch.tensor(result, dtype=torch.float32)


def context_tensors(c, sample, meta):
    mean = torch.tensor(meta["mean"]).reshape(6,1,1,1)
    std = torch.tensor(meta["std"]).reshape(6,1,1,1)
    batch = WeeklyDataset([sample], work(c), mean, std)[0]
    # Referencia causal: última observación válida por píxel entre las cuatro semanas.
    baseline = torch.zeros_like(batch["x"][:,0])
    support = batch["valid"].any(0)
    for t in range(4):
        baseline = torch.where(batch["valid"][t].unsqueeze(0), batch["x"][:,t], baseline)
    _, ph, pw = meta["patch_size"]
    quality = F.avg_pool2d(batch["valid"].float().unsqueeze(1), (ph,pw), stride=(ph,pw))
    return {"baseline":baseline, "support":support, "quality":quality,
            "known_time":known_time_features(sample)}


class ForecastDataset(Dataset):
    def __init__(self, c, samples, meta):
        self.c, self.samples, self.meta = c, samples, meta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        path = feature_path(self.c, self.meta, s)
        if not valid_feature(path, self.meta["encoder_signature"]):
            raise ValueError("Faltan características. Ejecuta pronostico.py features sin --max-records.")
        with np.load(path, allow_pickle=False) as z:
            features = torch.from_numpy(z["features"].astype(np.float32))
        result = context_tensors(self.c, s, self.meta)
        mean, std = [torch.tensor(self.meta[k]).reshape(6,1,1) for k in ["mean","std"]]
        with np.load(work(self.c)/s["target_path"], allow_pickle=False) as z:
            target = torch.from_numpy(z["reflectance"].astype(np.float32))
            valid = torch.from_numpy(z["valid"].astype(bool))
        result.update({"features":features, "target":(target*10000-mean)/std,
                       "valid":valid & result["support"], "index":i})
        if features.shape[0] != 4 or features.shape[1] != self.meta["embedding_dim"] or features.shape[-2:] != result["quality"].shape[-2:]:
            raise ValueError("Dimensiones incompatibles entre características, máscaras y modelo.")
        return result
