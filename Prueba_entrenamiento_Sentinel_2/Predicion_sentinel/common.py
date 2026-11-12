from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def weeks(start, end):
    """Semanas de lunes a domingo, recortadas en ambos extremos (UTC)."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("La fecha inicial debe ser anterior a la final.")
    result = []
    current = first
    while current <= last:
        stop = min(current + timedelta(days=6 - current.weekday()), last)
        result.append({"index": len(result), "start": current.isoformat(), "end": stop.isoformat()})
        current = stop + timedelta(days=1)
    return result


def load_config(path):
    path = Path(path).resolve()
    c = read_json(path)
    if c["data"]["bands"] != BANDS:
        raise ValueError(f"Las bandas y su orden deben ser {BANDS}.")
    d = c["data"]
    weeks(d["start"], d["end"])
    if d["size"] != 224 or d["resolution_m"] != 30:
        raise ValueError("Esta prueba está validada para ventanas de 224 x 224 a 30 m.")
    if not 0 < d["min_valid_fraction"] <= 1 or not 0 <= d["max_cloud"] <= 1:
        raise ValueError("Los porcentajes de validez y nubes se expresan entre 0 y 1.")
    if d["max_spatial_tiles"] is not None and d["max_spatial_tiles"] < 1:
        raise ValueError("max_spatial_tiles debe ser positivo o null (cobertura completa).")
    if d["candidates_per_week"] < 1 or d["candidates_per_week"] > 20:
        raise ValueError("candidates_per_week debe estar entre 1 y 20.")
    if d["frames"] != 4:
        raise ValueError("El pronóstico requiere cuatro fechas de entrada.")
    if not isinstance(c.get("seed"), int):
        raise ValueError("La semilla seed debe ser un entero.")
    if not 0 < d["train_fraction"] < 1 or not 0 < d["val_fraction"] < 1 - d["train_fraction"]:
        raise ValueError("Las fracciones de entrenamiento y validación dejan un conjunto de prueba.")
    c["_config_path"] = str(path)
    c["_work"] = str((ROOT / c["work_dir"]).resolve())
    return c


def data_signature(c):
    # El caché no depende de las épocas ni de la tasa de aprendizaje.
    return digest({"schema": 1, "data": c["data"], "aoi": c["aoi"], "epsg": c["grid_epsg"]})


def work(c):
    p = Path(c["_work"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_plan(c):
    p = read_json(work(c) / "plan.json")
    if p["data_signature"] != data_signature(c):
        raise ValueError("Cambió la configuración de datos. Ejecuta plan de nuevo en otra work_dir.")
    if p["fingerprint"] != digest({k: v for k, v in p.items() if k != "fingerprint"}):
        raise ValueError("El plan se modificó después de generarse; vuelve a ejecutar plan.")
    return p
