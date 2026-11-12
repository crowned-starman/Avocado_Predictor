from __future__ import annotations

import importlib.util
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from .common import read_json


@dataclass
class Bundle:
    model: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    config: dict


def model_class(c):
    m = c["model"]
    if len(m["revision"]) != 40:
        raise ValueError("Usa un commit fijo de 40 caracteres para reproducir el código y los pesos.")
    # Código oficial en un commit inmutable; no se toma la rama main cambiante.
    source = hf_hub_download(m["repo"], "prithvi_mae.py", revision=m["revision"])
    spec = importlib.util.spec_from_file_location("prithvi_pinned", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrithviMAE


def load_model_config(c):
    m = c["model"]
    config = read_json(hf_hub_download(m["repo"], "config.json", revision=m["revision"]))["pretrained_cfg"]
    if config["in_chans"] != 6 or config["num_frames"] != 4 or config.get("norm_pix_loss", False):
        raise ValueError("La configuración del modelo no coincide con esta prueba de seis bandas y cuatro fechas.")
    pt, ph, pw = config["patch_size"]
    if c["data"]["frames"] % pt or c["data"]["size"] % ph or c["data"]["size"] % pw:
        raise ValueError("Las dimensiones de entrada deben ser divisibles por el parche del modelo.")
    return config


def inspect_model(c):
    """Verifica arquitectura y conteos sin descargar ni reservar memoria para los pesos."""
    config = load_model_config(c)
    cls = model_class(c)
    with torch.device("meta"):
        model = cls(**config).requires_grad_(False)
        from .forecast_net import make_head
        head = make_head(c, {"embedding_dim":config["embed_dim"],"patch_size":config["patch_size"],
                             "mean":config["mean"],"std":config["std"]})
    counts = {"checkpoint_total":sum(p.numel() for p in model.parameters()),
              "encoder":sum(p.numel() for p in model.encoder.parameters()),
              "encoder_trainable":0,"forecast_head_trainable":sum(p.numel() for p in head.parameters())}
    pt, ph, pw = config["patch_size"]
    frames, size = c["data"]["frames"], c["data"]["size"]
    result = {"repo": c["model"]["repo"], "revision": c["model"]["revision"],
              "validation": "architecture_only_meta_device_no_weights_or_forward",
              "input_shape": [1, 6, frames, size, size], "patch_size": config["patch_size"],
              "feature_shape": [1, frames, model.encoder.embed_dim, size//ph, size//pw],
              "prediction_shape": [1,6,size,size],
              "embedding_dim": model.encoder.embed_dim, "parameters": counts,
              "parameter_storage_fp32_gib": counts["checkpoint_total"] * 4 / 2**30}
    print(json.dumps(result, indent=2))
    return result


def build_model(c):
    m = c["model"]
    config = load_model_config(c)
    model = model_class(c)(**config)
    checkpoint = torch.load(hf_hub_download(m["repo"], m["weights"], revision=m["revision"]),
                            map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    # Posiciones sinusoidales se regeneran; todo el resto debe coincidir exactamente.
    state = {k: v for k, v in checkpoint.items() if "pos_embed" not in k}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any("pos_embed" not in k for k in incompatible.missing_keys):
        raise ValueError(f"Pesos incompatibles: {incompatible}")
    mean = torch.tensor(config["mean"], dtype=torch.float32).reshape(6, 1, 1, 1)
    std = torch.tensor(config["std"], dtype=torch.float32).reshape(6, 1, 1, 1)
    if not torch.all(std > 0):
        raise ValueError("Desviaciones de normalización inválidas.")
    return Bundle(model.requires_grad_(False).eval(), mean, std, config)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def autocast(device, enabled):
    return torch.autocast(device_type="cuda",dtype=torch.float16) if device.type=="cuda" and enabled else nullcontext()


def save_checkpoint(path, payload):
    path=Path(path)
    tmp=path.with_suffix(".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def device_for(allow_cpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        print("Extracción Prithvi en CPU autorizada: requiere RAM y puede tardar bastante.")
        return torch.device("cpu")
    raise ValueError("No se detecta CUDA. Revisa doctor y PyTorch; para depurar, añade --allow-cpu.")


class WeeklyDataset(Dataset):
    def __init__(self, samples, directory, mean, std):
        self.samples, self.directory = samples, Path(directory)
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        images, masks = [], []
        for relative in sample["paths"]:
            with np.load(self.directory / relative, allow_pickle=False) as z:
                images.append(z["reflectance"].astype(np.float32))
                masks.append(z["valid"].astype(bool))
        image = torch.from_numpy(np.stack(images, axis=1))  # C,T,H,W
        valid = torch.from_numpy(np.stack(masks))            # T,H,W
        x = (image * 10000.0 - self.mean) / self.std
        x = torch.where(valid.unsqueeze(0), x, 0.0)
        temporal = []
        for value in sample["dates"]:
            d = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            temporal.append([d.year, d.timetuple().tm_yday])
        return {"x": x, "valid": valid, "time": torch.tensor(temporal, dtype=torch.float32),
                "location": torch.tensor([sample["lat"], sample["lon"]], dtype=torch.float32), "index": i}
