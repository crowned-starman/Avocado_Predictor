"""Inferencia con los pesos reales. No entrena y no carga el tercer modelo."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "TORCHDYNAMO_DISABLE", "1"
)  # Modo eager: evita exigir compilador C++ en Windows.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import Affine

from preparacion import BANDAS, escribir_json, huella_config
from recursos_modelos import CHRONOS_REVISION, CHRONOS_SHA256, TAMRF_ARCHIVOS, sha256


def dispositivo(cfg):
    dev = cfg.get("dispositivo", "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Se configuró cuda, pero PyTorch no encuentra CUDA. Cambia dispositivo a cpu."
        )
    hilos = int(cfg.get("hilos_cpu", 2))
    if hilos < 1:
        raise ValueError("hilos_cpu debe ser positivo.")
    torch.set_num_threads(hilos)
    return torch.device(dev)


def comprobar_preparacion(cfg, nombre, seccion):
    archivo = Path(cfg["carpeta_salida"]) / "preparados" / f"{nombre}_metadatos.json"
    if not archivo.is_file():
        raise FileNotFoundError(
            "Faltan datos preparados. Ejecuta primero: python prueba_ingesta.py preparar"
        )
    info = json.loads(archivo.read_text(encoding="utf-8"))
    if info["huella_config"] != huella_config(cfg, seccion):
        raise ValueError(
            "Cambió la configuración de los datos. Ejecuta preparar antes de probar."
        )
    return info


def cargar_tamrf(cfg, dev):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    base = Path(cfg["carpeta_modelos"])
    for carpeta in (base / "tamrfsits" / "src", base / "esrgan"):
        if not carpeta.is_dir():
            raise FileNotFoundError(
                "Falta código de TAMRF. Ejecuta: python prueba_ingesta.py descargar-modelos"
            )
        sys.path.insert(0, str(carpeta.resolve()))
    for nombre, (_, firma) in TAMRF_ARCHIVOS.items():
        ruta = base / "tamrf" / nombre
        if not ruta.is_file() or sha256(ruta) != firma:
            raise ValueError(
                f"Falta el archivo oficial o no coincide su SHA256: {ruta}. Ejecuta descargar-modelos."
            )
    config = OmegaConf.load(base / "tamrf" / "config.yaml")
    # Instanciar solo encoder/decoder oficiales evita dependencias del entrenamiento y sus pérdidas.
    encoder = instantiate(config.training_module.training_module.encoder)
    decoder = instantiate(config.training_module.training_module.decoder)
    checkpoint = torch.load(
        base / "tamrf" / "968639_s1t3.ckpt", map_location="cpu", weights_only=True
    )
    for nombre, modulo in (("encoder", encoder), ("decoder", decoder)):
        prefijo = f"the_modules.{nombre}."
        pesos = {
            k[len(prefijo) :].replace("._orig_mod", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(prefijo)
        }
        if not pesos:
            raise ValueError(f"El checkpoint no contiene pesos para {nombre}.")
        # Fallar ante incompatibilidades, nunca continuar con capas inicializadas al azar.
        modulo.load_state_dict(pesos, strict=True)
        modulo.to(dev).eval()
    media = torch.tensor(
        list(config.datamodule.hr_mean), dtype=torch.float32, device=dev
    ).reshape(1, 1, 10, 1, 1)
    std = torch.tensor(
        list(config.datamodule.hr_std), dtype=torch.float32, device=dev
    ).reshape(1, 1, 10, 1, 1)
    if not torch.all(std > 0) or int(config.datamodule.nb_hr_channels) != 10:
        raise ValueError(
            "La configuración de normalización no coincide con diez canales Sentinel-2."
        )
    return encoder, decoder, media, std, int(config.datamodule.nb_lr_channels)


def probar_tamrf(cfg: dict) -> dict:
    info = comprobar_preparacion(cfg, "sentinel2", "sentinel2")
    dev = dispositivo(cfg)
    with np.load(
        Path(cfg["carpeta_salida"]) / "preparados" / "sentinel2_entrada.npz",
        allow_pickle=False,
    ) as z:
        if z["bandas"].tolist() != BANDAS:
            raise ValueError("Orden de bandas incompatible con TAMRF.")
        x = torch.from_numpy(z["reflectancia"].copy()).to(dev)
        tiempos = torch.from_numpy(z["dias_desde_origen"].copy()).to(dev)
        consulta = torch.from_numpy(z["dias_consulta"].copy()).to(dev)
    encoder, decoder, media, std, canales_lr = cargar_tamrf(cfg, dev)
    salida = predecir_tamrf(
        encoder, decoder, media, std, canales_lr, x, tiempos, consulta
    )
    arreglo = salida.cpu().numpy()
    carpeta = Path(cfg["carpeta_salida"]) / "tamrf"
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        carpeta / "salida_tamrf.npz",
        reflectancia_estimada=arreglo,
        bandas=np.array(BANDAS),
        fecha=np.array([info["consulta_utc"]]),
    )
    with rasterio.open(
        carpeta / "prueba_salida_10_bandas.tif",
        "w",
        driver="GTiff",
        width=arreglo.shape[-1],
        height=arreglo.shape[-2],
        count=10,
        dtype="float32",
        crs=info["crs"],
        transform=Affine(*info["transform"]),
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(arreglo[0, 0])
        for i, b in enumerate(BANDAS, 1):
            dst.set_band_description(i, b)
        dst.update_tags(
            fecha_utc=info["consulta_utc"],
            tipo="estimacion de prueba sin validacion de precision",
            unidades="reflectancia",
        )
    resumen = {
        "modelo": "TAMRF baseline 968639_s1t3",
        "pesos_preentrenados": True,
        "carga_estricta_de_pesos": True,
        "forma_entrada": list(x.shape),
        "forma_salida": list(salida.shape),
        "dispositivo": str(dev),
        "fechas_reales_de_entrada": info["fechas_utc"],
        "fecha_consulta": info["consulta_utc"],
        "minimo_salida": float(salida.min()),
        "maximo_salida": float(salida.max()),
        "salida_finita": True,
        "normalizacion": "(reflectancia-media_checkpoint)/std_checkpoint; no dividir de nuevo entre 10000",
        "modalidad": "Solo Sentinel-2; Landsat ausente (None)",
        "comprobacion": "Entrada aceptada y salida calculada con pesos reales. No demuestra precisión futura con una sola fecha.",
    }
    escribir_json(carpeta / "informe_tamrf.json", resumen)
    logging.info("TAMRF recibió %s y devolvió %s.", tuple(x.shape), tuple(salida.shape))
    del encoder, decoder, x, salida
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return resumen


def probar_chronos(cfg: dict) -> dict:

    info = comprobar_preparacion(cfg, "chronos2", "era5_land")
    dev = dispositivo(cfg)
    tabla = pd.read_csv(
        Path(cfg["carpeta_salida"]) / "preparados" / "chronos2_entrada.csv",
        parse_dates=["timestamp"],
    )
    modelo = cargar_chronos(cfg, dev)
    # La serie está en mm por día; el modelo realiza su normalización interna.
    salida = modelo.predict_df(
        tabla,
        prediction_length=info["pasos_a_predecir"],
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    columnas = ["predictions", "0.1", "0.5", "0.9"]
    if not set(columnas).issubset(salida.columns):
        raise ValueError(
            f"La API de Chronos-2 devolvió columnas inesperadas: {list(salida.columns)}"
        )
    if (
        len(salida) != info["numero_series"] * info["pasos_a_predecir"]
        or not np.isfinite(salida[columnas].values).all()
    ):
        raise ValueError("Salida Chronos-2 de tamaño incorrecto o no finita.")
    if set(salida.item_id) != set(tabla.item_id):
        raise ValueError(
            "Chronos-2 devolvió identificadores de series distintos a la entrada."
        )
    for nombre, g in salida.groupby("item_id"):
        ultima = tabla.loc[tabla.item_id == nombre, "timestamp"].max()
        esperadas = pd.date_range(
            ultima + pd.Timedelta(days=1), periods=info["pasos_a_predecir"], freq="D"
        )
        if not pd.DatetimeIndex(g.timestamp).equals(esperadas):
            raise ValueError(
                "Las fechas pronosticadas no son los días posteriores al contexto de esa serie."
            )
    # Conservar la salida cruda, incluso cuantiles negativos. Añadir una versión física sin sobrescribirlos.
    salida["prediccion_mm_no_negativa"] = salida["predictions"].clip(lower=0)
    carpeta = Path(cfg["carpeta_salida"]) / "chronos2"
    carpeta.mkdir(parents=True, exist_ok=True)
    salida.to_csv(carpeta / "salida_chronos2.csv", index=False)
    resumen = {
        "modelo": "amazon/chronos-2",
        "revision": CHRONOS_REVISION,
        "pesos_preentrenados": True,
        "dispositivo": str(dev),
        "forma_contexto": info["forma_contexto"],
        "filas_salida": len(salida),
        "dias_de_contexto": info["dias_historicos"],
        "pasos_futuros": info["pasos_a_predecir"],
        "unidad": "mm/día",
        "salida_finita": True,
        "cuantiles": [0.1, 0.5, 0.9],
        "comprobacion": "Entrada aceptada y salida calculada con pesos reales. Tres días no validan estacionalidad ni precisión.",
    }
    escribir_json(carpeta / "informe_chronos2.json", resumen)
    logging.info(
        "Chronos-2 recibió %s y devolvió %d filas futuras.",
        info["forma_contexto"],
        len(salida),
    )
    del modelo
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return resumen


def predecir_tamrf(encoder, decoder, media, std, canales_lr, x, tiempos, consulta):
    """Mismo pase de inferencia para la prueba local y cada caso histórico."""
    from tamrfsits.core.time_series import MonoModalSITS

    if (
        x.ndim != 5
        or x.shape[0] != 1
        or x.shape[2] != 10
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        raise ValueError("TAMRF requiere float32 [1, fechas, 10, alto, ancho] finito.")
    if not torch.all(consulta > tiempos.max()):
        raise ValueError("La consulta debe ser posterior a todas las observaciones.")
    with torch.inference_mode():
        entrada = MonoModalSITS((x - media) / std, tiempos)
        latente = encoder(None, entrada)
        resultado = decoder(latente, consulta)
        if resultado.data.shape[2] != canales_lr + 10:
            raise ValueError(
                "Cantidad de canales del decoder distinta a la configuración oficial."
            )
        salida = resultado.data[:, :, canales_lr:] * std + media
    esperado = (1, consulta.shape[1], 10, x.shape[3], x.shape[4])
    if tuple(salida.shape) != esperado or not torch.isfinite(salida).all():
        raise ValueError(
            "La salida de TAMRF tiene dimensiones incorrectas o valores no finitos."
        )
    return salida


def cargar_chronos(cfg, dev):
    from chronos import Chronos2Pipeline

    base = Path(cfg["carpeta_modelos"]) / "chronos2"
    if not all(
        (base / n).is_file()
        for n in ["config.json", "model.safetensors", "procedencia.json"]
    ):
        raise FileNotFoundError("Faltan pesos de Chronos-2. Ejecuta descargar-modelos.")
    procedencia = json.loads((base / "procedencia.json").read_text())
    if (
        procedencia.get("revision") != CHRONOS_REVISION
        or sha256(base / "model.safetensors") != CHRONOS_SHA256
    ):
        raise ValueError("Chronos-2 no coincide con la revisión oficial fijada.")
    return Chronos2Pipeline.from_pretrained(
        str(base), device_map=str(dev), dtype=torch.float32
    )
