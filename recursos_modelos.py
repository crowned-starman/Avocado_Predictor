"""Descarga reproducible de pesos oficiales y del código que necesita TAMRF."""

from __future__ import annotations

from utilidades import sha256, guardar_json

import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import requests

TAMRF_COMMIT = "9989bfa4a89f7e1a13a6d45fba5246d091d539e5"
ESRGAN_COMMIT = "7b14d729454546e73de488b0a7e90355b86d43a2"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHRONOS_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
TAMRF_ZIP = "https://zenodo.org/records/17474541/files/model.zip?download=1"
TAMRF_ARCHIVOS = {
    "config.yaml": (
        "model/baseline/hydra_config/config.yaml",
        "070b8ca473db9ca55666c42d9f0a4a2bf0cde2533f650c6171d2b22034cc2da5",
    ),
    "968639_s1t3.ckpt": (
        "model/baseline/968639_s1t3.ckpt",
        "bc305b8b9b4fb9c8da641667e7e60b90206898b84621e07838960e7d69f119b2",
    ),
}


def obtener_codigo(repo: str, commit: str, carpeta: Path, prefijo: str):
    marca = carpeta / "revision_descargada.txt"
    if marca.is_file() and marca.read_text().strip() == commit:
        return
    carpeta.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{repo}/zip/{commit}"
    logging.info("Descargando código oficial de %s (%s).", repo, commit[:10])
    with tempfile.TemporaryDirectory(dir=carpeta.parent) as tmp:
        tmp = Path(tmp)
        paquete = tmp / "codigo.zip"
        with requests.get(url, timeout=(20, 90), stream=True) as respuesta:
            respuesta.raise_for_status()
            n = 0
            with paquete.open("wb") as f:
                for bloque in respuesta.iter_content(1024 * 1024):
                    n += len(bloque)
                    if n > 200_000_000:
                        raise ValueError("Archivo de código mayor al límite esperado.")
                    f.write(bloque)
        destino = tmp / "extraido"
        destino.mkdir()
        with zipfile.ZipFile(paquete) as z:
            for info in z.infolist():
                partes = PurePosixPath(info.filename).parts
                if (
                    len(partes) < 2
                    or ".." in partes
                    or PurePosixPath(info.filename).is_absolute()
                ):
                    continue
                relativo = PurePosixPath(*partes[1:])
                texto = relativo.as_posix()
                if not (
                    texto.startswith(prefijo + "/")
                    or texto in ("LICENSE", "LICENSE.txt")
                ):
                    continue
                if info.is_dir():
                    continue
                salida = destino.joinpath(*relativo.parts)
                salida.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as entrada, salida.open("wb") as f:
                    shutil.copyfileobj(entrada, f)
        if not (destino / prefijo).is_dir():
            raise ValueError(
                "La estructura del código oficial cambió o la descarga quedó incompleta."
            )
        if carpeta.exists():
            shutil.rmtree(carpeta)
        destino.replace(carpeta)
        marca.write_text(commit + "\n", encoding="utf-8")


def obtener_tamrf(cfg: dict) -> dict:
    from remotezip import RemoteZip

    base = Path(cfg["carpeta_modelos"])
    obtener_codigo(
        "Evoland-Land-Monitoring-Evolution/tamrfsits",
        TAMRF_COMMIT,
        base / "tamrfsits",
        "src/tamrfsits",
    )
    obtener_codigo("jmichel-otb/esrgan", ESRGAN_COMMIT, base / "esrgan", "esrgan")
    pesos = base / "tamrf"
    pesos.mkdir(parents=True, exist_ok=True)
    pendientes = [
        nombre
        for nombre, (_, firma) in TAMRF_ARCHIVOS.items()
        if not (pesos / nombre).is_file() or sha256(pesos / nombre) != firma
    ]
    if pendientes:
        logging.info(
            "Descargando solo el TAMRF baseline y su configuración desde Zenodo (~80 MB comprimidos)."
        )
        # El ZIP oficial incluye muchas ablaciones. HTTP Range permite extraer solo el baseline.
        try:
            with RemoteZip(TAMRF_ZIP, timeout=90) as z:
                for nombre in pendientes:
                    miembro, firma = TAMRF_ARCHIVOS[nombre]
                    temporal = (pesos / nombre).with_suffix(".part")
                    try:
                        with z.open(miembro) as entrada, temporal.open("wb") as f:
                            shutil.copyfileobj(entrada, f, length=1024 * 1024)
                        if sha256(temporal) != firma:
                            raise ValueError(
                                f"No coincide SHA256 del archivo oficial {nombre}."
                            )
                        temporal.replace(pesos / nombre)
                    finally:
                        temporal.unlink(missing_ok=True)
        except Exception as error:
            raise RuntimeError(
                "No se completó la descarga parcial de TAMRF. Se puede reintentar. "
                "Alternativa: descargar model.zip de Zenodo 17474541 y copiar "
                "model/baseline/968639_s1t3.ckpt y model/baseline/hydra_config/config.yaml "
                f"a {pesos}. No se sustituyen por pesos aleatorios."
            ) from error
    info = {
        "modelo": "TAMRF baseline s1t3",
        "checkpoint": str(pesos / "968639_s1t3.ckpt"),
        "config": str(pesos / "config.yaml"),
        "commit_codigo": TAMRF_COMMIT,
        "commit_esrgan": ESRGAN_COMMIT,
        "fuente_pesos": TAMRF_ZIP,
        "sha256_checkpoint": TAMRF_ARCHIVOS["968639_s1t3.ckpt"][1],
    }
    (pesos / "procedencia.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


TABICL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICL_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
TABICL_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"


def obtener_pesos_huggingface(repo_id, revision, ruta, archivos, peso_principal):
    """Descarga fijada por revisión y comprobación SHA256 compartidas por ambos modelos."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    ruta = Path(ruta)
    completo = all((ruta / n).is_file() for n in archivos)
    valido = completo and all(
        firma is None or sha256(ruta / n) == firma for n, firma in archivos.items()
    )
    if not valido:
        logging.info("Descargando pesos oficiales de %s.", repo_id)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(archivos),
            local_dir=str(ruta),
            force_download=completo,
        )
    firmas = {}
    for nombre, esperada in archivos.items():
        firma = sha256(ruta / nombre)
        if esperada is not None and firma != esperada:
            raise ValueError(f"SHA256 de {repo_id}/{nombre} distinto al fijado.")
        firmas[nombre] = firma
    info = {
        "modelo": repo_id,
        "revision": revision,
        "ruta": str(ruta),
        "sha256_pesos": firmas[peso_principal],
        "sha256_archivos": firmas,
    }
    guardar_json(ruta / "procedencia.json", info)
    return info


def obtener_chronos(cfg):
    return obtener_pesos_huggingface(
        "amazon/chronos-2",
        CHRONOS_REVISION,
        Path(cfg["carpeta_modelos"]) / "chronos2",
        {"config.json": None, "model.safetensors": CHRONOS_SHA256},
        "model.safetensors",
    )


def obtener_tabicl(cfg):
    return obtener_pesos_huggingface(
        "jingang/TabICL",
        TABICL_REVISION,
        Path(cfg["carpeta_modelos"]) / "tabiclv2",
        {TABICL_CHECKPOINT: TABICL_SHA256},
        TABICL_CHECKPOINT,
    )
