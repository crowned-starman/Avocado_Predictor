"""Operaciones compartidas por revisión, ingesta, territorio y piloto."""

from __future__ import annotations

import hashlib
import json
import os
import time
from getpass import getpass
from pathlib import Path

BANDAS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ARCHIVO_CREDENCIALES = Path(__file__).resolve().with_name("credenciales.json")
NOMBRES_CREDENCIALES = {"CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET", "CDS_API_KEY"}


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))


def guardar_json(ruta, objeto, *, compacto=False):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    opciones = {"separators": (",", ":")} if compacto else {"indent": 2}
    # Un temporal único evita colisiones entre procesos; reintentar cubre bloqueos
    # breves de antivirus/indexadores que son comunes al reemplazar archivos en Windows.
    temporal = ruta.with_name(f"{ruta.name}.{os.getpid()}.part")
    temporal.write_text(
        json.dumps(objeto, ensure_ascii=False, allow_nan=False, **opciones),
        encoding="utf-8",
    )
    for intento in range(6):
        try:
            temporal.replace(ruta)
            return
        except PermissionError:
            if intento == 5:
                raise
            time.sleep(0.2 * (2 ** intento))


def firma_objeto(valor, *, ensure_ascii=False):
    return hashlib.sha256(
        json.dumps(valor, sort_keys=True, ensure_ascii=ensure_ascii).encode()
    ).hexdigest()


def huella(valor, longitud=16):
    return firma_objeto(valor, ensure_ascii=True)[:longitud]


def sha256(ruta):
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_configuracion(ruta):
    ruta = Path(ruta).resolve()
    cfg = leer_json(ruta)
    for clave in ["carpeta_datos", "carpeta_salida", "carpeta_modelos"]:
        p = Path(cfg[clave]).expanduser()
        cfg[clave] = str(p if p.is_absolute() else ruta.parent / p)
    cfg["_raiz"] = str(ruta.parent)
    cfg["_configuracion"] = str(ruta)
    if cfg.get("dispositivo", "cpu") not in ["cpu", "cuda"]:
        raise ValueError("dispositivo debe ser cpu o cuda.")
    return cfg


def credencial_configurada(nombre):
    """Archivo local primero; entorno después. Nunca imprime ni serializa las claves."""
    if nombre not in NOMBRES_CREDENCIALES:
        raise ValueError("Nombre de credencial no admitido.")
    valores = {}
    if ARCHIVO_CREDENCIALES.is_file():
        try:
            valores = leer_json(ARCHIVO_CREDENCIALES)
        except (ValueError, UnicodeError):
            raise ValueError(
                "credenciales.json no es JSON válido. Revisa las comillas dobles y las comas."
            ) from None
        if not isinstance(valores, dict) or not set(valores) <= NOMBRES_CREDENCIALES:
            raise ValueError(
                "credenciales.json debe contener solo CDSE_CLIENT_ID, CDSE_CLIENT_SECRET y CDS_API_KEY."
            )
        if any(not isinstance(v, str) for v in valores.values()):
            raise ValueError(
                "Los valores de credenciales.json deben ser textos entre comillas dobles."
            )
    return valores.get(nombre, "").strip() or os.environ.get(nombre, "").strip()


def leer_credencial(nombre):
    valor = (
        credencial_configurada(nombre)
        or getpass(f"{nombre} (entrada oculta): ").strip()
    )
    if not valor:
        raise ValueError(f"Falta {nombre}.")
    return valor


def proyectar(geometria, origen, destino):
    from pyproj import Transformer
    from shapely import transform

    return transform(
        geometria,
        Transformer.from_crs(origen, destino, always_xy=True).transform,
        interleaved=False,
    )


def descargar_publico(url, ruta, max_bytes=50_000_000):
    import requests

    ruta = Path(ruta)
    if ruta.exists():
        return ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".part")
    try:
        with requests.get(url, timeout=(20, 90), stream=True) as respuesta:
            respuesta.raise_for_status()
            total = 0
            with temporal.open("wb") as f:
                for bloque in respuesta.iter_content(256 * 1024):
                    total += len(bloque)
                    if total > max_bytes:
                        raise ValueError("Respuesta mayor al límite de esta prueba.")
                    f.write(bloque)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)
    return ruta
