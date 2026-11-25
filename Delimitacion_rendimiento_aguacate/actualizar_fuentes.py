"""Descarga pública opcional. No usa cuentas ni descarga imágenes satelitales."""

import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import requests
import pandas as pd
from datos_agricolas import (
    PORTAL_AGRICOLA,
    descubrir_enlaces_agricolas,
    leer_csv_agricola,
    limpiar_agricola,
)

BASE = Path(__file__).resolve().parent
URL_INEGI = "https://gaia.inegi.org.mx/wscatgeo/v2/geo/mgem/16"


def obtener(url):
    r = requests.get(url, timeout=(20, 120))
    r.raise_for_status()
    return r.content


def main():
    salida = BASE / "fuentes"
    salida.mkdir(exist_ok=True)
    html = obtener(PORTAL_AGRICOLA).decode("utf-8")
    enlaces, diccionario = descubrir_enlaces_agricolas(html)
    anios = list(range(2020, 2026))
    if any(a not in enlaces for a in anios):
        raise ValueError("Faltan enlaces municipales para 2020–2025.")

    def leer(anio):
        data = obtener(enlaces[anio])
        with tempfile.TemporaryDirectory() as td:
            ruta = Path(td) / "fuente.csv"
            ruta.write_bytes(data)
            frame, encoding = leer_csv_agricola(ruta)
        d = limpiar_agricola(frame, anio, [16])
        return d, dict(
            anio=anio,
            url=enlaces[anio],
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_originales=len(data),
            filas_originales=len(frame),
            filas_filtradas=len(d),
            encoding=encoding,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pares = list(pool.map(leer, anios))
    detalle = pd.concat([p[0] for p in pares], ignore_index=True)
    raw = obtener(URL_INEGI)
    geo = json.loads(raw)
    if geo.get("type") != "FeatureCollection" or len(geo.get("features", [])) != 113:
        raise ValueError(
            "Respuesta municipal de Michoacán inesperada; revisar la fuente."
        )
    # El API declara EPSG:6365 en cada geometría; no se cambia a WGS84 sin transformar.
    if any(
        f["geometry"].get("crs", {}).get("properties", {}).get("name") != "EPSG:6365"
        for f in geo["features"]
    ):
        raise ValueError("Cambió el CRS oficial; revisar antes de proyectar.")
    detalle.to_csv(
        salida / "aguacate_detalle_2020_2025.csv", index=False, encoding="utf-8-sig"
    )
    (salida / "michoacan_inegi_original.json").write_bytes(raw)
    procedencia = dict(
        fecha_descarga_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        portal_agricola=PORTAL_AGRICOLA,
        diccionario=diccionario,
        archivos_agricolas=[p[1] for p in pares],
        inegi=dict(
            url=URL_INEGI,
            sha256=hashlib.sha256(raw).hexdigest(),
            crs_original="EPSG:6365",
            metadatos=geo.get("metadatos"),
        ),
        notas="Se conserva únicamente el detalle agrícola de aguacate de Michoacán; sin etiquetas 2026.",
    )
    (salida / "procedencia.json").write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Fuentes listas: {len(detalle)} filas de detalle; 113 municipios oficiales.")


if __name__ == "__main__":
    main()
