#!/usr/bin/env python3
"""Genera una malla regular de 10 km sobre los municipios seleccionados.

Las celdas solo organizan consultas y ventanas de lectura. El extractor Sentinel
aplica después una máscara con los polígonos INEGI exactos de los municipios.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from cobertura_territorial import AREA, cargar_municipios
from utilidades import sha256

TIPO_MALLA = "municipal_regular"


def crear_malla(configuracion: Path, salida: Path, tamano_km: float) -> tuple[int, Path]:
    configuracion = configuracion.resolve()
    if not math.isfinite(tamano_km) or tamano_km <= 0:
        raise ValueError("El tamaño de celda debe ser positivo y finito.")
    cfg = json.loads(configuracion.read_text(encoding="utf-8"))
    claves = sorted({str(c).zfill(5) for c in cfg["territorio"]["cvegeo_seleccionados"]})
    if not claves:
        raise ValueError("La configuración no contiene municipios seleccionados.")

    catalogo = cargar_municipios(configuracion.parent, claves)
    por_clave = {str(c).zfill(5): (nombre, geom) for c, nombre, geom in catalogo}
    faltantes = sorted(set(claves) - set(por_clave))
    if faltantes:
        raise ValueError(f"Faltan municipios en el catálogo INEGI: {faltantes}")

    aoi_wgs84 = unary_union([por_clave[c][1] for c in claves])
    aoi = aoi_wgs84  # geometrías de cargar_municipios ya están en WGS84
    from utilidades import proyectar
    aoi = proyectar(aoi, 4326, AREA)
    paso = tamano_km * 1000.0
    minx, miny, maxx, maxy = aoi.bounds
    x0, y0 = math.floor(minx / paso) * paso, math.floor(miny / paso) * paso
    nx, ny = math.ceil((maxx - x0) / paso), math.ceil((maxy - y0) / paso)
    if nx * ny > 50000:
        raise ValueError(f"La malla estimada tendría {nx * ny:,} celdas; revisa CRS y tamaño.")

    filas = []
    corte = 0
    for iy in range(ny):
        ymin = y0 + iy * paso
        for ix in range(nx):
            xmin = x0 + ix * paso
            celda = box(xmin, ymin, xmin + paso, ymin + paso)
            interseccion = celda.intersection(aoi)
            if interseccion.is_empty or interseccion.area <= 0:
                continue
            corte += 1
            filas.append({
                "corte": corte,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmin + paso,
                "ymax": ymin + paso,
                "crs": AREA.to_string(),
                "tipo_malla": TIPO_MALLA,
                "tamano_celda_m": paso,
                "fraccion_aoi": interseccion.area / celda.area,
            })
    if not filas:
        raise ValueError("La malla no produjo celdas que intersecten los municipios.")

    salida = salida.resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(salida, index=False, encoding="utf-8-sig", float_format="%.12g")
    geojson = configuracion.parent / "Delimitacion_rendimiento_aguacate/fuentes/michoacan_inegi_original.json"
    metadatos = {
        "tipo_malla": TIPO_MALLA,
        "resolucion_celda_m": paso,
        "crs": AREA.to_string(),
        "cvegeo": claves,
        "celdas": len(filas),
        "metodo": "rejilla regular alineada al CRS de área equal-area del proyecto; se retienen celdas con intersección positiva",
        "uso": "ventanas de consulta/lectura; los índices se enmascaran después con polígonos municipales exactos",
        "fuente_municipios_sha256": sha256(geojson),
    }
    ruta_meta = salida.with_name(salida.stem + "_metadatos.json")
    ruta_meta.write_text(json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(filas), ruta_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuracion", type=Path, default=Path("configuracion.json"))
    parser.add_argument("--salida", type=Path, default=Path("datos/sentinel2/malla_municipal_10km.csv"))
    parser.add_argument("--tamano-km", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.tamano_km) or args.tamano_km <= 0:
        parser.error("--tamano-km debe ser un número positivo.")
    filas, meta = crear_malla(args.configuracion, args.salida, args.tamano_km)
    print(f"Malla guardada: {args.salida} | {filas} celdas | resolución {args.tamano_km:g} km")
    print(f"Metadatos: {meta}")


if __name__ == "__main__":
    main()
