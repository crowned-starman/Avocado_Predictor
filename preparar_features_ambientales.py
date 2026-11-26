#!/usr/bin/env python3
"""Une ERA5-Land diario con resúmenes Sentinel-2 municipales anuales."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNAS_SENTINEL = [
    "CVEGEO", "Anio", "ndvi_mediana", "evi_mediana", "savi_mediana",
    "ndwi_mediana", "sentinel_escenas_validas",
]


def cargar_lluvia(raiz: Path) -> pd.DataFrame:
    archivos = sorted(raiz.rglob("lluvia_diaria_municipio.csv"))
    if not archivos:
        raise FileNotFoundError(f"No encontré lluvia_diaria_municipio.csv bajo {raiz}")
    diario = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig", dtype={"CVEGEO": str}) for p in archivos],
        ignore_index=True,
    )
    requeridas = {"CVEGEO", "fecha", "precipitacion_diaria_mm", "fraccion_area_con_dato"}
    if not requeridas <= set(diario.columns):
        raise ValueError(f"CSV ERA5-Land sin columnas requeridas: {sorted(requeridas - set(diario.columns))}")
    diario["CVEGEO"] = diario.CVEGEO.str.strip().str.zfill(5)
    diario["fecha"] = pd.to_datetime(diario.fecha, errors="raise")
    diario["precipitacion_diaria_mm"] = pd.to_numeric(diario.precipitacion_diaria_mm, errors="coerce")
    diario["fraccion_area_con_dato"] = pd.to_numeric(diario.fraccion_area_con_dato, errors="coerce")
    diario["Anio"] = diario.fecha.dt.year
    if diario.duplicated(["CVEGEO", "fecha"]).any():
        raise ValueError("ERA5-Land tiene fechas repetidas por municipio.")
    filas = []
    for (clave, anio), grupo in diario.groupby(["CVEGEO", "Anio"], sort=True):
        dias_esperados = 366 if pd.Timestamp(anio, 12, 31).dayofyear == 366 else 365
        valido = grupo.precipitacion_diaria_mm.notna() & (grupo.fraccion_area_con_dato >= 0.95)
        g = grupo.loc[valido].sort_values("fecha")
        cobertura = len(g) / dias_esperados
        if cobertura < 0.95:
            continue
        lluvia = g.precipitacion_diaria_mm.to_numpy(dtype=float)
        seco = lluvia < 1.0
        racha = maxima = 0
        for es_seco in seco:
            racha = racha + 1 if es_seco else 0
            maxima = max(maxima, racha)
        filas.append({
            "CVEGEO": clave, "Anio": int(anio),
            "precipitacion_total_anual_mm": float(lluvia.sum()),
            "dias_lluviosos_gt1mm": int((lluvia >= 1.0).sum()),
            "max_dias_secos_consecutivos": int(maxima),
            "cobertura_lluvia_pct": float(cobertura * 100),
        })
    if not filas:
        raise ValueError("No hay municipio-años ERA5-Land con al menos 95 % de cobertura diaria.")
    return pd.DataFrame(filas)


def cargar_sentinel(ruta: Path) -> pd.DataFrame:
    s2 = pd.read_csv(ruta, encoding="utf-8-sig", dtype={"CVEGEO": str})
    faltan = set(COLUMNAS_SENTINEL) - set(s2.columns)
    if faltan:
        raise ValueError(f"CSV Sentinel anual sin columnas requeridas: {sorted(faltan)}")
    s2 = s2[COLUMNAS_SENTINEL].copy()
    s2["CVEGEO"] = s2.CVEGEO.str.strip().str.zfill(5)
    s2["Anio"] = pd.to_numeric(s2.Anio, errors="raise").astype(int)
    for col in COLUMNAS_SENTINEL[2:]:
        s2[col] = pd.to_numeric(s2[col], errors="raise")
    if s2.duplicated(["CVEGEO", "Anio"]).any():
        raise ValueError("Sentinel-2 debe tener una fila resumen por municipio y año.")
    if not np.isfinite(s2[COLUMNAS_SENTINEL[2:]].to_numpy(dtype=float)).all():
        raise ValueError("Sentinel-2 contiene valores faltantes/no finitos; no se imputan.")
    if (s2.sentinel_escenas_validas <= 0).any():
        raise ValueError("Hay municipio-años sin observaciones Sentinel válidas.")
    return s2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lluvia-root", required=True, type=Path,
                        help="Carpeta generada por lluvia_historica.py (busca derivados/*/lluvia_diaria_municipio.csv).")
    parser.add_argument("--sentinel-anual", required=True, type=Path,
                        help="Resumen Sentinel-2: una fila por municipio-año con índices medianos.")
    parser.add_argument("--salida", type=Path, default=Path("datos/features_ambientales.csv"))
    args = parser.parse_args()
    lluvia = cargar_lluvia(args.lluvia_root)
    sentinel = cargar_sentinel(args.sentinel_anual)
    resultado = lluvia.merge(sentinel, on=["CVEGEO", "Anio"], how="inner", validate="one_to_one")
    if resultado.empty:
        parser.error("ERA5-Land y Sentinel-2 no tienen municipio-años en común.")
    resultado = resultado.sort_values(["CVEGEO", "Anio"]).reset_index(drop=True)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False, encoding="utf-8-sig", float_format="%.8g")
    print(f"Guardado: {args.salida} | {len(resultado)} municipio-año | "
          f"{resultado.CVEGEO.nunique()} municipios | {resultado.Anio.min()}–{resultado.Anio.max()}")


if __name__ == "__main__":
    main()

