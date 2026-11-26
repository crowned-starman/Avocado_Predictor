#!/usr/bin/env python3
"""Descarga precipitación diaria ERA5 desde Open-Meteo, sin cuenta Copernicus CDS.

Consulta puntos representativos dentro de los municipios configurados. ERA5 es
una rejilla de aproximadamente 0.25 grados (~25 km); cada valor es puntual,
no un promedio espacial del municipio ni de huertas.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

from cobertura_territorial import cargar_municipios

API = "https://archive-api.open-meteo.com/v1/archive"
MODELO = "era5"


def cargar_puntos(config_path: Path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    claves = [str(c).zfill(5) for c in cfg["territorio"]["cvegeo_seleccionados"]]
    municipios = cargar_municipios(config_path.parent, claves)
    puntos = []
    for clave, nombre, geom in municipios:
        p = geom.representative_point()
        puntos.append({"CVEGEO": clave, "municipio": nombre, "latitude": p.y, "longitude": p.x})
    return puntos


def pedir_lote(puntos, inicio: str, fin: str):
    params = {
        "latitude": ",".join(f"{p['latitude']:.6f}" for p in puntos),
        "longitude": ",".join(f"{p['longitude']:.6f}" for p in puntos),
        "start_date": inicio,
        "end_date": fin,
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "models": MODELO,
    }
    ultimo_error = None
    for intento in range(5):
        try:
            r = requests.get(API, params=params, timeout=(20, 180))
            if r.status_code == 429 or r.status_code >= 500:
                r.raise_for_status()
            r.raise_for_status()
            payload = r.json()
            respuestas = payload if isinstance(payload, list) else [payload]
            if len(respuestas) != len(puntos):
                raise ValueError(f"Open-Meteo devolvió {len(respuestas)} ubicaciones; se esperaban {len(puntos)}.")
            return respuestas, r.url
        except (requests.RequestException, ValueError) as exc:
            ultimo_error = exc
            if intento == 4:
                break
            time.sleep(min(2 ** intento, 20))
    raise RuntimeError(f"Falló la consulta a Open-Meteo tras 5 intentos: {ultimo_error}") from ultimo_error


def descargar(anios, config: Path, salida: Path, lote: int):
    puntos = cargar_puntos(config)
    if not puntos:
        raise ValueError("La configuración no tiene municipios seleccionados.")
    salida.mkdir(parents=True, exist_ok=True)
    for anio in anios:
        filas = []
        urls = []
        inicio, fin = f"{anio}-01-01", f"{anio}-12-31"
        for i in range(0, len(puntos), lote):
            grupo = puntos[i:i + lote]
            respuestas, url = pedir_lote(grupo, inicio, fin)
            urls.append(url.split("?")[0])
            for punto, respuesta in zip(grupo, respuestas):
                daily = respuesta.get("daily") or {}
                fechas = daily.get("time") or []
                lluvia = daily.get("precipitation_sum") or []
                if len(fechas) != len(lluvia):
                    raise ValueError(f"Fechas y lluvia no coinciden para CVEGEO {punto['CVEGEO']}, {anio}.")
                tabla = pd.DataFrame({"fecha": fechas, "precipitacion_diaria_mm": lluvia})
                tabla["fecha"] = pd.to_datetime(tabla["fecha"], errors="raise")
                tabla["precipitacion_diaria_mm"] = pd.to_numeric(tabla["precipitacion_diaria_mm"], errors="coerce")
                esperado = 366 if pd.Timestamp(anio, 12, 31).is_leap_year else 365
                if len(tabla) != esperado or tabla["fecha"].dt.year.ne(anio).any():
                    raise ValueError(f"Serie diaria incompleta o con años mezclados para CVEGEO {punto['CVEGEO']}, {anio}.")
                validos = tabla["precipitacion_diaria_mm"].dropna()
                if (validos < 0).any() or not all(math.isfinite(float(x)) for x in validos):
                    raise ValueError(f"Precipitación inválida para CVEGEO {punto['CVEGEO']}, {anio}.")
                for fecha, mm in zip(tabla["fecha"], tabla["precipitacion_diaria_mm"]):
                    filas.append({
                        "CVEGEO": punto["CVEGEO"], "municipio": punto["municipio"],
                        "fecha": fecha.date().isoformat(),
                        "precipitacion_diaria_mm": mm if pd.notna(mm) else None,
                        "fraccion_area_con_dato": 1.0 if pd.notna(mm) else 0.0,
                    })
        archivo = salida / f"lluvia_diaria_municipio_{anio}.csv"
        pd.DataFrame(filas).to_csv(archivo, index=False, encoding="utf-8-sig", float_format="%.8g")
        manifest = {
            "fuente": "Open-Meteo Historical Weather API",
            "api": API,
            "modelo": MODELO,
            "variable": "precipitation_sum",
            "unidad": "mm/día",
            "resolucion_modelo_aproximada": "0.25 grados (~25 km)",
            "metodo_espacial": "un punto representativo dentro de cada municipio; no es promedio municipal",
            "periodo": [inicio, fin],
            "municipios": len(puntos),
            "filas": len(filas),
            "consulta_endpoint": sorted(set(urls)),
        }
        (salida / f"manifest_{anio}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Guardado {archivo} | {len(filas)} filas | {len(puntos)} municipios")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuracion", type=Path, default=Path("configuracion.json"))
    parser.add_argument("--anios", nargs="+", type=int, default=list(range(2020, 2026)))
    parser.add_argument("--salida", type=Path, default=Path("datos/lluvia_open_meteo"))
    parser.add_argument("--lote", type=int, default=8, help="Municipios por consulta (máximo recomendado: 10).")
    args = parser.parse_args()
    if args.lote < 1 or args.lote > 10:
        parser.error("--lote debe estar entre 1 y 10.")
    if any(a < 1940 or a > pd.Timestamp.now(tz="UTC").year for a in args.anios):
        parser.error("Open-Meteo ERA5 cubre desde 1940 hasta el año actual.")
    descargar(sorted(set(args.anios)), args.configuracion, args.salida, args.lote)


if __name__ == "__main__":
    main()
