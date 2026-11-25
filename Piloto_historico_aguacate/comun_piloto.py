"""Rutas, firmas y validaciones compartidas; no contiene credenciales."""

from __future__ import annotations

from utilidades import leer_json, firma_objeto, sha256, cargar_configuracion
from pathlib import Path
import numpy as np
import pandas as pd


VERSION_PLAN = 1


def cargar_config(ruta_ingesta, ruta_piloto=None):
    ingesta = Path(ruta_ingesta).resolve()
    raiz = ingesta.parent
    if not ingesta.is_file():
        raise FileNotFoundError(
            f"No existe {ingesta}. Ejecuta desde la carpeta principal de tu proyecto."
        )
    cfg = cargar_configuracion(ingesta)
    piloto = leer_json(ruta_piloto) if ruta_piloto else cfg["piloto"]
    municipios = piloto["municipios_piloto"]
    contexto = piloto["anios_contexto"]
    consulta = piloto["anio_consulta"]
    if (
        len(set(municipios)) != len(municipios)
        or not municipios
        or any(
            not isinstance(c, str) or len(c) != 5 or not c.isdigit() for c in municipios
        )
    ):
        raise ValueError(
            "municipios_piloto debe contener claves de cinco dígitos sin repetir."
        )
    if (
        not contexto
        or any(not isinstance(a, int) for a in contexto + [consulta])
        or len(set(contexto)) != len(contexto)
    ):
        raise ValueError("Los años de contexto deben ser enteros únicos.")
    if not 2020 <= min(contexto) <= max(contexto) < consulta <= 2025:
        raise ValueError(
            "El contexto debe preceder a la consulta y estar entre 2020 y 2025. 2026 queda reservado."
        )
    if (
        piloto["mes"] != 11
        or piloto["primer_dia_contexto"] != 1
        or piloto["ultimo_dia_contexto"] != 25
        or piloto["dias_pronostico"] != 5
    ):
        raise ValueError(
            "Este piloto fija noviembre 1–25 como contexto y noviembre 26–30 como consulta."
        )
    if (
        piloto["tamanio_parche_px"] != 64
        or piloto["observaciones_sentinel_por_caso"] < 2
    ):
        raise ValueError(
            "Se requieren parches 64×64 a 10 m y al menos dos fechas reales por caso."
        )
    if not 0 <= piloto["fraccion_clara_minima"] <= 1:
        raise ValueError("Umbral de calidad fuera de 0–1.")
    if (
        piloto["max_descargas_sentinel_por_caso"]
        < piloto["observaciones_sentinel_por_caso"]
    ):
        raise ValueError(
            "El límite de descargas es menor que las observaciones requeridas."
        )
    if (
        not isinstance(piloto["tabicl_estimadores"], int)
        or piloto["tabicl_estimadores"] < 1
    ):
        raise ValueError("tabicl_estimadores debe ser un entero positivo.")
    if cfg.get("dispositivo", "cpu") not in ["cpu", "cuda"]:
        raise ValueError("Dispositivo debe ser cpu o cuda.")
    for nombre in ["modelos.py", "recursos_modelos.py", "preparacion.py"]:
        if not (raiz / nombre).is_file():
            raise FileNotFoundError(
                f"Falta {nombre} en {raiz}. Mantén la carpeta del piloto dentro del proyecto de ingesta."
            )
    return cfg, piloto


def ubicaciones(cfg, plan):
    raiz = Path(cfg["_raiz"])
    ident = plan["firma_plan"][:12]
    return raiz / "datos_piloto" / ident, raiz / "resultados_piloto" / ident


def carpeta_caso(cfg, plan, caso):
    datos, salida = ubicaciones(cfg, plan)
    return datos / caso["id"], salida / caso["id"]


def validar_lluvia(df, caso):
    if list(df.columns) != ["timestamp", "precipitacion_mm"]:
        raise ValueError("La lluvia del caso debe tener timestamp y precipitacion_mm.")
    fechas = pd.DatetimeIndex(pd.to_datetime(df.timestamp, utc=True)).tz_convert(None)
    esperadas = pd.date_range(caso["inicio_contexto"], caso["fin_contexto"], freq="D")
    if not fechas.equals(esperadas):
        raise ValueError(
            f"Fechas de lluvia incompletas o duplicadas en {caso['id']}; no se imputan ceros."
        )
    valores = pd.to_numeric(df.precipitacion_mm, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError("Lluvia observada no finita o negativa.")
    return fechas, valores


def firma_entradas(cfg, plan):
    archivos = {}
    for c in plan["casos"]:
        datos, _ = carpeta_caso(cfg, plan, c)
        meta = leer_json(datos / "sentinel2" / "secuencia.json")
        if meta["firma_plan"] != plan["firma_plan"]:
            raise ValueError("Secuencia Sentinel de otro plan.")
        for archivo in [
            datos / "lluvia.csv",
            datos / "sentinel2" / "secuencia.json",
            *[datos / "sentinel2" / n for n in meta["archivos"]],
        ]:
            archivos[str(archivo.relative_to(ubicaciones(cfg, plan)[0]))] = sha256(
                archivo
            )
    return firma_objeto(archivos)
