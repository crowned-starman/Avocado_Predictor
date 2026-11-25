"""Cobertura de cada recuadro por la unión municipal, calculada en m²."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import make_valid, segmentize
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from utilidades import guardar_json, leer_json, proyectar

AREA = CRS.from_proj4(
    "+proj=laea +lat_0=19.2 +lon_0=-101.8 +datum=WGS84 +units=m +no_defs"
)

# Selección roja del mapa del usuario. Se usa en configuraciones anteriores
# que aún no contienen cortes_preseleccionados, conservando sus demás ajustes.
RECUADROS_MAPA = [
    27, 43, 44, 45, 46, 47, 48, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 88, 89, 90, 91, 95, 96, 97,
    98, 99, 105, 106, 107, 108, 112, 113, 114, 115, 116, 122, 125, 126, 128,
]


def validar_umbral(valor):
    if isinstance(valor, bool) or not math.isfinite(float(valor)) or not 0 < float(valor) <= 1:
        raise ValueError("fraccion_minima_aoi debe ser mayor que 0 y como máximo 1.")
    return float(valor)


def geometria_recuadro(fila, crs):
    """Densificar antes de reproyectar evita aproximar bordes curvos por cuatro puntos."""
    origen = CRS.from_user_input(crs)
    g = box(*(float(fila[k]) for k in ("xmin", "ymin", "xmax", "ymax")))
    paso = 0.005 if origen.is_geographic else 500
    return proyectar(segmentize(g, paso), origen, 4326)


def cargar_malla(ruta, metadatos=None):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta}. Usa el CSV original de tu carpeta datos/sentinel2; "
            "no se reconstruye la cuadrícula desde una imagen."
        )
    t = pd.read_csv(ruta, encoding="utf-8-sig")
    requeridos = {"corte", "xmin", "ymin", "xmax", "ymax"}
    if not requeridos <= set(t.columns):
        raise ValueError(f"El CSV necesita las columnas {sorted(requeridos)}.")
    for col in requeridos:
        t[col] = pd.to_numeric(t[col], errors="raise")
    if t.empty or not np.isfinite(t[list(requeridos)].to_numpy(dtype=float)).all():
        raise ValueError("Malla vacía o con números no finitos.")
    if t.corte.duplicated().any() or (t.corte % 1 != 0).any():
        raise ValueError("Los números de corte deben ser enteros únicos.")
    if ((t.xmin >= t.xmax) | (t.ymin >= t.ymax)).any():
        raise ValueError("Hay extensiones de recuadro inválidas.")
    t["corte"] = t.corte.astype(int)
    valores = t.crs.dropna().astype(str).unique() if "crs" in t else []
    if len(valores) > 1:
        raise ValueError("La malla mezcla CRS; debe revisarse antes de filtrar.")
    meta = leer_json(metadatos) if metadatos and Path(metadatos).is_file() else {}
    crs = valores[0] if len(valores) else meta.get("crs")
    if not crs:
        raise ValueError("Falta el CRS: sube también metadatos.json o añade la columna crs al CSV.")
    crs = CRS.from_user_input(crs)
    t["crs"] = crs.to_string()
    return t, crs


def cargar_municipios(raiz, claves):
    origen = Path(raiz) / "Delimitacion_rendimiento_aguacate/fuentes/michoacan_inegi_original.json"
    documento = leer_json(origen)
    municipios = []
    for f in documento["features"]:
        props = f["properties"]
        g = make_valid(shape(f["geometry"]))
        crs = f["geometry"].get("crs", {}).get("properties", {}).get("name")
        if not crs:
            raise ValueError("El catálogo municipal no indica su CRS.")
        municipios.append((props["cvegeo"], props["nomgeo"], proyectar(g, crs, 4326)))
    if not claves or not set(claves) <= {c for c, _, _ in municipios}:
        raise ValueError("La selección municipal contiene claves ausentes o está vacía.")
    return municipios


def calcular_cobertura(tabla, crs, municipios, claves, umbral=0.85, cortes_preseleccionados=None):
    umbral = validar_umbral(umbral)
    union = unary_union([make_valid(g) for c, _, g in municipios if c in set(claves)])
    if union.is_empty:
        raise ValueError("El área de interés está vacía.")
    union_m = proyectar(union, 4326, AREA)
    t = tabla.copy()
    if cortes_preseleccionados is not None:
        if (not isinstance(cortes_preseleccionados, list) or not cortes_preseleccionados or
                any(not isinstance(c, int) or isinstance(c, bool) for c in cortes_preseleccionados) or
                len(set(cortes_preseleccionados)) != len(cortes_preseleccionados)):
            raise ValueError("cortes_preseleccionados debe ser una lista de enteros únicos no vacía.")
        faltantes = set(cortes_preseleccionados) - set(t.corte.astype(int))
        if faltantes:
            raise ValueError(f"La malla no contiene estos recuadros de la selección previa: {sorted(faltantes)}. "
                             "Sube el CSV de la misma cuadrícula del mapa; no se sustituirán sus coordenadas.")
    t["incluido_seleccion_previa"] = (t.corte.isin(cortes_preseleccionados)
                                      if cortes_preseleccionados is not None else True)
    geometria, superficies, intersecciones, fracciones = [], [], [], []
    for fila in t.to_dict("records"):
        g = geometria_recuadro(fila, crs)
        gm = proyectar(g, 4326, AREA)
        area = gm.area
        if area <= 0:
            raise ValueError("Recuadro de superficie nula.")
        inter = gm.intersection(union_m).area
        fraccion = min(1.0, max(0.0, inter / area))
        geometria.append(g)
        superficies.append(area / 10000)
        intersecciones.append(inter / 10000)
        fracciones.append(fraccion)
    t["superficie_recuadro_ha"] = superficies
    t["interseccion_geometrica_ha"] = intersecciones
    t["fraccion_aoi"] = fracciones
    t["porcentaje_aoi"] = t.fraccion_aoi * 100
    t["fraccion_minima_aoi"] = umbral
    t["seleccionado_por_territorio"] = (t.incluido_seleccion_previa &
        (t.fraccion_aoi + 1e-12 >= umbral) & (t.interseccion_geometrica_ha > 0))
    t["motivo_filtro"] = np.select(
        [~t.incluido_seleccion_previa, t.seleccionado_por_territorio],
        ["fuera_de_seleccion_previa", "conservado"], default="cobertura_inferior_al_umbral")
    return t, geometria, union


def escribir_filtro(salida, tabla, geometria, umbral):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    elegidos = tabla.seleccionado_por_territorio.to_numpy(dtype=bool)
    for nombre, contenido in [
        ("plan_territorial_completo.csv", tabla),
        ("recuadros_seleccionados.csv", tabla.loc[elegidos]),
        ("recuadros_descartados.csv", tabla.loc[~elegidos]),
    ]:
        contenido.to_csv(salida / nombre, index=False, encoding="utf-8-sig")
    guardar_json(salida / "recuadros.geojson", {
        "type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"corte": int(f.corte),
                "porcentaje_aoi": float(f.porcentaje_aoi), "seleccionado": bool(sel)},
                "geometry": mapping(g)}
            for f, g, sel in zip(tabla.itertuples(), geometria, elegidos)
        ]})
    resumen = {
        "posiciones": len(tabla), "seleccionadas": int(elegidos.sum()),
        "recuadros_seleccion_previa": tabla.loc[tabla.incluido_seleccion_previa, "corte"].astype(int).tolist(),
        "cantidad_seleccion_previa": int(tabla.incluido_seleccion_previa.sum()),
        "descartadas": int((~elegidos).sum()), "fraccion_minima_aoi": float(umbral),
        "comparacion_umbrales": {str(p): int((tabla.incluido_seleccion_previa &
                                             (tabla.fraccion_aoi + 1e-12 >= p / 100)).sum())
                                  for p in (80, 85, 90)},
        "definicion": "Área(recuadro ∩ unión de municipios seleccionados) / Área(recuadro)",
        "nota": "Superficie municipal; no porcentaje de huertas ni porcentaje libre de nubes. No se renumeran ni borran TIFF.",
    }
    guardar_json(salida / "informe_filtro.json", resumen)
    return resumen
