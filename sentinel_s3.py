"""Histórico Sentinel a resolución nativa: COG público → RAM del ejecutor → S3.

Ejecutar la transferencia en Colab o en una máquina de la nube. En disco solo
se escriben tablas, catálogos y planes; ningún TIFF se escribe al disco.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.shutil import copy as copiar_raster
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from pyproj import CRS

from cobertura_territorial import (
    AREA, RECUADROS_MAPA, calcular_cobertura, cargar_malla, cargar_municipios, escribir_filtro,
)
from datos_sentinel import ASSETS, buscar_escenas, fecha_escena
from utilidades import guardar_json, leer_json, proyectar, sha256

RAIZ = Path(__file__).resolve().parent
VERSION = "sentinel-nativo-s3-v1"
RESOLUCIONES = {b: 10 if b in ("B02", "B03", "B04", "B08") else 20 for b in ASSETS}
TIPO_AOI = "union_municipios_seleccionados_no_huertas"
VERSION_CALIDAD = "scl-recuadro-20m-v1"


def firma(objeto):
    return hashlib.sha256(json.dumps(objeto, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def resolver(raiz, valor):
    p = Path(valor).expanduser()
    return p.resolve() if p.is_absolute() else (Path(raiz) / p).resolve()


def configuracion(ruta, anio=None, mes_inicial=None, mes_final=None):
    ruta = Path(ruta).resolve()
    cfg = leer_json(ruta)
    # Completar configuraciones anteriores sin reemplazar rutas, bucket ni claves.
    cfg.setdefault("sentinel_s3", {})
    defaults = {
        "bucket": "CAMBIAR_POR_TU_BUCKET", "region": "us-west-2",
        "prefijo": "aguacate/sentinel2", "fecha_inicial": "2020-01-01",
        "fecha_final": "2025-12-31", "archivo_malla": "", "archivo_metadatos_malla": "",
        "carpeta_trabajo": "resultados_sentinel_s3", "max_pixeles_por_banda": 16_000_000,
        "clases_scl_claras": [4, 5, 6], "mes_inicial": 1, "mes_final": 12,
        "seleccion_temporal": {"modo": "mejor_por_bloque", "fraccion_valida_minima": 0.95,
                               "fraccion_clara_minima": 0.80, "usar_cache_s3": True},
    }
    for clave, valor in defaults.items():
        cfg["sentinel_s3"].setdefault(clave, valor)
    cfg.setdefault("territorio", {})
    cfg["territorio"].setdefault("fraccion_minima_aoi", 0.85)
    if cfg["territorio"].get("cortes_preseleccionados") is None:
        cfg["territorio"]["cortes_preseleccionados"] = RECUADROS_MAPA.copy()
    cfg["_raiz"] = ruta.parent
    anio = anio if anio is not None else cfg["sentinel_s3"].get("anio_descarga")
    cfg["_trabajo"] = resolver(ruta.parent, cfg["sentinel_s3"]["carpeta_trabajo"])
    if anio is not None:
        if isinstance(anio, bool) or not isinstance(anio, int) or not 2020 <= anio <= 2025:
            raise ValueError("El año debe ser un entero entre 2020 y 2025. 2026 se reserva para evaluación.")
        cfg["sentinel_s3"].update(anio_descarga=anio, fecha_inicial=f"{anio}-01-01", fecha_final=f"{anio}-12-31")
        cfg["_trabajo"] /= f"anio_{anio}"
    ini, fin = [dt.date.fromisoformat(cfg["sentinel_s3"][k]) for k in ("fecha_inicial", "fecha_final")]
    if ini > fin or ini < dt.date(2020, 1, 1) or fin > dt.date(2025, 12, 31):
        raise ValueError("Esta ingesta es para desarrollo 2020–2025. 2026 queda fuera para evaluación.")
    for clave, valor in (("mes_inicial", mes_inicial), ("mes_final", mes_final)):
        if valor is not None:
            cfg["sentinel_s3"][clave] = valor
    rango_meses(cfg)
    return cfg


def rango_meses(cfg):
    """La asignación de trabajo no altera el periodo ni la identidad del plan anual."""
    n = cfg["sentinel_s3"]
    a, b = n.get("mes_inicial", 1), n.get("mes_final", 12)
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (a, b)) or not 1 <= a <= b <= 12:
        raise ValueError("Elige meses enteros con 1 <= mes_inicial <= mes_final <= 12.")
    if (a, b) != (1, 12) and n["fecha_inicial"][:4] != n["fecha_final"][:4]:
        raise ValueError("Para dividir por meses, selecciona un solo año con anio_descarga.")
    return a, b


def consulta_catalogo(cfg, seleccion):
    n = cfg["sentinel_s3"]
    return {"version": VERSION, "seleccion_firma": seleccion["firma"],
            "fecha_inicial": n["fecha_inicial"], "fecha_final": n["fecha_final"],
            "nubosidad_maxima_escena": None, "bandas": list(ASSETS)}


def ids_calidad(cfg, catalogo_id):
    calidad_id = firma({"catalogo": catalogo_id, "algoritmo": VERSION_CALIDAD,
                       "clases": cfg["sentinel_s3"].get("clases_scl_claras", [4, 5, 6])})[:20]
    return calidad_id, firma({"calidad_id": calidad_id, "parametros": parametros_calidad(cfg)})[:20]


def ruta_malla(cfg):
    nube = cfg["sentinel_s3"]
    if nube.get("archivo_malla"):
        ruta = resolver(cfg["_raiz"], nube["archivo_malla"])
    else:
        s2 = resolver(cfg["_raiz"], cfg["carpeta_datos"]) / "sentinel2"
        rutas = [s2 / "cortes_georreferenciados.csv", s2 / "plan_recuadros.csv"]
        ruta = next((p for p in rutas if p.is_file()), rutas[0])
    meta = (resolver(cfg["_raiz"], nube["archivo_metadatos_malla"])
            if nube.get("archivo_metadatos_malla") else ruta.parent / "metadatos.json")
    return ruta, meta


def dibujar_filtro(municipios, claves, tabla, geometria, destino, umbral):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    fig, ax = plt.subplots(figsize=(12, 9))
    for c, _, g in municipios:
        for pol in getattr(g, "geoms", [g]):
            if pol.geom_type != "Polygon":
                continue
            vertices, codigos = [], []
            for anillo in [pol.exterior, *pol.interiors]:
                coords = list(anillo.coords)
                vertices += coords
                codigos += [MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(vertices, codigos), facecolor="#84a95a" if c in claves else "#eeeeee",
                                   edgecolor="#69756c", linewidth=0.4))
    for f, g in zip(tabla.itertuples(), geometria):
        ok = f.seleccionado_por_territorio
        x, y = g.exterior.xy
        ax.plot(x, y, color="#d33d2d" if ok else "#aaaaaa", linewidth=0.85 if ok else 0.45)
        if ok:
            p = g.representative_point()
            ax.text(p.x, p.y, str(f.corte), ha="center", fontsize=6)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(19.2)))
    ax.set(xlabel="Longitud", ylabel="Latitud",
           title=f"Michoacán: recuadros con al menos {umbral:.0%} de área municipal seleccionada\n"
                 f"{int(tabla.seleccionado_por_territorio.sum())} conservados de {len(tabla)}; no es una máscara de huertas")
    ax.legend(handles=[Patch(facecolor="#84a95a", label="Municipios seleccionados"),
                       Patch(facecolor="none", edgecolor="#d33d2d", label="Recuadros conservados"),
                       Patch(facecolor="none", edgecolor="#aaaaaa", label="Recuadros descartados")], loc="lower left")
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def preparar(cfg):
    carpeta = cfg["_trabajo"]
    ruta, meta = ruta_malla(cfg)
    tabla, crs = cargar_malla(ruta, meta)
    claves = cfg["territorio"]["cvegeo_seleccionados"]
    umbral = cfg["territorio"].get("fraccion_minima_aoi", 0.85)
    municipios = cargar_municipios(cfg["_raiz"], claves)
    previos = cfg["territorio"].get("cortes_preseleccionados")
    tabla, geometria, union = calcular_cobertura(tabla, crs, municipios, claves, umbral, previos)
    resumen = escribir_filtro(carpeta, tabla, geometria, umbral)
    seleccion = []
    for f, g in zip(tabla.itertuples(), geometria):
        if f.seleccionado_por_territorio:
            seleccion.append({"corte": int(f.corte), "bbox_original": [f.xmin, f.ymin, f.xmax, f.ymax],
                              "crs_original": crs.to_string(), "fraccion_aoi": float(f.fraccion_aoi),
                              "geometry": mapping(g)})
    if not seleccion:
        raise ValueError("Ningún recuadro alcanza el umbral. Revisa informe_filtro.json y los descartados.")
    plan = {"version": VERSION, "tipo_aoi": TIPO_AOI, "cvegeo": sorted(claves),
            "cortes_preseleccionados": sorted(previos) if previos is not None else None,
            "fraccion_minima_aoi": float(umbral), "aoi": mapping(union), "recuadros": seleccion,
            "fuentes": {"csv_sha256": sha256(ruta), "municipios_sha256": sha256(
                cfg["_raiz"] / "Delimitacion_rendimiento_aguacate/fuentes/michoacan_inegi_original.json")}}
    plan["firma"] = firma(plan)
    guardar_json(carpeta / "seleccion_s3.json", plan)
    dibujar_filtro(municipios, set(claves), tabla, geometria, carpeta / "mapa_recuadros_filtrados.png", umbral)
    huella = unary_union([proyectar(shape(c["geometry"]), 4326, AREA) for c in seleccion])
    cobertura = []
    for c, nombre, g in municipios:
        if c in claves:
            gm = proyectar(g, 4326, AREA)
            cobertura.append({"CVEGEO": c, "municipio": nombre,
                              "fraccion_municipio_en_recuadros_aceptados": gm.intersection(huella).area / gm.area})
    import pandas as pd
    pd.DataFrame(cobertura).to_csv(carpeta / "cobertura_municipal_tras_filtro.csv", index=False, encoding="utf-8-sig")
    print(f"Conservados {resumen['seleccionadas']}/{resumen['posiciones']} al {umbral:.0%}. "
          f"Comparación 80/85/90 %: {resumen['comparacion_umbrales']}")
    print("Mapa:", carpeta / "mapa_recuadros_filtrados.png")
    return plan


def meses(inicial, final):
    actual = inicial.replace(day=1)
    while actual <= final:
        ultimo = actual.replace(day=calendar.monthrange(actual.year, actual.month)[1])
        yield max(inicial, actual), min(final, ultimo)
        actual = (ultimo + dt.timedelta(days=1)).replace(day=1)


def validar_seleccion(plan):
    if plan.get("version") != VERSION or plan.get("tipo_aoi") != TIPO_AOI:
        raise ValueError("Formato de selección incompatible; ejecuta preparar.")
    if plan.get("firma") != firma({k: v for k, v in plan.items() if k != "firma"}):
        raise ValueError("La selección cambió después de prepararse; ejecuta preparar.")


def recortar_item(item):
    """Conservar procedencia y radiometría, excluyendo miniaturas y otras bandas."""
    for banda, clave in ASSETS.items():
        if clave not in item.get("assets", {}) or not item["assets"][clave].get("href"):
            raise ValueError(f"A {item['id']} le falta {banda}; catálogo incompleto.")
        rb = item["assets"][clave].get("raster:bands", [{}])[0]
        if banda != "SCL" and not all(k in rb for k in ("scale", "offset")):
            raise ValueError(f"A {item['id']}/{banda} le falta escala/offset.")
    return {"id": item["id"], "collection": item["collection"], "geometry": item["geometry"],
            "properties": item["properties"], "assets": {v: item["assets"][v] for v in ASSETS.values()},
            "links": [x for x in item.get("links", []) if x.get("rel") in ("self", "license")]}


def planificar_todas(cfg, *, s3=None):
    carpeta = cfg["_trabajo"]
    seleccion = leer_json(carpeta / "seleccion_s3.json")
    validar_seleccion(seleccion)
    esperado = cfg["territorio"]
    if (sorted(esperado["cvegeo_seleccionados"]) != seleccion["cvegeo"] or
            float(esperado.get("fraccion_minima_aoi", 0.85)) != seleccion["fraccion_minima_aoi"]):
        raise ValueError("Cambió la selección o el umbral; ejecuta preparar otra vez.")
    previos = esperado.get("cortes_preseleccionados")
    if (sorted(previos) if previos is not None else None) != seleccion.get("cortes_preseleccionados"):
        raise ValueError("Cambió la lista de recuadros de partida; ejecuta preparar otra vez.")
    inicial, final = [dt.date.fromisoformat(cfg["sentinel_s3"][k]) for k in ("fecha_inicial", "fecha_final")]
    consulta = consulta_catalogo(cfg, seleccion)
    plan_id = firma(consulta)[:20]
    salida = carpeta / "catalogos" / plan_id
    salida.mkdir(parents=True, exist_ok=True)
    geometria = {c["corte"]: shape(c["geometry"]) for c in seleccion["recuadros"]}
    bbox = unary_union(list(geometria.values())).bounds
    areas = {c: proyectar(g, 4326, AREA).area for c, g in geometria.items()}
    partes, total_items, total_casos, estimacion = [], 0, 0, 0
    for ini, fin in meses(inicial, final):
        ruta = salida / f"{ini:%Y-%m}.json"
        peticion = {**consulta, "intervalo": [ini.isoformat(), fin.isoformat()], "bbox": list(bbox)}
        huella = firma(peticion)
        cache_key = f"{cfg['sentinel_s3']['prefijo'].strip('/')}/cache/catalogos/{plan_id}/{ini:%Y-%m}.json"
        if not ruta.exists() and s3 is not None:
            remoto = leer_json_s3(s3, cfg["sentinel_s3"]["bucket"], cache_key)
            if remoto is not None:
                if remoto.get("peticion_firma") != huella:
                    raise ValueError("La caché remota del catálogo no coincide con esta consulta.")
                guardar_json(ruta, remoto, compacto=True)
        if ruta.exists():
            doc = leer_json(ruta)
            if doc.get("peticion_firma") != huella:
                raise ValueError("Catálogo incompatible; no se mezclan consultas.")
        else:
            items = buscar_escenas(bbox, [ini.isoformat(), fin.isoformat()], max_cloud=None)
            compactos, tareas = [], []
            for item in items:
                g = shape(item["geometry"])
                cortes = [c for c, gc in geometria.items() if gc.intersects(g) and
                          proyectar(gc.intersection(g), 4326, AREA).area > 1]
                if cortes:
                    indice = len(compactos)
                    compactos.append(recortar_item(item))
                    tareas.extend({"item": indice, "corte": c} for c in cortes)
            doc = {"peticion_firma": huella, "consulta": peticion,
                   "items": compactos, "tareas": tareas}
            guardar_json(ruta, doc, compacto=True)
        if s3 is not None:
            guardar_s3(s3, cfg["sentinel_s3"]["bucket"], cache_key, doc)
        # Cachés verificadas nuevamente: fechas, bandas y referencias.
        for item in doc["items"]:
            if not ini <= fecha_escena(item).date() <= fin:
                raise ValueError("Hay una fecha fuera del intervalo en el catálogo.")
            recortar_item(item)
        for tarea in doc["tareas"]:
            if tarea["corte"] not in geometria or not 0 <= tarea["item"] < len(doc["items"]):
                raise ValueError("Referencia de tarea no válida.")
            # 4 bandas de 10 m + 6 de 20 m (uint16) y SCL de 20 m (uint8).
            estimacion += areas[tarea["corte"]] * (4 * 2 / 100 + 6 * 2 / 400 + 1 / 400)
        partes.append({"archivo": str(ruta.relative_to(carpeta)), "sha256": sha256(ruta),
                       "escenas": len(doc["items"]), "casos": len(doc["tareas"])})
        total_items += len(doc["items"])
        total_casos += len(doc["tareas"])
        print(f"{ini:%Y-%m}: {len(doc['items'])} escenas, {len(doc['tareas'])} pares escena–recuadro.", flush=True)
    if not total_casos and cfg["sentinel_s3"].get("seleccion_temporal", {}).get("modo") != "mejor_por_bloque":
        raise ValueError("No hay escenas para la selección y el periodo. No se descargará otra zona o fecha.")
    plan = {"version": VERSION, "plan_id": plan_id, "consulta": consulta,
            "seleccion": seleccion, "partes": partes, "escenas": total_items,
            "casos_escena_recuadro": total_casos, "archivos_banda_previstos": total_casos * len(ASSETS),
            "gib_nativos_sin_comprimir_estimados": round(estimacion / 2**30, 2),
            "nota_tamanio": "Estimación geométrica sin comprimir, sin máscaras internas/overviews/metadata. "
                           "Incluye el recuadro entero por cada escena que lo intersecta, aun si es parcial. "
                           "No es el tamaño facturado en S3. No se descartan fechas por nubosidad de escena.",
            "catalogo_completo": True}
    if cfg["sentinel_s3"].get("seleccion_temporal", {}).get("modo") == "mejor_por_bloque":
        plan["listo_para_transferir"] = False
        plan["seleccion_temporal"] = parametros_calidad(cfg)
    guardar_json(carpeta / "plan_descarga_s3.json", plan)
    print(f"Catálogo de candidatos: {total_items} escenas; {total_casos} pares escena–recuadro. "
          f"Estimación nativa: {plan['gib_nativos_sin_comprimir_estimados']} GiB sin comprimir.")
    return plan


def leer_json_s3(s3, bucket, clave):
    from botocore.exceptions import ClientError
    try:
        r = s3.get_object(Bucket=bucket, Key=clave)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    cuerpo = r["Body"]
    try:
        return json.loads(cuerpo.read().decode("utf-8"))
    finally:
        cuerpo.close()


def parametros_calidad(cfg):
    p = dict(cfg["sentinel_s3"].get("seleccion_temporal", {}))
    p.setdefault("modo", "todas")
    if p["modo"] not in ("todas", "mejor_por_bloque"):
        raise ValueError("Modo temporal desconocido.")
    p.setdefault("fraccion_valida_minima", 0.95)
    p.setdefault("fraccion_clara_minima", 0.80)
    p.setdefault("usar_cache_s3", True)
    for k in ("fraccion_valida_minima", "fraccion_clara_minima"):
        if isinstance(p[k], bool) or not math.isfinite(float(p[k])) or not 0 <= float(p[k]) <= 1:
            raise ValueError(f"{k} debe estar entre 0 y 1.")
        p[k] = float(p[k])
    if not isinstance(p["usar_cache_s3"], bool):
        raise ValueError("usar_cache_s3 debe ser true o false.")
    return p


def bloques_mes(inicio, final):
    for numero, (d0, d1) in enumerate(((1, 7), (8, 14), (15, 21),
            (22, calendar.monthrange(inicio.year, inicio.month)[1])), 1):
        a, b = inicio.replace(day=d0), inicio.replace(day=d1)
        a, b = max(a, inicio), min(b, final)
        if a <= b:
            yield numero, a, b


def clave_adquisicion(item):
    """Agrupar teselas de una adquisición, nunca satélites u órbitas distintos."""
    p = item["properties"]
    plataforma = p.get("platform") or item["id"].split("_")[0]
    toma = p.get("s2:datatake_id")
    # La versión de procesado no convierte una toma en otra observación.
    if toma:
        toma = re.sub(r"_N\d+(?:\.\d+)?$", "", str(toma))
    else:
        producto = str(p.get("s2:product_uri", ""))
        m = re.search(r"_MSIL2A_(\d{8}T\d{6})_.*?_R(\d{3})_", producto)
        toma = "_".join(m.groups()) if m else str(item["id"])
    return f"{plataforma}|{fecha_escena(item).date()}|{toma}"


def ruta_lectura_sentinel(href):
    """Resolver S3 público por HTTPS regional sin cambiar la procedencia ni sus firmas."""
    regiones = {"sentinel-s2-l2a": "eu-central-1", "sentinel-cogs": "us-west-2",
                "e84-earth-search-sentinel-data": "us-west-2"}
    for bucket, region in regiones.items():
        prefijo = f"s3://{bucket}/"
        if str(href).startswith(prefijo):
            return f"https://{bucket}.s3.{region}.amazonaws.com/{str(href)[len(prefijo):]}"
    return href


def medir_calidad(items, corte, clases=(4, 5, 6), max_pixeles=16_000_000):
    """Leer únicamente SCL y su validez; el resto de las bandas no se abre."""
    crs = CRS.from_user_input(corte["crs_original"])
    if not crs.is_projected or not math.isclose(crs.axis_info[0].unit_conversion_factor, 1):
        centro = shape(corte["geometry"]).centroid
        zona = min(60, max(1, int((centro.x + 180) // 6) + 1))
        crs = CRS.from_epsg((32600 if centro.y >= 0 else 32700) + zona)
    limites = transform_bounds(corte["crs_original"], crs, *corte["bbox_original"], densify_pts=41)
    x0, y1 = math.floor(limites[0] / 20) * 20, math.ceil(limites[3] / 20) * 20
    w, h = math.ceil((limites[2] - x0) / 20), math.ceil((y1 - limites[1]) / 20)
    if w <= 0 or h <= 0 or w * h > max_pixeles:
        raise ValueError("Recuadro fuera del límite de RAM para evaluar SCL a 20 m.")
    tr = from_origin(x0, y1, 20, 20)
    g = proyectar(shape(corte["geometry"]), 4326, crs)
    dentro = geometry_mask([mapping(g)], out_shape=(h, w), transform=tr, invert=True)
    n = int(dentro.sum())
    if not n:
        raise ValueError("El recuadro no contiene centros de píxel en la rejilla de análisis.")
    validos, claros = np.zeros((h, w), bool), np.zeros((h, w), bool)
    geometrias = []
    for item in items:
        geometrias.append(proyectar(shape(item["geometry"]), 4326, AREA))
        # Las fuentes Sentinel son públicas; no firmar con las claves del bucket de destino.
        with rasterio.Env(aws_unsigned=True, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=4,
                          GDAL_HTTP_RETRY_DELAY=2, GDAL_CACHEMAX=64 * 1024 * 1024,
                          VSI_CACHE=False):
            with rasterio.open(ruta_lectura_sentinel(item["assets"][ASSETS["SCL"]]["href"])) as src:
                if src.count != 1 or not src.crs:
                    raise ValueError("SCL sin CRS o con cantidad de bandas inesperada.")
                with WarpedVRT(src, crs=crs, transform=tr, width=w, height=h,
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    a = vrt.read(1)
                    mask = vrt.read_masks(1) > 0
                if not np.isin(a[mask], np.arange(12)).all():
                    raise ValueError("SCL contiene categorías inesperadas.")
                valido = dentro & mask & (a != 0) & (a != 1)
                validos |= valido
                claros |= valido & np.isin(a, clases)
    gm = proyectar(shape(corte["geometry"]), 4326, AREA)
    geom_frac = min(1., gm.intersection(unary_union(geometrias)).area / gm.area)
    return {"version_calidad": VERSION_CALIDAD, "resolucion_analisis_m": 20,
            "pixeles_recuadro": n, "pixeles_validos": int(validos.sum()),
            "pixeles_claros": int(claros.sum()), "fraccion_valida": float(validos.sum() / n),
            "fraccion_clara": float(claros.sum() / n), "fraccion_cobertura_geometrica": geom_frac,
            "numero_teselas": len(items)}


def escoger_candidato(candidatos, inicio, final, p):
    elegibles = [c for c in candidatos if c.get("estado") == "evaluado" and
                 c["metricas"]["fraccion_valida"] + 1e-12 >= p["fraccion_valida_minima"] and
                 c["metricas"]["fraccion_clara"] + 1e-12 >= p["fraccion_clara_minima"] and
                 c["metricas"]["fraccion_cobertura_geometrica"] + 1e-12 >= p["fraccion_valida_minima"]]
    if not elegibles:
        return None
    centro = dt.datetime.combine(inicio, dt.time(), tzinfo=dt.timezone.utc) + (final - inicio) / 2
    return min(elegibles, key=lambda c: (-c["metricas"]["fraccion_clara"],
        -c["metricas"]["fraccion_valida"], abs((dt.datetime.fromisoformat(c["fecha_utc"].replace("Z", "+00:00")) - centro).total_seconds()),
        c["adquisicion"]))


def recuperar_plan_compartido(cfg, s3):
    """Recupera el plan anual exacto, sin evaluar SCL ni volver a consultar STAC."""
    n, carpeta = cfg["sentinel_s3"], cfg["_trabajo"]
    bucket, prefix = validar_destino(n)
    selection = leer_json(carpeta / "seleccion_s3.json")
    validar_seleccion(selection)
    territorio = cfg["territorio"]
    if (sorted(territorio["cvegeo_seleccionados"]) != selection["cvegeo"] or
            float(territorio.get("fraccion_minima_aoi", .85)) != selection["fraccion_minima_aoi"] or
            (sorted(territorio["cortes_preseleccionados"]) if territorio.get("cortes_preseleccionados") is not None else None)
            != selection.get("cortes_preseleccionados")):
        raise ValueError("Cambió la selección o el filtro; ejecuta preparar antes de recuperar el plan.")
    esperado = firma(consulta_catalogo(cfg, selection))[:20]
    _, plan_id = ids_calidad(cfg, esperado)
    remote = leer_json_s3(s3, bucket, f"{prefix}/planes/{plan_id}/plan_descarga_s3.json")
    if remote is None:
        return None
    print(f"Recuperando catálogos del plan compartido {plan_id} desde S3...", flush=True)
    if (remote.get("plan_id") != plan_id or remote.get("version") != VERSION or
            remote.get("seleccion", {}).get("firma") != selection["firma"] or
            remote.get("seleccion_temporal") != parametros_calidad(cfg) or
            not remote.get("listo_para_transferir") or not remote.get("catalogo_completo")):
        raise ValueError("El plan compartido no coincide con el año, la malla o la calidad. No se mezclará.")
    validar_seleccion(remote["seleccion"])
    consulta = {**consulta_catalogo(cfg, selection), "seleccion_temporal": parametros_calidad(cfg)}
    if remote.get("consulta") != consulta:
        raise ValueError("La consulta del plan remoto no corresponde a la configuración actual.")
    documentos = []
    for parte in remote["partes"]:
        rel = Path(parte["archivo"])
        if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
            raise ValueError("Ruta no válida dentro del plan remoto.")
        doc = leer_json_s3(s3, bucket, f"{prefix}/planes/{plan_id}/{rel.as_posix()}")
        if doc is None:
            print("El plan remoto todavía se está publicando. Espera y repite esta sección.", flush=True)
            return None
        # El productor usa el mismo formato compacto: la huella debe coincidir byte por byte.
        reconstruido = json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if hashlib.sha256(reconstruido).hexdigest() != parte["sha256"]:
            raise ValueError("Un catálogo del plan compartido no coincide con su SHA256.")
        documentos.append((rel, doc))
    inicial, final = [dt.date.fromisoformat(n[k]) for k in ("fecha_inicial", "fecha_final")]
    intervalos = [[a.isoformat(), b.isoformat()] for a,b in meses(inicial, final)]
    if [doc.get("consulta", {}).get("intervalo") for _,doc in documentos] != intervalos:
        raise ValueError("El plan remoto no contiene todos los meses del periodo anual solicitado.")
    if sum(len(doc.get("casos", [])) for _,doc in documentos) != remote.get("conjuntos_periodo_recuadro"):
        raise ValueError("El número de conjuntos del plan compartido no coincide con sus catálogos.")
    for rel, doc in documentos:
        guardar_json(carpeta / rel, doc, compacto=True)
    informe = [x for _, d in documentos for x in d.get("cobertura_temporal", [])]
    import pandas as pd
    pd.DataFrame(informe).to_csv(carpeta / "cobertura_temporal.csv", index=False, encoding="utf-8-sig")
    guardar_json(carpeta / "informe_seleccion_temporal.json", remote["resumen_temporal"])
    guardar_json(carpeta / "plan_descarga_s3.json", remote)
    print(f"Plan anual compartido recuperado: {plan_id}; {len(documentos)} meses. No se releyeron imágenes SCL.", flush=True)
    return remote


def publicar_plan_compartido(cfg, plan, s3):
    """Las partes se publican primero; el índice anual completo se publica al final."""
    bucket, prefix = validar_destino(cfg["sentinel_s3"])
    root = f"{prefix}/planes/{plan['plan_id']}"
    for parte in plan["partes"]:
        archivo = cfg["_trabajo"] / parte["archivo"]
        if sha256(archivo) != parte["sha256"]:
            raise ValueError("Un catálogo cambió; no se publicará un plan incompatible.")
        guardar_s3(s3, bucket, f"{root}/{parte['archivo']}", leer_json(archivo))
    guardar_s3(s3, bucket, f"{root}/plan_descarga_s3.json", plan)


def planificar(cfg, *, s3=None, solo_recuperar=False, usar_plan_compartido=True):
    p = parametros_calidad(cfg)
    if p["modo"] == "todas":
        if solo_recuperar:
            raise ValueError("solo_recuperar requiere el modo mejor_por_bloque.")
        return planificar_todas(cfg)
    nube, carpeta = cfg["sentinel_s3"], cfg["_trabajo"]
    bucket, prefijo = validar_destino(nube) if p["usar_cache_s3"] else (nube.get("bucket"), nube.get("prefijo", "aguacate/sentinel2").strip("/"))
    if p["usar_cache_s3"]:
        s3 = s3 or cliente_s3(nube)
        region = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
        region = "eu-west-1" if region == "EU" else region
        if region != nube["region"]:
            raise ValueError(f"El bucket está en {region}; corrige la región antes de planificar.")
        if usar_plan_compartido or solo_recuperar:
            recuperado = recuperar_plan_compartido(cfg, s3)
            if recuperado is not None:
                return recuperado
    else:
        s3 = None
    if solo_recuperar:
        raise ValueError("Todavía no hay un plan anual compartido para esta configuración. "
                         "Pide al primer equipo que termine la sección 6 y después repite esta sección. "
                         "Activa usar_cache_s3 en ambos equipos.")
    raw = planificar_todas(cfg, s3=s3)
    calidad_id, plan_id = ids_calidad(cfg, raw["plan_id"])
    cortes = {c["corte"]: c for c in raw["seleccion"]["recuadros"]}
    destino = carpeta / "seleccionados" / plan_id
    destino.mkdir(parents=True, exist_ok=True)
    partes, informe, errores = [], [], []
    cantidad, fragmentos, estimacion = 0, 0, 0.
    for parte in raw["partes"]:
        doc = leer_json(carpeta / parte["archivo"])
        ini, fin = [dt.date.fromisoformat(x) for x in doc["consulta"]["intervalo"]]
        metricas_path = carpeta / "calidad" / calidad_id / f"{ini:%Y-%m}.json"
        cache_key = f"{prefijo}/cache/calidad/{calidad_id}/{ini:%Y-%m}.json"
        cache = leer_json(metricas_path) if metricas_path.exists() else None
        if cache is None and s3 is not None:
            cache = leer_json_s3(s3, bucket, cache_key)
        if cache is None:
            cache = {"calidad_id": calidad_id, "evaluaciones": {}}
        if cache.get("calidad_id") != calidad_id:
            raise ValueError("La caché de calidad no corresponde a la selección actual.")
        grupos = defaultdict(list)
        for t in doc["tareas"]:
            item = doc["items"][t["item"]]
            grupos[(t["corte"], clave_adquisicion(item))].append(t["item"])
        evaluados = defaultdict(list)
        cambios = 0
        try:
            for (corte, adquisicion), indices in sorted(grupos.items()):
                indices = sorted(set(indices), key=lambda i: doc["items"][i]["id"])
                items = [doc["items"][i] for i in indices]
                huella = firma({"version": VERSION_CALIDAD, "corte": cortes[corte], "adquisicion": adquisicion,
                               "scl": [it["assets"][ASSETS["SCL"]] for it in items],
                               "geometrias": [it["geometry"] for it in items]})
                entrada = cache["evaluaciones"].get(huella)
                if entrada is None or entrada.get("estado") == "error":
                    entrada = {"corte": corte, "adquisicion": adquisicion,
                               "fecha_utc": min(fecha_escena(it) for it in items).isoformat(), "items": indices}
                    try:
                        entrada.update(estado="evaluado", metricas=medir_calidad(items, cortes[corte],
                            clases=nube.get("clases_scl_claras", [4, 5, 6]),
                            max_pixeles=int(nube.get("max_pixeles_por_banda", 16_000_000))))
                    except Exception as e:
                        entrada.update(estado="error", error=f"{type(e).__name__}: {e}"[:700])
                    cache["evaluaciones"][huella] = entrada
                    guardar_json(metricas_path, cache, compacto=True)
                    cambios += 1
                    if s3 is not None and cambios % 10 == 0:
                        guardar_s3(s3, bucket, cache_key, cache)
                if entrada["items"] != indices:
                    raise ValueError("El orden de escenas del catálogo cambió; no se reutiliza una evaluación incompatible.")
                evaluados[corte].append(entrada)
                if entrada["estado"] == "error":
                    errores.append({"corte": corte, "adquisicion": adquisicion, "error": entrada["error"]})
                print(f"SCL {ini:%Y-%m} · recuadro {corte} · {entrada['fecha_utc'][:10]} · "
                      f"{entrada['estado']}", flush=True)
        finally:
            guardar_json(metricas_path, cache, compacto=True)
            if s3 is not None:
                guardar_s3(s3, bucket, cache_key, cache)
        casos, tareas = [], []
        for corte in sorted(cortes):
            for numero, a, b in bloques_mes(ini, fin):
                candidatos = [c for c in evaluados[corte] if a <= dt.datetime.fromisoformat(c["fecha_utc"]).date() <= b]
                pendientes = any(c["estado"] == "error" for c in candidatos)
                mejor = None if pendientes else escoger_candidato(candidatos, a, b, p)
                estado = ("error_evaluacion" if pendientes else "seleccionado" if mejor else
                          "calidad_insuficiente" if candidatos else "sin_adquisiciones")
                fila = {"corte": corte, "anio": a.year, "mes": a.month,
                        "bloque_id": f"{a:%Y-%m}-B{numero}", "bloque": numero,
                        "fecha_inicial": a.isoformat(), "fecha_final": b.isoformat(), "estado": estado,
                        "candidatos": len(candidatos), "fecha_elegida": mejor["fecha_utc"] if mejor else None,
                        "fraccion_clara": mejor["metricas"]["fraccion_clara"] if mejor else None,
                        "fraccion_valida": mejor["metricas"]["fraccion_valida"] if mejor else None}
                informe.append(fila)
                if mejor:
                    caso = {**fila, "fecha_utc": mejor["fecha_utc"],
                            "intervalo": [a.isoformat(), b.isoformat()], "adquisicion": mejor["adquisicion"],
                            "items": mejor["items"], "metricas": mejor["metricas"]}
                    casos.append(caso)
                    tareas.extend({"corte": corte, "item": i, "bloque_id": fila["bloque_id"]} for i in mejor["items"])
                    area = proyectar(shape(cortes[corte]["geometry"]), 4326, AREA).area
                    estimacion += len(mejor["items"]) * area * (4 * 2 / 100 + 6 * 2 / 400 + 1 / 400)
        archivo = destino / f"{ini:%Y-%m}.json"
        contenido = {"consulta": doc["consulta"], "items": doc["items"], "tareas": tareas,
                     "casos": casos, "cobertura_temporal": [x for x in informe if x["fecha_inicial"][:7] == f"{ini:%Y-%m}"]}
        guardar_json(archivo, contenido, compacto=True)
        partes.append({"archivo": str(archivo.relative_to(carpeta)), "sha256": sha256(archivo),
                       "escenas": len({t["item"] for t in tareas}), "casos": len(tareas), "conjuntos": len(casos)})
        cantidad += len(casos)
        fragmentos += len(tareas)
        print(f"{ini:%Y-%m}: {len(casos)}/{len(cortes) * len(list(bloques_mes(ini, fin)))} conjuntos seleccionados.", flush=True)
    import pandas as pd
    pd.DataFrame(informe).to_csv(carpeta / "cobertura_temporal.csv", index=False, encoding="utf-8-sig")
    resumen = {"periodos_recuadro": len(informe), "seleccionados": cantidad,
               "faltantes": len(informe) - cantidad, "errores_evaluacion": errores,
               "criterio": p, "version_calidad": VERSION_CALIDAD,
               "nota": "SCL se usa para calidad; no se añade como canal del modelo. Se mantienen fechas reales."}
    guardar_json(carpeta / "informe_seleccion_temporal.json", resumen)
    plan = {**raw, "plan_id": plan_id, "catalogo_candidatos_id": raw["plan_id"], "partes": partes,
            "consulta": {**raw["consulta"], "seleccion_temporal": p}, "seleccion_temporal": p,
            "listo_para_transferir": not errores, "conjuntos_periodo_recuadro": cantidad,
            "casos_escena_recuadro": fragmentos, "archivos_banda_previstos": fragmentos * len(ASSETS),
            "gib_nativos_sin_comprimir_estimados": round(estimacion / 2**30, 2), "resumen_temporal": resumen,
            "nota_tamanio": "Estimación de fragmentos nativos seleccionados sin comprimir. Las teselas de una misma toma "
                           "se guardan independientes para conservar píxeles originales; el manifiesto las agrupa."}
    guardar_json(carpeta / "plan_descarga_s3.json", plan)
    if s3 is not None:
        guardar_s3(s3, bucket, f"{prefijo}/planes/{plan_id}/informe_seleccion_temporal.json", resumen)
        guardar_s3(s3, bucket, f"{prefijo}/planes/{plan_id}/cobertura_temporal.json", informe)
    if errores:
        raise ValueError(f"Quedan {len(errores)} evaluaciones SCL con error. Repite planificar; "
                         "se reintentan solo las pendientes. No se transferirán bandas con un plan incompleto.")
    if s3 is not None:
        publicar_plan_compartido(cfg, plan, s3)
    print(f"Selección lista: {cantidad} conjuntos reales; {len(informe) - cantidad} faltantes registrados. "
          f"{fragmentos * len(ASSETS)} COG nativos previstos.")
    return plan


def validar_destino(nube):
    bucket = nube["bucket"]
    if (not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or "CAMBIAR" in bucket or
            ".." in bucket or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", bucket)):
        raise ValueError("Escribe el nombre real de tu bucket en sentinel_s3.bucket.")
    prefijo = nube["prefijo"].strip("/")
    if not prefijo or any(x in (".", "..") for x in prefijo.split("/")):
        raise ValueError("El prefijo S3 no es válido.")
    return bucket, prefijo


def cliente_s3(nube):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name=nube["region"], config=Config(
        retries={"max_attempts": 6, "mode": "standard"}, connect_timeout=20, read_timeout=120))


def cabecera(s3, bucket, clave):
    from botocore.exceptions import ClientError
    try:
        return s3.head_object(Bucket=bucket, Key=clave)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise  # Un 403 no se interpreta como archivo ausente.


def ventana_nativa(src, corte):
    if not src.crs or src.count != 1 or src.transform.b or src.transform.d:
        raise ValueError("Se esperaba un COG monobanda georreferenciado sin rotación.")
    bbox = transform_bounds(corte["crs_original"], src.crs, *corte["bbox_original"], densify_pts=41)
    w = from_bounds(*bbox, transform=src.transform)
    c0, r0 = max(0, math.floor(w.col_off + 1e-8)), max(0, math.floor(w.row_off + 1e-8))
    c1 = min(src.width, math.ceil(w.col_off + w.width - 1e-8))
    r1 = min(src.height, math.ceil(w.row_off + w.height - 1e-8))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("La escena y el recuadro no comparten píxeles. Revisa geometría y CRS.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def huella_banda(item, corte, banda):
    return firma({"version": VERSION, "collection": item["collection"], "id": item["id"],
                  "corte": corte, "banda": banda, "asset": item["assets"][ASSETS[banda]]})


def construir_tiff(item, corte, banda, *, max_pixeles=16_000_000, clases_claras=(4, 5, 6)):
    """Devuelve MemoryFile cerrado por el llamador. Conserva DN, máscara y rejilla nativa."""
    asset = item["assets"][ASSETS[banda]]
    rb = asset.get("raster:bands", [{}])[0]
    memoria = None
    try:
        # Solo HTTP Range de los bloques necesarios. Caché GDAL acotada a RAM.
        # También hay escenas públicas JP2 con href s3://; sus lecturas deben ser anónimas.
        with rasterio.Env(aws_unsigned=True, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=4,
                          GDAL_HTTP_RETRY_DELAY=2, GDAL_CACHEMAX=64 * 1024 * 1024,
                          GDAL_TIFF_INTERNAL_MASK=True, VSI_CACHE=False,
                          CPL_VSIL_CURL_CACHE_SIZE=16 * 1024 * 1024):
            with rasterio.open(ruta_lectura_sentinel(asset["href"])) as src:
                w = ventana_nativa(src, corte)
                if int(w.width * w.height) > max_pixeles:
                    raise ValueError("Recuadro demasiado grande para el límite de RAM por banda. "
                                     "Divídelo conservando resolución; no se reducirá automáticamente.")
                resolucion = RESOLUCIONES[banda]
                if not (math.isclose(abs(src.res[0]), resolucion, abs_tol=0.01) and
                        math.isclose(abs(src.res[1]), resolucion, abs_tol=0.01)):
                    raise ValueError(f"{banda} no está a su resolución nativa esperada ({resolucion} m).")
                a = src.read(1, window=w)  # Sin out_shape, WarpedVRT, normalización ni remuestreo.
                valido = src.read_masks(1, window=w) > 0
                transform = src.window_transform(w)
                escala = float(rb.get("scale", 1))
                offset = float(rb.get("offset", 0))
                if banda != "SCL" and not all(k in rb for k in ("scale", "offset")):
                    raise ValueError("La banda no tiene escala/offset explícitos.")
                if not np.isfinite([escala, offset]).all():
                    raise ValueError("Escala/offset no finitos.")
                if banda == "SCL" and not np.isin(a[valido], np.arange(12)).all():
                    raise ValueError("SCL contiene clases inesperadas.")
                # Una máscara interna conserva validez sin sobrescribir los DN originales.
                perfil = {"driver": "GTiff", "height": a.shape[0], "width": a.shape[1],
                          "count": 1, "dtype": a.dtype, "crs": src.crs, "transform": transform,
                          "nodata": src.nodata, "tiled": True, "blockxsize": 256, "blockysize": 256,
                          "compress": "DEFLATE", "predictor": 2 if a.dtype.kind in "ui" else 3,
                          "BIGTIFF": "IF_SAFER"}
                memoria = MemoryFile()
                with memoria.open(**perfil) as dst:
                    dst.write(a, 1)
                    dst.write_mask(valido.astype("uint8") * 255)
                    dst.scales, dst.offsets = (escala,), (offset,)
                    dst.set_band_description(1, banda)
                    dst.update_tags(BANDA=banda, ITEM_ID=item["id"], COLECCION=item["collection"],
                                    FECHA_UTC=item["properties"]["datetime"],
                                    CORTE_ORIGINAL=str(corte["corte"]), VERSION_INGESTA=VERSION,
                                    RADIOMETRIA="DN original; reflectancia = DN * scale + offset; sin clipping",
                                    SOURCE_FINGERPRINT=huella_banda(item, corte, banda))
                info = {"banda": banda, "resolucion_m": resolucion, "dtype": str(a.dtype),
                        "shape": list(a.shape), "crs": src.crs.to_string(),
                        "transform": list(transform)[:6], "scale": escala, "offset": offset,
                        "nodata": src.nodata, "pixeles_validos": int(valido.sum()),
                        "fraccion_valida_rectangulo": float(valido.mean()), "asset_origen": asset}
                if banda == "SCL":
                    rec = proyectar(shape(corte["geometry"]), 4326, src.crs)
                    dentro = geometry_mask([mapping(rec)], out_shape=a.shape, transform=transform, invert=True)
                    denominador = int(dentro.sum())
                    info["fraccion_clara_recuadro"] = float((dentro & valido & np.isin(a, clases_claras)).sum() / denominador) if denominador else None
                    info["conteo_scl_valido"] = np.bincount(a[valido].astype(int), minlength=12).tolist()
        # Reordenar como COG en otra MemoryFile; tampoco crea un TIFF en disco.
        cog = MemoryFile()
        try:
            copiar_raster(memoria.name, cog.name, driver="COG", COMPRESS="DEFLATE",
                          BLOCKSIZE=256, OVERVIEWS="NONE", BIGTIFF="IF_SAFER")
        except BaseException:
            cog.close()
            raise
        memoria.close()
        memoria = cog
        memoria.seek(0)
        h = hashlib.sha256()
        for bloque in iter(lambda: memoria.read(1024 * 1024), b""):
            h.update(bloque)
        info.update(bytes=len(memoria), sha256=h.hexdigest(), fingerprint=huella_banda(item, corte, banda))
        memoria.seek(0)
        return memoria, info
    except BaseException:
        if memoria is not None:
            memoria.close()
        raise


def segmento(valor):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(valor)):
        raise ValueError("Identificador STAC no válido para la ruta S3.")
    return str(valor)


def base_caso(prefijo, item, corte):
    fecha = fecha_escena(item).date()
    return (f"{prefijo}/nativo/{VERSION}/recuadro_{corte['corte']:03d}/{fecha:%Y/%m/%d}/"
            f"{segmento(item['collection'])}/{segmento(item['id'])}")


def guardar_s3(s3, bucket, clave, objeto):
    s3.put_object(Bucket=bucket, Key=clave,
                  Body=json.dumps(objeto, ensure_ascii=False, allow_nan=False).encode(),
                  ContentType="application/json", ServerSideEncryption="AES256")


def inventario_s3(s3, bucket, prefijo):
    """Una consulta por 1000 objetos evita un HEAD por banda al reanudar el histórico."""
    objetos, token, vistos = {}, None, set()
    while True:
        parametros = {"Bucket": bucket, "Prefix": prefijo + "/", "MaxKeys": 1000}
        if token:
            parametros["ContinuationToken"] = token
        pagina = s3.list_objects_v2(**parametros)
        objetos.update((x["Key"], x["Size"]) for x in pagina.get("Contents", []))
        if not pagina.get("IsTruncated"):
            return objetos
        token = pagina.get("NextContinuationToken")
        if not token or token in vistos:
            raise ValueError("S3 repitió una página de inventario; se detiene la reanudación.")
        vistos.add(token)


def firma_caso(item, corte):
    return firma({b: huella_banda(item, corte, b) for b in ASSETS})


def grupos_plan(cfg, plan, por_bloque):
    for parte in plan["partes"]:
        doc = leer_json(cfg["_trabajo"] / parte["archivo"])
        for grupo in (doc["casos"] if por_bloque else
                      [{"corte": t["corte"], "items": [t["item"]]} for t in doc["tareas"]]):
            items = [doc["items"][i] for i in grupo["items"]]
            yield grupo, items


def grupo_asignado(cfg, items):
    a, b = rango_meses(cfg)
    meses_grupo = {fecha_escena(it).month for it in items}
    if len(meses_grupo) != 1:
        raise ValueError("Un conjunto contiene fechas de meses distintos; no se repartirá.")
    return a <= next(iter(meses_grupo)) <= b


def inventario_nativo_plan(cfg, plan, s3):
    bucket, prefix = validar_destino(cfg["sentinel_s3"])
    consulta = plan["consulta"]
    years = range(int(consulta["fecha_inicial"][:4]), int(consulta["fecha_final"][:4]) + 1)
    out = {}
    for corte in plan["seleccion"]["recuadros"]:
        for year in years:
            out.update(inventario_s3(s3, bucket,
                f"{prefix}/nativo/{VERSION}/recuadro_{corte['corte']:03d}/{year:04d}"))
    return out


def grupo_completo(prefijo, plan_id, corte, grupo, items, existentes, conjuntos, por_bloque):
    completo = all(existentes.get(
        f"{base_caso(prefijo, it, corte)}/_completo_{firma_caso(it, corte)}.json", 0) > 0 and
        all(existentes.get(f"{base_caso(prefijo, it, corte)}/{b}.tif", 0) > 0 for b in ASSETS)
        for it in items)
    key = (f"{prefijo}/selecciones/{plan_id}/recuadro_{corte['corte']:03d}/"
           f"{grupo.get('bloque_id', items[0]['id'])}/_completo.json")
    return completo and (not por_bloque or conjuntos.get(key, 0) > 0)


def resumen_estado_s3(cfg, plan, por_bloque, s3):
    bucket, prefix = validar_destino(cfg["sentinel_s3"])
    nativos = inventario_nativo_plan(cfg, plan, s3)
    conjuntos = inventario_s3(s3, bucket, f"{prefix}/selecciones/{plan['plan_id']}") if por_bloque else {}
    cortes = {c["corte"]: c for c in plan["seleccion"]["recuadros"]}
    total = completos = asignados = completos_asignados = 0
    faltantes = []
    for grupo, items in grupos_plan(cfg, plan, por_bloque):
        total += 1
        asignado = grupo_asignado(cfg, items)
        completo = grupo_completo(prefix, plan["plan_id"], cortes[grupo["corte"]], grupo,
                                   items, nativos, conjuntos, por_bloque)
        completos += int(completo)
        asignados += int(asignado)
        completos_asignados += int(asignado and completo)
        if not completo and len(faltantes) < 20:
            faltantes.append({"corte": grupo["corte"], "periodo": grupo.get("bloque_id", items[0]["id"])})
    a, b = rango_meses(cfg)
    return {"plan_id": plan["plan_id"], "mes_inicial": a, "mes_final": b,
            "conjuntos_asignados": asignados, "conjuntos_asignados_completos_en_s3": completos_asignados,
            "conjuntos_asignados_pendientes": asignados - completos_asignados,
            "rango_completo_en_s3": asignados == completos_asignados,
            "conjuntos_plan_anual": total, "conjuntos_anio_completos_en_s3": completos,
            "conjuntos_anio_pendientes": total - completos,
            "anio_completo_en_s3": total > 0 and total == completos,
            "ejemplos_pendientes": faltantes,
            "nota": "Comprueba presencia y tamaño positivo de recibos, TIFF y conjuntos del plan seleccionado; "
                    "no vuelve a descargar los píxeles. Los bloques descartados por calidad "
                    "no son descargas pendientes. El estado refleja lo observado durante este inventario."}


def verificar(cfg, *, s3=None):
    """Solo lectura: comprueba la asignación actual y todo el plan anual en S3."""
    plan, por_bloque = validar_plan_transferencia(cfg)
    s3 = s3 or cliente_s3(cfg["sentinel_s3"])
    print("Comprobando los meses asignados y el plan anual en S3; solo se listan objetos.", flush=True)
    report = resumen_estado_s3(cfg, plan, por_bloque, s3)
    guardar_json(cfg["_trabajo"] / "verificacion_s3.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class LimiteTransferencia(Exception):
    pass


def transferir_banda(s3, bucket, clave, item, corte, banda, *, limite_bytes=None,
                     bytes_usados=0, max_pixeles=16_000_000, clases_claras=(4, 5, 6)):
    huella = huella_banda(item, corte, banda)
    existente = cabecera(s3, bucket, clave)
    if existente:
        meta = existente.get("Metadata", {})
        if meta.get("source-fingerprint") != huella or not meta.get("sha256") or existente["ContentLength"] <= 0:
            raise ValueError(f"Objeto existente incompatible: {clave}. Se conserva sin sobrescribir.")
        return {"banda": banda, "key": clave, "estado": "reutilizado",
                "bytes": existente["ContentLength"], "sha256": meta["sha256"], "fingerprint": huella}
    memoria, info = construir_tiff(item, corte, banda, max_pixeles=max_pixeles, clases_claras=clases_claras)
    try:
        if limite_bytes is not None and bytes_usados + info["bytes"] > limite_bytes:
            raise LimiteTransferencia("Límite de GiB de esta ejecución alcanzado. Repite para continuar.")
        from boto3.s3.transfer import TransferConfig
        s3.upload_fileobj(memoria, bucket, clave, ExtraArgs={"ContentType": "image/tiff",
            "ServerSideEncryption": "AES256", "StorageClass": "STANDARD",
            "Metadata": {"source-fingerprint": huella, "sha256": info["sha256"],
                         "ingest-version": VERSION}},
            Config=TransferConfig(multipart_threshold=8 * 1024**2, multipart_chunksize=8 * 1024**2,
                                  max_concurrency=2, use_threads=True, preferred_transfer_client="classic"))
        recibido = cabecera(s3, bucket, clave)
        if not recibido or recibido["ContentLength"] != info["bytes"] or recibido.get("Metadata", {}).get("sha256") != info["sha256"]:
            raise IOError("S3 no confirmó el tamaño y SHA256 del TIFF transferido.")
        return {**info, "key": clave, "estado": "subido"}
    finally:
        memoria.close()


def validar_plan_transferencia(cfg):
    nube, carpeta = cfg["sentinel_s3"], cfg["_trabajo"]
    rango_meses(cfg)
    plan = leer_json(carpeta / "plan_descarga_s3.json")
    if not plan.get("catalogo_completo") or plan.get("version") != VERSION:
        raise ValueError("El plan no está completo. Ejecuta planificar.")
    temporal = parametros_calidad(cfg)
    por_bloque = temporal["modo"] == "mejor_por_bloque"
    if por_bloque and (plan.get("seleccion_temporal") != temporal or not plan.get("listo_para_transferir")):
        raise ValueError("El plan por bloques no está listo o cambió la calidad. Ejecuta planificar otra vez.")
    if por_bloque and not plan.get("conjuntos_periodo_recuadro"):
        raise ValueError("No hay conjuntos seleccionados para transferir. Revisa cobertura_temporal.csv y los criterios de calidad.")
    if not por_bloque and plan.get("seleccion_temporal", {}).get("modo", "todas") != "todas":
        raise ValueError("El plan temporal no corresponde al modo configurado.")
    validar_seleccion(plan["seleccion"])
    actual = leer_json(carpeta / "seleccion_s3.json")
    validar_seleccion(actual)
    if actual["firma"] != plan["seleccion"]["firma"]:
        raise ValueError("El filtro cambió; planifica otra vez antes de transferir.")
    if (sorted(cfg["territorio"]["cvegeo_seleccionados"]) != actual["cvegeo"] or
            float(cfg["territorio"].get("fraccion_minima_aoi", 0.85)) != actual["fraccion_minima_aoi"]):
        raise ValueError("Cambió el umbral o los municipios; ejecuta preparar y planificar.")
    previos = cfg["territorio"].get("cortes_preseleccionados")
    if (sorted(previos) if previos is not None else None) != actual.get("cortes_preseleccionados"):
        raise ValueError("Cambió la lista de recuadros de partida; ejecuta preparar y planificar.")
    if [nube["fecha_inicial"], nube["fecha_final"]] != [plan["consulta"]["fecha_inicial"], plan["consulta"]["fecha_final"]]:
        raise ValueError("Cambió el periodo; vuelve a planificar.")
    for parte in plan["partes"]:
        if sha256(carpeta / parte["archivo"]) != parte["sha256"]:
            raise ValueError("Un catálogo cambió después de planificar. No se transfiere una mezcla de versiones.")
    return plan, por_bloque


def transferir(cfg, *, max_casos=0, max_gib=0, s3=None):
    nube, carpeta = cfg["sentinel_s3"], cfg["_trabajo"]
    bucket, prefijo = validar_destino(nube)
    if max_casos < 0 or max_gib < 0 or not math.isfinite(max_gib):
        raise ValueError("Los límites no pueden ser negativos ni no finitos; 0 significa sin límite.")
    plan, por_bloque = validar_plan_transferencia(cfg)
    s3 = s3 or cliente_s3(nube)
    region = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
    region = "eu-west-1" if region == "EU" else region
    if region != nube["region"]:
        raise ValueError(f"El bucket está en {region}; corrige sentinel_s3.region.")
    indice = {c["corte"]: c for c in plan["seleccion"]["recuadros"]}
    print("Consultando los archivos ya guardados en S3 para reanudar...", flush=True)
    existentes = inventario_nativo_plan(cfg, plan, s3)
    pref_conjuntos = f"{prefijo}/selecciones/{plan['plan_id']}"
    conjuntos_existentes = inventario_s3(s3, bucket, pref_conjuntos) if por_bloque else {}
    pref_plan = f"{prefijo}/planes/{plan['plan_id']}"
    publicar_plan_compartido(cfg, plan, s3)
    mes_inicial, mes_final = rango_meses(cfg)
    asignados = sum(grupo_asignado(cfg, items) for _, items in grupos_plan(cfg, plan, por_bloque))
    limite = int(max_gib * 2**30) if max_gib else None
    estado = {"plan_id": plan["plan_id"], "bucket": bucket, "inicio_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "bytes_tiff_nuevos": 0, "casos_completados_esta_ejecucion": 0, "casos_reutilizados": 0,
              "bandas_nuevas": 0, "bandas_reutilizadas": 0, "errores": [], "terminado": False,
              "mes_inicial": mes_inicial, "mes_final": mes_final, "conjuntos_asignados": asignados,
              "alcance_terminado": "solo los meses asignados a este equipo",
              "anio_completo_en_s3": None,
              "nota_limite": "max_gib limita TIFF nuevos de esta ejecución, no el costo mensual ni los JSON."}
    print(f"Trabajo de este equipo: meses {mes_inicial:02d} a {mes_final:02d}; "
          f"{asignados} conjuntos asignados del plan anual {plan['plan_id']}.", flush=True)
    detener = False
    try:
        for parte in plan["partes"]:
            doc = leer_json(carpeta / parte["archivo"])
            grupos = (doc["casos"] if por_bloque else
                      [{"corte": t["corte"], "items": [t["item"]]} for t in doc["tareas"]])
            for grupo in grupos:
                corte = indice[grupo["corte"]]
                items = [doc["items"][i] for i in grupo["items"]]
                if not grupo_asignado(cfg, items):
                    continue
                nombre_grupo = grupo.get("bloque_id", items[0]["id"])
                clave_conjunto = f"{pref_conjuntos}/recuadro_{corte['corte']:03d}/{nombre_grupo}/_completo.json"
                if grupo_completo(prefijo, plan["plan_id"], corte, grupo, items,
                                  existentes, conjuntos_existentes, por_bloque):
                    estado["casos_reutilizados"] += 1
                    estado["bandas_reutilizadas"] += len(ASSETS) * len(items)
                    if estado["casos_reutilizados"] % 100 == 0:
                        print(f"{estado['casos_reutilizados']} conjuntos completos reutilizados.", flush=True)
                    continue
                nuevo_grupo = False
                fragmentos = []
                try:
                    for item in items:
                        base = base_caso(prefijo, item, corte)
                        recibo = f"{base}/_completo_{firma_caso(item, corte)}.json"
                        presente = existentes.get(recibo, 0) > 0 and all(
                            existentes.get(f"{base}/{b}.tif", 0) > 0 for b in ASSETS)
                        resultados = []
                        if presente:
                            estado["bandas_reutilizadas"] += len(ASSETS)
                        else:
                            for banda in ASSETS:
                                resultado = transferir_banda(s3, bucket, f"{base}/{banda}.tif", item, corte, banda,
                                    limite_bytes=limite, bytes_usados=estado["bytes_tiff_nuevos"],
                                    max_pixeles=int(nube.get("max_pixeles_por_banda", 16_000_000)),
                                    clases_claras=nube.get("clases_scl_claras", [4, 5, 6]))
                                resultados.append(resultado)
                                nuevo = resultado["estado"] == "subido"
                                estado["bandas_nuevas" if nuevo else "bandas_reutilizadas"] += 1
                                if nuevo:
                                    estado["bytes_tiff_nuevos"] += resultado["bytes"]
                                    nuevo_grupo = True
                                existentes[resultado["key"]] = resultado["bytes"]
                            documento = {"version": VERSION, "corte": corte, "item": item,
                                "plan_id": plan["plan_id"], "bandas": resultados,
                                "radiometria": "DN originales y escala/offset por banda; sin normalizar ni remuestrear",
                                "advertencia": "Cada fragmento conserva la rejilla de su fuente y la fecha real."}
                            guardar_s3(s3, bucket, f"{base}/_completo.json", documento)
                            guardar_s3(s3, bucket, recibo, documento)
                            existentes[recibo] = 1
                        fragmentos.append({"item": item["id"], "fecha_real_utc": item["properties"]["datetime"],
                                           "manifiesto": f"{base}/_completo.json", "recibo": recibo,
                                           "bandas": {b: f"{base}/{b}.tif" for b in ASSETS}})
                    if por_bloque:
                        nuevo_grupo |= conjuntos_existentes.get(clave_conjunto, 0) <= 0
                        guardar_s3(s3, bucket, clave_conjunto, {"version": VERSION, "plan_id": plan["plan_id"],
                            "seleccion": grupo, "corte": corte, "fragmentos": fragmentos,
                            "nota": "Una adquisición por periodo y recuadro. Puede contener varias teselas "
                                    "de la misma toma, conservadas a resolución nativa. No mezcla fechas."})
                        conjuntos_existentes[clave_conjunto] = 1
                    estado["casos_completados_esta_ejecucion" if nuevo_grupo else "casos_reutilizados"] += 1
                    print(f"Recuadro {corte['corte']} · {nombre_grupo} · "
                          f"{'completado' if nuevo_grupo else 'reutilizado'} · "
                          f"{estado['bytes_tiff_nuevos'] / 2**30:.3f} GiB nuevos", flush=True)
                    if max_casos and estado["casos_completados_esta_ejecucion"] >= max_casos:
                        estado["motivo_parada"] = "max_casos"
                        detener = True
                        break
                except LimiteTransferencia as e:
                    estado["motivo_parada"] = str(e)
                    detener = True
                    break
                except Exception as e:
                    from botocore.exceptions import ClientError
                    codigo = e.response.get("Error", {}).get("Code", "") if isinstance(e, ClientError) else ""
                    estado["errores"].append({"corte": corte["corte"], "periodo": nombre_grupo,
                                             "tipo": type(e).__name__, "mensaje": str(e)[:700]})
                    print(f"ERROR recuadro {corte['corte']} / {nombre_grupo}: {e}", file=sys.stderr, flush=True)
                    if codigo in ("AccessDenied", "403", "ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                        detener = True
                        break
            if detener:
                break
        estado["terminado"] = not detener and not estado["errores"]
    finally:
        # Un equipo puede terminar su semestre mientras el otro todavía descarga el suyo.
        try:
            comprobacion = resumen_estado_s3(cfg, plan, por_bloque, s3)
            estado.update({k: v for k, v in comprobacion.items() if k not in ("nota", "ejemplos_pendientes")})
        except Exception as error:
            estado["error_verificacion_anual"] = f"{type(error).__name__}: {error}"[:700]
        estado["fin_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        guardar_json(carpeta / "ultima_transferencia.json", estado)
        guardar_json(carpeta / f"ultima_transferencia_meses_{mes_inicial:02d}_{mes_final:02d}.json", estado)
        clave_estado = (f"{pref_plan}/ejecuciones/meses_{mes_inicial:02d}_{mes_final:02d}/"
                        f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%f}.json")
        try:
            guardar_s3(s3, bucket, clave_estado, estado)
        except Exception:
            print("No se pudo guardar el informe de ejecución en S3; la copia está en ultima_transferencia.json.", file=sys.stderr)
    print(json.dumps(estado, ensure_ascii=False, indent=2))
    return estado


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("accion", choices=["preparar", "planificar", "transferir", "verificar"])
    p.add_argument("--config", type=Path, default=RAIZ / "configuracion.json")
    p.add_argument("--anio", type=int, choices=range(2020, 2026), help="Procesar un año completo de 2020 a 2025.")
    p.add_argument("--mes-inicial", type=int, choices=range(1, 13), default=None, help="Primer mes asignado a este equipo.")
    p.add_argument("--mes-final", type=int, choices=range(1, 13), default=None, help="Último mes asignado a este equipo.")
    p.add_argument("--solo-recuperar-plan", action="store_true", help="No evaluar SCL: esperar un plan anual compartido.")
    p.add_argument("--max-casos", type=int, default=0, help="Casos con datos nuevos por ejecución; 0 = todos.")
    p.add_argument("--max-gib", type=float, default=0, help="GiB de TIFF nuevos por ejecución; 0 = sin límite.")
    args = p.parse_args(argv)
    cfg = configuracion(args.config, anio=args.anio, mes_inicial=args.mes_inicial, mes_final=args.mes_final)
    if args.accion == "preparar":
        preparar(cfg)
    elif args.accion == "planificar":
        planificar(cfg, solo_recuperar=args.solo_recuperar_plan)
    elif args.accion == "verificar":
        verificar(cfg)
    else:
        estado = transferir(cfg, max_casos=args.max_casos, max_gib=args.max_gib)
        if estado["errores"]:
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
