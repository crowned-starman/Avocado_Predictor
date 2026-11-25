"""Sentinel público en Earth Search y lluvia CDS; caché por plan y fechas reales."""

import json
import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from datos_era5 import (
    DATASET_ERA5,
    abrir_netcdf,
    crear_cliente_cds,
    interpretar_precipitacion,
)
from datos_sentinel import (
    EARTH_SEARCH_COLECCIONES,
    PROCEDENCIA,
    RADIOMETRIA,
    buscar_escenas,
    leer_escenas,
    resolver_escenas_earth,
)
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union
from utilidades import BANDAS, guardar_json, leer_json, proyectar, sha256

from .comun_piloto import carpeta_caso, ubicaciones, validar_lluvia


def evaluar_tiff(ruta, lugar, p):
    from preparacion import mapa_bandas

    with rasterio.open(ruta) as src:
        idx = mapa_bandas(src)
        if src.shape != (64, 64) or str(src.crs) != str(
            rasterio.crs.CRS.from_user_input(lugar["crs"])
        ):
            raise ValueError("Muestra Sentinel con forma o CRS distintos al plan.")
        if not src.transform.almost_equals(Affine(*lugar["transform"])):
            raise ValueError("La muestra cambió de terreno entre fechas.")
        arr = src.read([idx[b] for b in BANDAS])
        scl = src.read(idx["SCL"])
        mask = src.read(idx["dataMask"])
        if (
            not np.isfinite(arr).all()
            or not np.isin(mask, [0, 1]).all()
            or not np.isin(scl, np.arange(12)).all()
        ):
            raise ValueError("Datos o etiquetas de calidad no válidos.")
        datos = mask > 0
        claro = datos & np.isin(scl, p["clases_scl_claras"])
        if not datos.all():
            return False, float(claro.mean())
        if np.median(np.abs(arr)) > 2 or np.max(np.abs(arr)) > 100:
            raise ValueError("Las bandas no parecen reflectancia.")
        return float(claro.mean()) >= p["fraccion_clara_minima"], float(claro.mean())


def descargar_sentinel(cfg, plan, caso, lugar):

    datos, _ = carpeta_caso(cfg, plan, caso)
    destino = datos / "sentinel2"
    destino.mkdir(parents=True, exist_ok=True)
    secuencia = destino / "secuencia.json"
    p = plan["configuracion"]
    if secuencia.exists():
        m = leer_json(secuencia)
        if (
            m["firma_plan"] == plan["firma_plan"]
            and len(m["archivos"]) == p["observaciones_sentinel_por_caso"]
        ):
            if all(
                (destino / n).is_file() and sha256(destino / n) == h
                for n, h in m["sha256"].items()
            ):
                logging.info("%s: secuencia Sentinel reutilizada.", caso["id"])
                return
        secuencia.unlink()
    bb_wgs84 = transform_bounds(
        lugar["crs"], "EPSG:4326", *lugar["bbox"], densify_pts=21
    )
    ruta_catalogo = destino / "catalogo.json"
    consulta = dict(
        version=2,
        bbox=list(bb_wgs84),
        intervalo=[caso["inicio_contexto"], caso["fin_contexto"]],
        colecciones=list(EARTH_SEARCH_COLECCIONES),
    )
    cache = leer_json(ruta_catalogo) if ruta_catalogo.exists() else None
    vigente = (
        isinstance(cache, dict)
        and cache.get("consulta") == consulta
        and isinstance(cache.get("items"), list)
    )
    items = cache.get("items", []) if vigente else []
    dias = {it["properties"]["datetime"][:10] for it in items}
    if not vigente or len(dias) < p["observaciones_sentinel_por_caso"]:
        if ruta_catalogo.exists():
            logging.info(
                "%s: actualizando catálogo con ambas colecciones Earth Search; se conservan los TIFF.",
                caso["id"],
            )
        items = buscar_escenas(
            bb_wgs84, (caso["inicio_contexto"], caso["fin_contexto"])
        )
        guardar_json(ruta_catalogo, dict(consulta=consulta, items=items))
    por_dia = {}
    for it in items:
        ts = pd.Timestamp(it["properties"]["datetime"])
        dia = ts.strftime("%Y-%m-%d")
        if not caso["inicio_contexto"] <= dia <= caso["fin_contexto"]:
            raise ValueError("El catálogo devolvió una fecha fuera del corte.")
        nube = float(it["properties"].get("eo:cloud_cover", 100))
        if dia not in por_dia or nube < por_dia[dia][0]:
            por_dia[dia] = (nube, it)
    if len(por_dia) < p["observaciones_sentinel_por_caso"]:
        raise ValueError(
            f"{caso['id']}: se encontraron {len(por_dia)} fechas distintas y se necesitan "
            f"{p['observaciones_sentinel_por_caso']} entre {caso['inicio_contexto']} y {caso['fin_contexto']}. "
            f"Fechas disponibles: {sorted(por_dia)}. Se consultaron ambas colecciones Earth Search. "
            "No se repiten imágenes para fabricar historia."
        )
    pendientes = set(por_dia)
    aceptadas = []
    intentos = []
    objetivos = pd.date_range(
        caso["inicio_contexto"],
        caso["fin_contexto"],
        periods=p["observaciones_sentinel_por_caso"],
    )
    for numero in range(min(len(pendientes), p["max_descargas_sentinel_por_caso"])):
        objetivo = objetivos[min(len(aceptadas), len(objetivos) - 1)]
        dia = min(
            pendientes,
            key=lambda d: (
                abs((pd.Timestamp(d) - objetivo).total_seconds()),
                por_dia[d][0],
                d,
            ),
        )
        pendientes.remove(dia)
        item = por_dia[dia][1]
        ts = datetime.fromisoformat(
            item["properties"]["datetime"].replace("Z", "+00:00")
        )
        archivo = destino / f"{dia}.tif"
        if not archivo.exists():
            escenas = resolver_escenas_earth(items, bb_wgs84, ts, destino)
            logging.info(
                "%s: Earth Search, leyendo muestra 64×64 de %s.", caso["id"], dia
            )
            arr = leer_escenas(escenas, lugar["bbox"], lugar["crs"], (64, 64))
            if arr.shape != (64, 64, 12) or not np.isfinite(arr).all():
                raise ValueError("Respuesta Sentinel de forma o valores inesperados.")
            temporal = archivo.with_suffix(".part")
            try:
                with rasterio.open(
                    temporal,
                    "w",
                    driver="GTiff",
                    height=64,
                    width=64,
                    count=12,
                    dtype="float32",
                    crs=lugar["crs"],
                    transform=Affine(*lugar["transform"]),
                    compress="deflate",
                    predictor=3,
                ) as dst:
                    dst.write(arr.transpose(2, 0, 1))
                    for i, b in enumerate(BANDAS + ["SCL", "dataMask"], 1):
                        dst.set_band_description(i, b)
                    dst.update_tags(
                        fecha_utc=dia,
                        fecha_hora_catalogo=item["properties"]["datetime"],
                        firma_plan=plan["firma_plan"],
                        unidades="reflectancia; etiquetas SCL y dataMask",
                        tipo="muestra fija de prueba",
                        proveedor=PROCEDENCIA,
                        radiometria=RADIOMETRIA,
                        ids_earth_search=json.dumps([e["id"] for e in escenas]),
                    )
                temporal.replace(archivo)
            finally:
                temporal.unlink(missing_ok=True)
        bien, clara = evaluar_tiff(archivo, lugar, p)
        intentos.append(
            dict(
                fecha=dia,
                fraccion_clara=clara,
                aceptada=bool(bien),
                id_catalogo=item["id"],
            )
        )
        guardar_json(destino / "calidad_candidatos.json", intentos)
        if bien:
            aceptadas.append((dia, archivo.name))
        if len(aceptadas) == p["observaciones_sentinel_por_caso"]:
            break
    if len(aceptadas) != p["observaciones_sentinel_por_caso"]:
        raise ValueError(
            f"{caso['id']}: solo {len(aceptadas)} fechas pasan calidad tras el límite de intentos. Revisa calidad_candidatos.json; no se cambian fechas ni se imputan imágenes."
        )
    aceptadas.sort()
    guardar_json(
        secuencia,
        dict(
            firma_plan=plan["firma_plan"],
            fechas=[x[0] for x in aceptadas],
            archivos=[x[1] for x in aceptadas],
            sha256={n: sha256(destino / n) for _, n in aceptadas},
            tipo="observaciones reales distintas; mismo terreno y malla de 10 m",
            procedencia="Observaciones Sentinel conservadas o Earth Search; proveedor en cada TIFF",
        ),
    )


def promediar_municipio(diario, lugar):
    # Pesos constantes de intersección con celdas nativas de 0.1°, medidos en una proyección equivalente.
    area = "+proj=laea +lat_0=19.2 +lon_0=-101.8 +datum=WGS84 +units=m +no_defs"
    gm = proyectar(shape(lugar["geometria_municipal"]), 4326, area)
    vals = diario.values
    pesos = np.zeros(vals.shape[1:], dtype=float)
    detalle = []
    for i, lat in enumerate(diario.latitude.values):
        for j, lon in enumerate(diario.longitude.values):
            celda = proyectar(
                box(
                    float(lon) - 0.05,
                    float(lat) - 0.05,
                    float(lon) + 0.05,
                    float(lat) + 0.05,
                ),
                4326,
                area,
            )
            a = gm.intersection(celda).area
            if a > 1:
                completa = bool(np.isfinite(vals[:, i, j]).all())
                if completa:
                    pesos[i, j] = a
                detalle.append(
                    dict(
                        latitud=float(lat),
                        longitud=float(lon),
                        interseccion_ha=a / 10000,
                        serie_completa=completa,
                    )
                )
    cobertura = float(pesos.sum() / gm.area)
    if cobertura < 0.99:
        raise ValueError(
            f"{lugar['municipio']}: celdas de lluvia completas cubren solo {cobertura:.1%} del municipio. No se extrapola la serie regional anterior."
        )
    buenos = pesos > 0
    media = np.average(vals[:, buenos], axis=1, weights=pesos[buenos])
    return (
        pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(diario.time.values).strftime("%Y-%m-%d"),
                "precipitacion_mm": media,
            }
        ),
        detalle,
        cobertura,
    )


def descargar_lluvia(cfg, plan):
    datos, _ = ubicaciones(cfg, plan)
    comun = datos / "era5_land"
    comun.mkdir(parents=True, exist_ok=True)
    union = unary_union(
        [shape(lugar["geometria_municipal"]) for lugar in plan["lugares"]]
    )
    oeste, sur, este, norte = union.bounds
    area = [
        round(math.ceil(norte * 10) / 10 + 0.1, 1),
        round(math.floor(oeste * 10) / 10 - 0.1, 1),
        round(math.floor(sur * 10) / 10 - 0.1, 1),
        round(math.ceil(este * 10) / 10 + 0.1, 1),
    ]
    cliente = None
    for anio in sorted({c["anio"] for c in plan["casos"]}):
        # Para los días 1–25 se solicitan los acumulados 00 UTC de los días 2–26.
        solicitud = dict(
            variable=["total_precipitation"],
            year=[str(anio)],
            month=["11"],
            day=[f"{d:02d}" for d in range(2, 27)],
            time=["00:00"],
            data_format="netcdf",
            download_format="unarchived",
            area=area,
        )
        ruta = comun / f"precipitacion_00utc_{anio}.nc"
        if not ruta.exists():
            if cliente is None:
                cliente = crear_cliente_cds()
            logging.info(
                "ERA5-Land %d: solicitando 25 días para la envolvente de los municipios piloto. Puede esperar en cola.",
                anio,
            )
            temp = ruta.with_suffix(".part")
            try:
                cliente.retrieve(DATASET_ERA5, solicitud, str(temp))
                abrir_netcdf(temp)
                temp.replace(ruta)
            finally:
                temp.unlink(missing_ok=True)
        diario = interpretar_precipitacion(abrir_netcdf(ruta))
        for lugar in plan["lugares"]:
            caso = next(
                c
                for c in plan["casos"]
                if c["CVEGEO"] == lugar["CVEGEO"] and c["anio"] == anio
            )
            d, _ = carpeta_caso(cfg, plan, caso)
            d.mkdir(parents=True, exist_ok=True)
            tabla, pesos, cobertura = promediar_municipio(diario, lugar)
            validar_lluvia(tabla, caso)
            tabla.to_csv(d / "lluvia.csv", index=False)
            pd.DataFrame(pesos).to_csv(d / "pesos_lluvia.csv", index=False)
            guardar_json(
                d / "lluvia_procedencia.json",
                dict(
                    firma_plan=plan["firma_plan"],
                    dataset=DATASET_ERA5,
                    solicitud=solicitud,
                    sha256_original=sha256(ruta),
                    fraccion_municipal_cubierta=cobertura,
                    unidades="mm/día",
                    calendario="UTC",
                    conversion="metros × 1000; fecha 00 UTC menos un día, aplicada una vez",
                    agregacion="Media ponderada por área de intersección municipal con celdas nativas; no lluvia por hectárea de aguacate.",
                ),
            )


def descargar(cfg, plan):
    for caso in plan["casos"]:
        lugar = next(
            lugar for lugar in plan["lugares"] if lugar["CVEGEO"] == caso["CVEGEO"]
        )
        descargar_sentinel(cfg, plan, caso, lugar)
    descargar_lluvia(cfg, plan)
    return dict(
        casos=len(plan["casos"]),
        fuentes="Sentinel-2 L2A y ERA5-Land reales; sin datos simulados",
    )
