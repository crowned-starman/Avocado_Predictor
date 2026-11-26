#!/usr/bin/env python3
"""Extrae índices Sentinel-2 por municipio-año desde el plan público Earth Search.

No necesita bucket S3: lee por HTTP Range los COG públicos de las escenas del
plan ya filtradas por SCL y conserva un SQLite para reanudar por recuadro/bloque.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape

import sentinel_s3
from cobertura_territorial import cargar_municipios
from utilidades import proyectar


BANDAS_REFLECTANCIA = ("B02", "B03", "B04", "B08")
CLASES_CLARAS = (4, 5, 6)


def crear_config_temporal(origen: Path, temporal: Path, malla: Path, metadatos: Path | None,
                          carpeta_trabajo: str, malla_municipal: bool = False,
                          piloto: bool = False) -> None:
    cfg = json.loads(origen.read_text(encoding="utf-8"))
    cfg.setdefault("sentinel_s3", {})
    if piloto:
        claves_piloto = sorted({str(c).zfill(5) for c in cfg.get("piloto", {}).get("municipios_piloto", [])})
        claves_configuradas = {str(c).zfill(5) for c in cfg.get("territorio", {}).get("cvegeo_seleccionados", [])}
        if not claves_piloto or not set(claves_piloto) <= claves_configuradas:
            raise ValueError("piloto.municipios_piloto debe ser un subconjunto no vacío de los municipios configurados.")
        cfg["territorio"]["cvegeo_seleccionados"] = claves_piloto
        malla_municipal = True
    if malla_municipal:
        tabla_malla = pd.read_csv(malla, encoding="utf-8-sig")
        if "tipo_malla" not in tabla_malla or set(tabla_malla["tipo_malla"].dropna()) != {"municipal_regular"}:
            raise ValueError("Con --malla-municipal usa el CSV creado por generar_malla_sentinel.py.")
        cfg.setdefault("territorio", {})
        # La cuadrícula nueva sustituye la selección histórica de 49 cortes.
        # Cada ventana solo incluye celdas que intersectan el AOI; los índices
        # se enmascaran después con los polígonos municipales exactos.
        cfg["territorio"]["cortes_preseleccionados"] = None
        cfg["territorio"]["fraccion_minima_aoi"] = 1e-12
    cfg["sentinel_s3"].update({
        "archivo_malla": str(malla.resolve()),
        "archivo_metadatos_malla": str(metadatos.resolve()) if metadatos else "",
        "carpeta_trabajo": carpeta_trabajo,
    })
    cfg["sentinel_s3"].setdefault("seleccion_temporal", {})
    cfg["sentinel_s3"]["seleccion_temporal"]["modo"] = "mejor_por_bloque"
    # Catálogo/STAC y COG son públicos; se evita escribir datos o costes en S3.
    cfg["sentinel_s3"]["seleccion_temporal"]["usar_cache_s3"] = False
    temporal.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_banda_publica(item, corte, banda, max_pixeles):
    memoria, _ = sentinel_s3.construir_tiff(item, corte, banda, max_pixeles=max_pixeles)
    try:
        with memoria.open() as src:
            datos = src.read(1).astype(np.float32)
            valido = src.read_masks(1) > 0
            perfil = {"shape": datos.shape, "transform": src.transform, "crs": src.crs,
                      "nodata": src.nodata, "scales": src.scales, "offsets": src.offsets}
        return datos, valido, perfil
    finally:
        memoria.close()


def reflectancia(banda, datos, perfil):
    escala, offset = float(perfil["scales"][0]), float(perfil["offsets"][0])
    if not np.isfinite([escala, offset]).all() or escala <= 0:
        raise ValueError(f"Calibración inválida en {banda}.")
    return np.maximum(datos * escala + offset, 0).astype(np.float32)


def medir_item(item, corte, municipios, max_pixeles):
    datos, mascaras, perfiles = {}, {}, {}
    for banda in BANDAS_REFLECTANCIA + ("SCL",):
        arr, mask, perfil = leer_banda_publica(item, corte, banda, max_pixeles)
        datos[banda], mascaras[banda], perfiles[banda] = arr, mask, perfil
    ref = perfiles["B04"]
    for banda in BANDAS_REFLECTANCIA:
        p = perfiles[banda]
        if p["shape"] != ref["shape"] or p["transform"] != ref["transform"] or p["crs"] != ref["crs"]:
            raise ValueError("Bandas reflectivas no comparten la rejilla de B04.")
    scl_p = perfiles["SCL"]
    scl_10m = np.zeros(ref["shape"], dtype=np.uint8)
    scl_mask = np.zeros(ref["shape"], dtype=np.uint8)
    reproject(datos["SCL"].astype(np.uint8), scl_10m,
              src_transform=scl_p["transform"], src_crs=scl_p["crs"],
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    reproject(mascaras["SCL"].astype(np.uint8), scl_mask,
              src_transform=scl_p["transform"], src_crs=scl_p["crs"],
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    valido = np.logical_and.reduce([mascaras[b] for b in BANDAS_REFLECTANCIA])
    valido &= (scl_mask > 0) & np.isin(scl_10m, CLASES_CLARAS)
    b02, b03, b04, b08 = [reflectancia(b, datos[b], perfiles[b]) for b in BANDAS_REFLECTANCIA]
    ndvi = np.divide(b08 - b04, b08 + b04, out=np.full(ref["shape"], np.nan, np.float32), where=np.abs(b08 + b04) > 1e-6)
    evi_den = b08 + 6 * b04 - 7.5 * b02 + 1
    evi = np.divide(2.5 * (b08 - b04), evi_den, out=np.full(ref["shape"], np.nan, np.float32), where=np.abs(evi_den) > 1e-6)
    savi_den = b08 + b04 + 0.5
    savi = np.divide(1.5 * (b08 - b04), savi_den, out=np.full(ref["shape"], np.nan, np.float32), where=np.abs(savi_den) > 1e-6)
    ndwi_den = b03 + b08
    ndwi = np.divide(b03 - b08, ndwi_den, out=np.full(ref["shape"], np.nan, np.float32), where=np.abs(ndwi_den) > 1e-6)
    scene_day = sentinel_s3.fecha_escena(item).date().isoformat()
    rows = []
    bbox_raster = rasterio.transform.array_bounds(*ref["shape"], ref["transform"])
    from shapely.geometry import box
    extent = box(*bbox_raster)
    for clave, nombre, geometria in municipios:
        geom = proyectar(geometria, 4326, ref["crs"])
        if not geom.intersects(extent):
            continue
        dentro = geometry_mask([mapping(geom)], out_shape=ref["shape"], transform=ref["transform"], invert=True)
        m = dentro & valido
        if not m.any():
            continue
        vals = []
        for index in (ndvi, evi, savi, ndwi):
            x = index[m]
            x = x[np.isfinite(x)]
            vals.append(float(np.median(x)) if len(x) else np.nan)
        if np.isfinite(vals).all():
            rows.append({"CVEGEO": str(clave).zfill(5), "Anio": int(scene_day[:4]),
                         "fecha_escena": scene_day, "NDVI": vals[0], "EVI": vals[1],
                         "SAVI": vals[2], "NDWI": vals[3], "pixeles_validos": int(m.sum())})
    return rows


def inicializar_db(path: Path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS procesados (
        id TEXT PRIMARY KEY, anio INTEGER NOT NULL, corte INTEGER NOT NULL, bloque TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS resumen (
        id TEXT NOT NULL, CVEGEO TEXT NOT NULL, Anio INTEGER NOT NULL, fecha_escena TEXT NOT NULL,
        NDVI REAL NOT NULL, EVI REAL NOT NULL, SAVI REAL NOT NULL, NDWI REAL NOT NULL,
        pixeles_validos INTEGER NOT NULL, PRIMARY KEY(id,CVEGEO,fecha_escena))""")
    con.commit()
    return con


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuracion", type=Path, default=Path("configuracion.json"))
    parser.add_argument("--anio", type=int, required=True, choices=range(2020, 2026))
    parser.add_argument("--malla", type=Path, required=True, help="CSV de la cuadrícula georreferenciada.")
    parser.add_argument("--malla-municipal", action="store_true",
                        help="Interpreta la malla como la cuadrícula generada desde los polígonos municipales; ignora los 49 cortes históricos.")
    parser.add_argument("--piloto", action="store_true",
                        help="Procesa solo los municipios de piloto.municipios_piloto y guarda la salida en datos/sentinel_piloto/.")
    parser.add_argument("--metadatos-malla", type=Path, default=None, help="JSON con CRS si el CSV no trae la columna crs.")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()
    root = args.configuracion.resolve().parent
    if not args.malla.is_file():
        parser.error(f"No encuentro el CSV de la malla: {args.malla}")
    trabajo_cfg = root / f".config_sentinel_features_{args.anio}.json"
    modo_malla_municipal = args.malla_municipal or args.piloto
    nombre_trabajo = ("resultados_sentinel_piloto_features" if args.piloto else "resultados_sentinel_features")
    crear_config_temporal(args.configuracion.resolve(), trabajo_cfg, args.malla, args.metadatos_malla,
                          f"{nombre_trabajo}_{args.anio}", modo_malla_municipal, args.piloto)
    try:
        cfg = sentinel_s3.configuracion(trabajo_cfg, anio=args.anio)
        if modo_malla_municipal:
            # Reafirmar el modo municipal tras normalizar la configuración y antes
            # de crear la selección; nunca reutilizar los IDs de los 49 recuadros viejos.
            cfg["territorio"]["cortes_preseleccionados"] = None
            cfg["territorio"]["fraccion_minima_aoi"] = 1e-12
            modo = "piloto" if args.piloto else "municipal"
            print(f"Malla {modo}: usar todas las celdas que intersectan el AOI; se ignoran los 49 cortes históricos.")
            if args.piloto:
                print(f"Municipios del piloto: {', '.join(sorted(cfg['territorio']['cvegeo_seleccionados']))}")
        sentinel_s3.preparar(cfg)
        plan = sentinel_s3.planificar(cfg, usar_plan_compartido=False)
        claves = {str(c).zfill(5) for c in cfg["territorio"]["cvegeo_seleccionados"]}
        municipios = [m for m in cargar_municipios(root, claves) if str(m[0]).zfill(5) in claves]
        if {str(m[0]).zfill(5) for m in municipios} != claves:
            raise ValueError("No se pudieron cargar todos los polígonos municipales configurados.")
        cortes = {int(x["corte"]): x for x in plan["seleccion"]["recuadros"]}
        por_bloque = plan.get("seleccion_temporal", {}).get("modo") == "mejor_por_bloque"
        cfg["sentinel_s3"]["prefijo"] = cfg["sentinel_s3"].get("prefijo", "aguacate/sentinel2")
        carpeta_salida = "sentinel_piloto" if args.piloto else "sentinel"
        salida = args.salida or root / "datos" / carpeta_salida / f"features_sentinel_municipio_{args.anio}.csv"
        salida.parent.mkdir(parents=True, exist_ok=True)
        db = inicializar_db(salida.with_suffix(".sqlite"))
        max_pix = int(cfg["sentinel_s3"].get("max_pixeles_por_banda", 16_000_000))
        prefijo = cfg["sentinel_s3"]["prefijo"].strip("/")
        total = 0
        for grupo, items in sentinel_s3.grupos_plan(cfg, plan, por_bloque):
            corte_id = int(grupo["corte"])
            bloque = str(grupo.get("bloque_id", items[0]["id"]))
            huella = hashlib.sha256(json.dumps(["indice-v1", plan["plan_id"], args.anio,
                                                corte_id, bloque, [i["id"] for i in items]],
                                               sort_keys=True).encode()).hexdigest()
            if db.execute("SELECT 1 FROM procesados WHERE id=?", (huella,)).fetchone():
                continue
            filas = []
            for item in items:
                filas.extend(medir_item(item, cortes[corte_id], municipios, max_pix))
            if filas:
                df = pd.DataFrame(filas).groupby(["CVEGEO", "Anio", "fecha_escena"], as_index=False).agg(
                    NDVI=("NDVI", "median"), EVI=("EVI", "median"), SAVI=("SAVI", "median"),
                    NDWI=("NDWI", "median"), pixeles_validos=("pixeles_validos", "sum"))
                db.executemany(
                    "INSERT OR REPLACE INTO resumen VALUES (?,?,?,?,?,?,?,?,?)",
                    [(huella, r.CVEGEO, int(r.Anio), r.fecha_escena, float(r.NDVI), float(r.EVI),
                      float(r.SAVI), float(r.NDWI), int(r.pixeles_validos)) for r in df.itertuples(index=False)])
            db.execute("INSERT OR REPLACE INTO procesados VALUES (?,?,?,?)",
                       (huella, args.anio, corte_id, bloque))
            db.commit()
            total += 1
            if total % 25 == 0:
                print(f"Bloques procesados esta sesión: {total}; caché recuperable: {db.execute('SELECT COUNT(*) FROM procesados').fetchone()[0]}", flush=True)
        salida_df = pd.read_sql_query(
            "SELECT CVEGEO, Anio, fecha_escena, NDVI, EVI, SAVI, NDWI FROM resumen WHERE Anio=?",
            db, params=(args.anio,))
        # Consolidar escenas a municipio-año usando la mediana temporal de los índices.
        if not salida_df.empty:
            salida_df = salida_df.groupby(["CVEGEO", "Anio", "fecha_escena"], as_index=False).median(numeric_only=True)
            salida_df = salida_df.rename(columns={"NDVI": "ndvi_mediana", "EVI": "evi_mediana",
                                                  "SAVI": "savi_mediana", "NDWI": "ndwi_mediana"})
            salida_df = salida_df.groupby(["CVEGEO", "Anio"], as_index=False).agg(
                ndvi_mediana=("ndvi_mediana", "median"), evi_mediana=("evi_mediana", "median"),
                savi_mediana=("savi_mediana", "median"), ndwi_mediana=("ndwi_mediana", "median"),
                sentinel_escenas_validas=("fecha_escena", "nunique"))
            salida_df.to_csv(salida, index=False, encoding="utf-8-sig", float_format="%.8g")
        db.close()
        if salida_df.empty:
            raise ValueError("No hubo píxeles claros que intersectaran municipios seleccionados.")
        print(f"Guardado {salida}: {len(salida_df)} municipios con índices Sentinel-2 de {args.anio}.")
        print("NDWI = (Green - NIR)/(Green + NIR); el proxy cubre vegetación general dentro de los recuadros, no huertas.")
    finally:
        trabajo_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

