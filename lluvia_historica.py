"""Histórico local de precipitación ERA5-Land para el proyecto de aguacate.

Ejemplos (desde la raíz del proyecto):
    python lluvia_historica.py estimar
    python lluvia_historica.py todo
    python lluvia_historica.py todo --anio 2022
    python lluvia_historica.py descargar --anio 2020
    python lluvia_historica.py procesar --anio 2020
    python lluvia_historica.py estado

Conserva originales horarios de CDS y genera diarios UTC y tablas municipales.
No modifica la configuración del proyecto ni el código de Sentinel.

Instalación: python -m pip install -r requirements-lluvia.txt
Colocar este script junto a configuracion.json, datos_era5.py,
cobertura_territorial.py y utilidades.py del proyecto existente.

Acceso CDS: https://cds.climate.copernicus.eu/how-to-api
Aceptar primero la licencia en la página del dataset reanalysis-era5-land.
Se reutiliza CDS_API_KEY del proyecto/entorno o la configuración .cdsapirc.
Si falta, se pide el token con entrada oculta. Las claves AWS no se necesitan.

Salida por defecto: datos/lluvia_era5_land. Repetir el mismo comando reutiliza
meses completos verificados. Los originales horarios contienen acumulados;
para las series diarias se toma SOLO 00 UTC del día siguiente y se convierte
de metros a mm. El borde del 1 de enero siguiente cierra el 31 de diciembre.
Los promedios municipales usan áreas de intersección celda-polígono; con menos
de 95 % del área con dato se registra un faltante, y no lluvia cero.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import logging
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely import segmentize
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from cobertura_territorial import AREA, cargar_municipios
from datos_era5 import crear_cliente_cds, interpretar_precipitacion
from utilidades import guardar_json, leer_json, proyectar, sha256

RAIZ = Path(__file__).resolve().parent
DATASET = "reanalysis-era5-land"
VERSION = "lluvia-local-v1"
PASO = 0.1
COBERTURA_MINIMA = 0.95
HORAS = [f"{h:02d}:00" for h in range(24)]
FUENTES = {
    "dataset": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land",
    "api": "https://cds.climate.copernicus.eu/how-to-api",
    "documentacion": "https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation",
}


def firma(objeto):
    contenido = json.dumps(objeto, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(contenido.encode()).hexdigest()


def siguiente_mes(fecha):
    return (fecha.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def zona_proyecto(config):
    config = Path(config).resolve()
    cfg = leer_json(config)
    claves = sorted(str(c).zfill(5) for c in cfg.get("territorio", {}).get("cvegeo_seleccionados", []))
    if not claves or len(set(claves)) != len(claves):
        raise ValueError("La configuración necesita una selección municipal no vacía y sin duplicados.")
    municipios = [m for m in cargar_municipios(config.parent, claves) if m[0] in claves]
    union = unary_union([g for _, _, g in municipios])
    oeste, sur, este, norte = union.bounds
    # Incluir los centros de las celdas que tocan los bordes de los municipios.
    area = [math.ceil((norte + .05) * 10) / 10,
            math.floor((oeste - .05) * 10) / 10,
            math.floor((sur - .05) * 10) / 10,
            math.ceil((este + .05) * 10) / 10]
    fuente = config.parent / "Delimitacion_rendimiento_aguacate/fuentes/michoacan_inegi_original.json"
    zona = {"version": VERSION, "dataset": DATASET, "area_cds_norte_oeste_sur_este": area,
            "paso_grados": PASO, "cvegeo": claves, "municipios_sha256": sha256(fuente),
            "unidad_territorial": "municipios completos seleccionados; no mascara de huertas",
            "fuentes": FUENTES}
    zona["firma_zona"] = firma(zona)
    return zona, municipios


def coordenadas(zona):
    n, w, s, e = zona["area_cds_norte_oeste_sur_este"]
    lat = np.round(np.arange(round(s * 10), round(n * 10) + 1) / 10, 8)
    lon = np.round(np.arange(round(w * 10), round(e * 10) + 1) / 10, 8)
    return lat, lon


def peticion(zona, fecha, borde=False):
    return {"variable": ["total_precipitation"], "year": [f"{fecha.year:04d}"],
            "month": [f"{fecha.month:02d}"],
            "day": ([f"{fecha.day:02d}"] if borde else
                    [f"{d:02d}" for d in range(1, calendar.monthrange(fecha.year, fecha.month)[1] + 1)]),
            "time": ["00:00"] if borde else HORAS.copy(),
            "data_format": "netcdf", "download_format": "unarchived",
            "area": zona["area_cds_norte_oeste_sur_este"]}


def tareas(zona, inicio, fin):
    trabajos = []
    for anio in range(inicio, fin + 1):
        for mes in range(1, 13):
            fecha = dt.date(anio, mes, 1)
            trabajos.append({"archivo": f"originales/{anio}/era5_land_tp_{anio}_{mes:02d}.nc",
                             "peticion": peticion(zona, fecha), "borde": False})
    fecha = dt.date(fin + 1, 1, 1)
    trabajos.append({"archivo": f"bordes/tp_{fecha:%Y%m%d}_00utc.nc",
                     "peticion": peticion(zona, fecha, borde=True), "borde": True})
    return trabajos


def estimacion(zona, inicio, fin):
    lat, lon = coordenadas(zona)
    dias = (dt.date(fin + 1, 1, 1) - dt.date(inicio, 1, 1)).days
    n = len(lat) * len(lon)
    original32 = (dias * 24 + 1) * n * 4
    diarios32 = dias * n * 4
    return {"periodo": [f"{inicio}-01-01", f"{fin}-12-31"],
            "municipios": len(zona["cvegeo"]), "dias": dias,
            "latitudes": len(lat), "longitudes": len(lon), "celdas_rectangulo": n,
            "peticiones_cds": (fin - inicio + 1) * 12 + 1,
            "originales_float32_MB_sin_compresion": round(original32 / 1e6, 3),
            "originales_float64_MB_sin_compresion": round(original32 * 2 / 1e6, 3),
            "diarios_float32_MB_sin_compresion": round(diarios32 / 1e6, 3),
            "filas_diarias_municipales": dias * len(zona["cvegeo"]),
            "nota": "Calculo de valores numericos; NetCDF, compresion, tablas y metadatos cambian el tamano real. "
                    "No incluye las bibliotecas de Python. El borde final solo completa la lluvia del 31 de diciembre."}


def asegurar_destino(salida, zona, municipios):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / "zona.json"
    if ruta.is_file() and leer_json(ruta).get("firma_zona") != zona["firma_zona"]:
        raise ValueError("La carpeta contiene otra zona o selección. Elige otra carpeta con --salida; "
                         "se conservan los archivos existentes.")
    guardar_json(ruta, zona)
    guardar_json(salida / "municipios.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"CVEGEO": c, "municipio": nombre}, "geometry": mapping(g)}
        for c, nombre, g in municipios]})


def normalizar_tp(ds):
    nombres = [n for n in ("tp", "total_precipitation") if n in ds.data_vars]
    if len(nombres) != 1:
        raise ValueError("La respuesta no contiene una única variable tp/total_precipitation.")
    tp = ds[nombres[0]]
    if "valid_time" in tp.dims:
        tp = tp.rename({"valid_time": "time"})
    if not {"time", "latitude", "longitude"} <= set(tp.dims):
        raise ValueError("NetCDF inesperado: se requieren tiempo, latitud y longitud.")
    for dim in set(tp.dims) - {"time", "latitude", "longitude"}:
        if tp.sizes[dim] != 1:
            raise ValueError(f"Dimensión {dim} no resuelta. Se conserva la respuesta para revisión.")
        tp = tp.isel({dim: 0}, drop=True)
    unidades = str(tp.attrs.get("units", "")).strip().lower()
    if unidades not in ("m", "metres", "meters", "m of water equivalent"):
        raise ValueError(f"Se esperaba precipitación original en metros, no {unidades!r}.")
    tp = tp.assign_coords(longitude=(tp.longitude + 180) % 360 - 180)
    tp = tp.sortby("longitude").sortby("latitude").sortby("time")
    fechas = pd.DatetimeIndex(tp.time.values)
    if fechas.has_duplicates or fechas.hasnans:
        raise ValueError("El archivo contiene horas duplicadas o inválidas.")
    for dim in ("latitude", "longitude"):
        coord = np.asarray(tp[dim], dtype=float)
        if len(coord) < 2 or not np.allclose(np.diff(coord), .1, atol=1e-4, rtol=0):
            raise ValueError("Se requiere la rejilla de 0.1 grados de CDS, sin remuestrear.")
    if not np.isfinite(tp.values).any() or float(tp.min(skipna=True)) < -1e-6:
        raise ValueError("Precipitación ausente o negativa fuera de tolerancia.")
    return tp.transpose("time", "latitude", "longitude").rename("tp")


def abrir_tp(ruta):
    with xr.open_dataset(ruta, engine="netcdf4") as ds:
        return normalizar_tp(ds).load()


def horas_esperadas(request):
    fechas = [f"{request['year'][0]}-{request['month'][0]}-{d} {h}"
              for d in request["day"] for h in request["time"]]
    return pd.DatetimeIndex(pd.to_datetime(fechas)).sort_values()


def validar_archivo(ruta, trabajo, zona):
    tp = abrir_tp(ruta)
    if not pd.DatetimeIndex(tp.time.values).equals(horas_esperadas(trabajo["peticion"])):
        raise ValueError("La respuesta CDS está incompleta o contiene horas diferentes de las solicitadas.")
    lat, lon = coordenadas(zona)
    for dim, esperada in (("latitude", lat), ("longitude", lon)):
        if tp.sizes[dim] != len(esperada) or not np.allclose(tp[dim], esperada, atol=1e-4, rtol=0):
            raise ValueError(f"La rejilla recibida no coincide con el área solicitada: {dim}.")
    return tp


def disponible(ruta, trabajo, zona):
    ruta = Path(ruta)
    if not ruta.is_file():
        return False
    recibo = ruta.with_suffix(".json")
    if recibo.is_file():
        info = leer_json(recibo)
        if info.get("firma_peticion") != firma(trabajo["peticion"]) or info.get("firma_zona") != zona["firma_zona"]:
            raise ValueError(f"El archivo existente corresponde a otra solicitud: {ruta}.")
        if info.get("sha256") != sha256(ruta):
            raise ValueError(f"El SHA256 no coincide: {ruta}. Revisa ese archivo antes de continuar.")
    validar_archivo(ruta, trabajo, zona)
    return True


def extraer_respuesta(temporal):
    """Admitir respuestas CDS empaquetadas sin crear carpetas externas."""
    if not zipfile.is_zipfile(temporal):
        return
    extraido = temporal.with_suffix(".extraido.part")
    try:
        with zipfile.ZipFile(temporal) as z:
            miembros = [n for n in z.namelist() if n.lower().endswith((".nc", ".nc4", ".netcdf"))]
            if len(miembros) != 1:
                raise ValueError("La respuesta CDS contiene varios NetCDF; se requiere revisar su estructura.")
            with z.open(miembros[0]) as entrada, extraido.open("wb") as destino:
                shutil.copyfileobj(entrada, destino)
        extraido.replace(temporal)
    finally:
        extraido.unlink(missing_ok=True)


def descargar(salida, zona, inicio, fin, cliente=None):
    completados, reutilizados = 0, 0
    lista = tareas(zona, inicio, fin)
    for numero, trabajo in enumerate(lista, 1):
        ruta = Path(salida) / trabajo["archivo"]
        if disponible(ruta, trabajo, zona):
            reutilizados += 1
            logging.info("[%s/%s] Reutilizado: %s", numero, len(lista), trabajo["archivo"])
            continue
        # Si se descargó el siguiente enero completo, no pedir su primera hora otra vez.
        if trabajo["borde"]:
            fecha = dt.date.fromisoformat(f"{trabajo['peticion']['year'][0]}-01-01")
            mensual = Path(salida) / f"originales/{fecha.year}/era5_land_tp_{fecha.year}_01.nc"
            if mensual.is_file():
                completo = {"peticion": peticion(zona, fecha)}
                if disponible(mensual, completo, zona):
                    reutilizados += 1
                    logging.info("Borde final disponible en el siguiente enero ya descargado.")
                    continue
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_suffix(".nc.part")
        temporal.unlink(missing_ok=True)
        if cliente is None:
            cliente = crear_cliente_cds()
        logging.info("[%s/%s] Solicitando a CDS: %s. La cola del servicio puede tardar.",
                     numero, len(lista), trabajo["archivo"])
        cliente.retrieve(DATASET, trabajo["peticion"], str(temporal))
        extraer_respuesta(temporal)
        validar_archivo(temporal, trabajo, zona)
        temporal.replace(ruta)
        guardar_json(ruta.with_suffix(".json"), {"dataset": DATASET, "peticion": trabajo["peticion"],
            "firma_peticion": firma(trabajo["peticion"]), "firma_zona": zona["firma_zona"],
            "sha256": sha256(ruta), "bytes": ruta.stat().st_size,
            "descarga_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "interpretacion": "tp acumulada desde 00 UTC; NO sumar las 24 muestras acumuladas."})
        completados += 1
        logging.info("Guardado: %s (%.2f MB)", trabajo["archivo"], ruta.stat().st_size / 1e6)
    return {"archivos_nuevos": completados, "archivos_reutilizados": reutilizados}


def pesos_municipales(zona, municipios):
    lat, lon = coordenadas(zona)
    poligonos = [proyectar(segmentize(box(x-.05, y-.05, x+.05, y+.05), .01), 4326, AREA)
                 for y in lat for x in lon]
    pesos, filas = [], []
    for clave, nombre, geom in municipios:
        gm = proyectar(geom, 4326, AREA)
        areas = np.array([gm.intersection(p).area if gm.intersects(p) else 0. for p in poligonos])
        cobertura = float(areas.sum() / gm.area)
        if not .999 <= cobertura <= 1.001:
            raise ValueError(f"La rejilla no cubre el municipio {clave}: {cobertura:.3%}.")
        peso = areas / areas.sum()
        pesos.append(peso)
        for k in np.flatnonzero(peso):
            iy, ix = divmod(int(k), len(lon))
            filas.append({"CVEGEO": clave, "municipio": nombre, "latitud": lat[iy],
                          "longitud": lon[ix], "area_interseccion_m2": areas[k], "peso": peso[k]})
    return np.asarray(pesos), pd.DataFrame(filas)


def primera_hora_siguiente(salida, zona, fecha):
    ruta = Path(salida) / f"originales/{fecha.year}/era5_land_tp_{fecha.year}_{fecha.month:02d}.nc"
    if ruta.is_file():
        validar_archivo(ruta, {"peticion": peticion(zona, fecha)}, zona)
    else:
        ruta = Path(salida) / f"bordes/tp_{fecha:%Y%m%d}_00utc.nc"
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el borde de cierre {fecha}: vuelve a ejecutar descargar para ese año.")
        validar_archivo(ruta, {"peticion": peticion(zona, fecha, borde=True)}, zona)
    return abrir_tp(ruta).sel(time=[pd.Timestamp(fecha)])


def diario_mes(salida, zona, anio, mes):
    fecha = dt.date(anio, mes, 1)
    ruta = Path(salida) / f"originales/{anio}/era5_land_tp_{anio}_{mes:02d}.nc"
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta descargar o todo para ese año.")
    tp = validar_archivo(ruta, {"peticion": peticion(zona, fecha)}, zona)
    fin = siguiente_mes(fecha)
    borde = primera_hora_siguiente(salida, zona, fin)
    # Se usa UN acumulado de 24 h por día: 00 UTC del día siguiente.
    horas = pd.DatetimeIndex(tp.time.values)
    ceros = tp.isel(time=(horas.hour == 0) & (horas.date > fecha))
    acumulados = xr.concat([ceros, borde], dim="time", join="exact")
    diario = interpretar_precipitacion(acumulados.to_dataset())
    esperado = pd.date_range(fecha, fin - dt.timedelta(days=1), freq="D")
    if not pd.DatetimeIndex(diario.time.values).equals(esperado):
        raise ValueError(f"Calendario diario incompleto en {anio}-{mes:02d}.")
    return diario


def tabla_municipal(diario, municipios, pesos, cobertura_minima=COBERTURA_MINIMA):
    valores = np.asarray(diario.values, dtype=np.float64).reshape(diario.sizes["time"], -1)
    validos = np.isfinite(valores)
    cobertura = validos.astype(float) @ pesos.T
    suma = np.where(validos, valores, 0.) @ pesos.T
    media = np.divide(suma, cobertura, out=np.full(suma.shape, np.nan), where=cobertura > 0)
    media[cobertura < cobertura_minima - 1e-10] = np.nan
    dias = pd.DatetimeIndex(diario.time.values)
    filas = []
    for j, (clave, nombre, _) in enumerate(municipios):
        filas.append(pd.DataFrame({"CVEGEO": clave, "municipio": nombre,
            "fecha": dias.strftime("%Y-%m-%d"), "precipitacion_diaria_mm": media[:, j],
            "fraccion_area_con_dato": np.clip(cobertura[:, j], 0, 1)}))
    return pd.concat(filas, ignore_index=True).sort_values(["CVEGEO", "fecha"])


def resumir_periodos(diaria, frecuencia):
    t = diaria.copy()
    fechas = pd.to_datetime(t["fecha"])
    if frecuencia == "bloque":
        numero = np.minimum((fechas.dt.day - 1) // 7 + 1, 4)
        t["periodo"] = fechas.dt.strftime("%Y-%m") + "-B" + numero.astype(str)
    elif frecuencia == "mes":
        t["periodo"] = fechas.dt.strftime("%Y-%m")
    elif frecuencia == "anio":
        t["periodo"] = fechas.dt.strftime("%Y")
    else:
        raise ValueError("Frecuencia no admitida.")
    filas = []
    for (clave, nombre, periodo), grupo in t.groupby(["CVEGEO", "municipio", "periodo"], sort=True):
        validos = int(grupo.precipitacion_diaria_mm.notna().sum())
        completo = validos == len(grupo)
        filas.append({"CVEGEO": clave, "municipio": nombre, "periodo": periodo,
            "fecha_inicial": grupo.fecha.min(), "fecha_final": grupo.fecha.max(),
            "dias_esperados": len(grupo), "dias_con_dato": validos,
            "precipitacion_total_mm": float(grupo.precipitacion_diaria_mm.sum()) if completo else np.nan,
            "completo": completo})
    return pd.DataFrame(filas)


def escribir_csv(tabla, ruta):
    temporal = Path(ruta).with_suffix(".csv.part")
    tabla.to_csv(temporal, index=False, encoding="utf-8-sig", float_format="%.8g")
    temporal.replace(ruta)


def procesar(salida, zona, municipios, inicio, fin):
    pesos, tabla_pesos = pesos_municipales(zona, municipios)
    escribir_csv(tabla_pesos, Path(salida) / "pesos_municipales.csv")
    informes = []
    for anio in range(inicio, fin + 1):
        logging.info("Preparando diarios UTC y tablas municipales de %s.", anio)
        partes = [diario_mes(salida, zona, anio, mes) for mes in range(1, 13)]
        diario = xr.concat(partes, dim="time", join="exact")
        fechas = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
        if not pd.DatetimeIndex(diario.time.values).equals(fechas):
            raise ValueError(f"Calendario anual incompleto: {anio}.")
        diario.attrs.update(units="mm", zona_horaria="UTC",
            procedencia=DATASET, firma_zona=zona["firma_zona"],
            interpretacion="Acumulado de las 00 UTC del dia siguiente; no suma de 24 acumulados.")
        tabla = tabla_municipal(diario, municipios, pesos)
        carpeta = Path(salida) / "derivados" / str(anio)
        carpeta.mkdir(parents=True, exist_ok=True)
        nc = carpeta / "precipitacion_diaria_mm.nc"
        temporal = nc.with_suffix(".nc.part")
        diario.to_dataset().to_netcdf(temporal, engine="netcdf4",
            encoding={"precipitacion_diaria_mm": {"dtype": "float32", "zlib": True, "complevel": 4}})
        temporal.replace(nc)
        escribir_csv(tabla, carpeta / "lluvia_diaria_municipio.csv")
        for nombre, frecuencia in (("lluvia_bloques_municipio.csv", "bloque"),
                                  ("lluvia_mensual_municipio.csv", "mes"),
                                  ("lluvia_anual_municipio.csv", "anio")):
            escribir_csv(resumir_periodos(tabla, frecuencia), carpeta / nombre)
        # Contrato de entrada tabular usado después por Chronos-2. Un id por municipio.
        chronos = tabla[["CVEGEO", "fecha", "precipitacion_diaria_mm"]].rename(columns={
            "CVEGEO": "id", "fecha": "timestamp", "precipitacion_diaria_mm": "target"})
        escribir_csv(chronos, carpeta / "chronos2_diario.csv")
        informe = {"version": VERSION, "anio": anio, "firma_zona": zona["firma_zona"],
            "dias": len(fechas), "municipios": len(municipios), "filas_diarias": len(tabla),
            "filas_sin_lluvia_por_cobertura_insuficiente": int(tabla.precipitacion_diaria_mm.isna().sum()),
            "cobertura_minima_area_con_dato": COBERTURA_MINIMA,
            "unidades": "mm", "zona_horaria": "UTC",
            "bloques": "1-7, 8-14, 15-21 y 22-fin; mismos identificadores mensuales que Sentinel",
            "promedio_municipal": "Pesos por area real de interseccion celda-municipio en proyeccion de igual area.",
            "archivos": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(carpeta.iterdir()) if p.suffix in (".nc", ".csv")}}
        guardar_json(carpeta / "informe.json", informe)
        informes.append(informe)
        logging.info("%s listo: %s días, %s municipios; %s filas sin dato suficiente.",
                     anio, len(fechas), len(municipios), informe["filas_sin_lluvia_por_cobertura_insuficiente"])
    return informes


def inventario(salida):
    salida = Path(salida)
    archivos = [{"archivo": p.relative_to(salida).as_posix(), "bytes": p.stat().st_size}
                for p in sorted(salida.rglob("*")) if p.is_file() and not p.name.endswith(".part")
                and p.name != "inventario_local.json"]
    return {"archivos": archivos, "total_bytes": sum(a["bytes"] for a in archivos),
            "total_MB": round(sum(a["bytes"] for a in archivos) / 1e6, 3)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accion", choices=["estimar", "descargar", "procesar", "todo", "estado"])
    parser.add_argument("--config", type=Path, default=RAIZ / "configuracion.json")
    parser.add_argument("--inicio", type=int, default=2020, help="Primer año; por defecto 2020.")
    parser.add_argument("--fin", type=int, default=2025, help="Último año; por defecto 2025.")
    parser.add_argument("--anio", type=int, help="Procesar únicamente este año.")
    parser.add_argument("--salida", type=Path, help="Carpeta local; por defecto datos/lluvia_era5_land.")
    args = parser.parse_args(argv)
    inicio, fin = (args.anio, args.anio) if args.anio is not None else (args.inicio, args.fin)
    if not 2020 <= inicio <= fin <= 2025:
        parser.error("Este histórico de desarrollo comprende 2020–2025. 2026 se reserva para evaluación.")
    config = args.config.resolve()
    salida = args.salida.resolve() if args.salida else config.parent / "datos/lluvia_era5_land"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    try:
        zona, municipios = zona_proyecto(config)
        calculo = estimacion(zona, inicio, fin)
        print(json.dumps({"carpeta_local": str(salida), **calculo}, ensure_ascii=False, indent=2), flush=True)
        if args.accion == "estimar":
            return 0
        if args.accion == "estado":
            print(json.dumps(inventario(salida), ensure_ascii=False, indent=2))
            return 0
        asegurar_destino(salida, zona, municipios)
        guardar_json(salida / f"plan_{inicio}_{fin}.json", {
            "version": VERSION, "firma_zona": zona["firma_zona"], "estimacion": calculo,
            "tareas": tareas(zona, inicio, fin),
            "nota_borde": "La hora 00 UTC del 1 de enero siguiente completa el 31 de diciembre; "
                          "no añade días de 2026 al conjunto de desarrollo."})
        if args.accion in ("descargar", "todo"):
            resultado = descargar(salida, zona, inicio, fin)
            print(json.dumps(resultado, ensure_ascii=False, indent=2), flush=True)
        if args.accion in ("procesar", "todo"):
            procesar(salida, zona, municipios, inicio, fin)
        datos = inventario(salida)
        guardar_json(salida / "inventario_local.json", datos)
        print(f"Terminado. Datos locales: {salida}\nTamaño actual: {datos['total_MB']:.2f} MB", flush=True)
        return 0
    except KeyboardInterrupt:
        logging.warning("Ejecución interrumpida. Repite el mismo comando para reutilizar los meses completos.")
        return 130
    except Exception as e:
        logging.error("%s: %s", type(e).__name__, e)
        logging.error("Se conservan los archivos completos. Verifica token CDS, licencia aceptada y el mensaje anterior.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
