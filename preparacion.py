"""Adaptadores de los GeoTIFF y el NetCDF/CSV producidos por el notebook aprobado."""

from __future__ import annotations

from utilidades import BANDAS, guardar_json as escribir_json

import hashlib
import json
import logging
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def huella_config(cfg: dict, seccion: str) -> str:
    relevante = {
        "datos": cfg["carpeta_datos"],
        "raiz": cfg["_raiz"],
        "seccion": cfg[seccion],
    }
    return hashlib.sha256(json.dumps(relevante, sort_keys=True).encode()).hexdigest()


def resolver(ruta: str, base: Path) -> Path:
    p = Path(ruta).expanduser()
    return p if p.is_absolute() else base / p


def mapa_bandas(src) -> dict[str, int]:
    nombres = [(s or "").strip() for s in src.descriptions]
    requeridas = BANDAS + ["SCL", "dataMask"]
    if any(nombres.count(b) != 1 for b in requeridas):
        raise ValueError(
            f"{src.name}: se necesitan las diez bandas identificadas, SCL y dataMask, sin duplicados."
        )
    return {b: nombres.index(b) + 1 for b in requeridas}


def archivos_sentinel(cfg: dict) -> tuple[list[Path], bool]:
    explicitos = cfg["sentinel2"].get("archivos", [])
    if explicitos:
        archivos = [resolver(s, Path(cfg["_raiz"])) for s in explicitos]
        if len(archivos) != len(set(p.resolve() for p in archivos)):
            raise ValueError(
                "La lista sentinel2.archivos repite un archivo. No se duplican fechas para simular una serie."
            )
        return archivos, False
    raiz = Path(cfg["carpeta_datos"]) / "sentinel2"
    metadatos = raiz / "metadatos.json"
    carpetas = []
    if metadatos.exists():
        info = json.loads(metadatos.read_text(encoding="utf-8"))
        # El path de Colab ya no existe en Windows; reutilizar solo el nombre de su carpeta.
        nombre = (
            str(info.get("carpeta_recuadros", "")).replace("\\", "/").rsplit("/", 1)[-1]
        )
        if nombre and (raiz / nombre).is_dir():
            carpetas = [raiz / nombre]
    if not carpetas:
        carpetas = sorted(p for p in raiz.glob("escenas_completas_*") if p.is_dir())
    if len(carpetas) > 1:
        raise ValueError(
            "Hay varias descargas Sentinel-2 y no se identifica una activa. Indica un GeoTIFF en sentinel2.archivos."
        )
    carpeta = carpetas[0] if carpetas else raiz
    archivos = sorted(carpeta.glob("recuadro_*.tif"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay recuadro_*.tif en {carpeta}. Copia los resultados del notebook o configura las rutas."
        )
    tabla = raiz / "cortes_georreferenciados.csv"
    if tabla.exists():
        df = pd.read_csv(tabla)
        if {"archivo", "fraccion_util_rectangulo"}.issubset(df.columns):
            puntuacion = {
                str(row.archivo).replace("\\", "/").rsplit("/", 1)[-1]: float(
                    row.fraccion_util_rectangulo
                )
                for row in df.itertuples()
                if pd.notna(row.archivo)
            }
            archivos.sort(key=lambda p: (-puntuacion.get(p.name, -1), p.name))
    limite = int(cfg["sentinel2"]["max_recuadros_a_buscar"])
    if limite < 1:
        raise ValueError("max_recuadros_a_buscar debe ser positivo.")
    return archivos[:limite], True


def sumas_ventanas(
    mascara: np.ndarray, filas: np.ndarray, columnas: np.ndarray, lado: int
) -> np.ndarray:
    integral = np.pad(mascara.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r, c = np.ix_(filas, columnas)
    return (
        integral[r + lado, c + lado]
        - integral[r, c + lado]
        - integral[r + lado, c]
        + integral[r, c]
    )


def elegir_ventana(
    lectores, indices, lado: int, clases: list, umbral: float
) -> tuple[Window, float]:
    h, w = lectores[0].height, lectores[0].width
    if min(h, w) < lado:
        raise ValueError(f"El recuadro {w}×{h} es menor que el parche {lado}×{lado}.")
    paso = max(1, lado // 2)
    filas = np.unique(np.r_[np.arange(0, h - lado + 1, paso), h - lado])
    columnas = np.unique(np.r_[np.arange(0, w - lado + 1, paso), w - lado])
    disponibles = np.ones((len(filas), len(columnas)), dtype=bool)
    calidad = np.ones(disponibles.shape, dtype=float)
    for src, idx in zip(lectores, indices):
        scl, datamask = src.read([idx["SCL"], idx["dataMask"]])
        datos = (datamask > 0) & np.isfinite(scl)
        clara = datos & np.isin(scl, clases)
        disponibles &= sumas_ventanas(datos, filas, columnas, lado) == lado * lado
        calidad = np.minimum(
            calidad, sumas_ventanas(clara, filas, columnas, lado) / (lado * lado)
        )
    puntuacion = np.where(disponibles, calidad, -1)
    pos = np.unravel_index(int(np.argmax(puntuacion)), puntuacion.shape)
    mejor = float(puntuacion[pos])
    if mejor < umbral:
        raise ValueError(
            f"No hay parche sin nodata y con {umbral:.0%} de clases claras en todas las fechas. Mejor fracción: {max(mejor, 0):.1%}."
        )
    return Window(int(columnas[pos[1]]), int(filas[pos[0]]), lado, lado), mejor


def preparar_grupo_sentinel(cfg: dict, archivos: list[Path]) -> dict:
    parametros = cfg["sentinel2"]
    lado = int(parametros["tamanio_parche_px"])
    if not 16 <= lado <= 256:
        raise ValueError(
            "Para la prueba tamanio_parche_px debe estar entre 16 y 256; recomendado 64."
        )
    umbral = float(parametros["fraccion_clara_minima"])
    if not 0 <= umbral <= 1:
        raise ValueError("fraccion_clara_minima debe estar entre 0 y 1.")
    fichas = []
    for archivo in archivos:
        if not archivo.is_file():
            raise FileNotFoundError(archivo)
        with rasterio.open(archivo) as src:
            mapa_bandas(src)
            try:
                fecha = pd.Timestamp(src.tags()["fecha_utc"]).normalize()
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"{archivo}: falta una fecha_utc válida en las etiquetas del GeoTIFF."
                ) from error
            if pd.isna(fecha):
                raise ValueError(f"Fecha inválida en {archivo}.")
            fichas.append((fecha, archivo))
    fichas.sort(key=lambda f: f[0])
    fechas = pd.DatetimeIndex([f[0] for f in fichas])
    if fechas.has_duplicates:
        raise ValueError(
            "Hay varios recuadros de una misma fecha. Una serie requiere el MISMO terreno en fechas diferentes."
        )
    # Los archivos explícitos son observaciones de una misma zona en distintas fechas.
    with ExitStack() as stack:
        originales = [stack.enter_context(rasterio.open(p)) for _, p in fichas]
        ref = originales[0]
        if (
            ref.crs is None
            or not ref.crs.is_projected
            or not np.allclose(ref.res, (10, 10))
        ):
            raise ValueError(
                "La referencia debe ser un GeoTIFF proyectado en metros con píxel de 10 m, como el notebook."
            )
        if (
            ref.crs.linear_units not in ("metre", "meter")
            or abs(ref.transform.b) > 1e-9
            or abs(ref.transform.d) > 1e-9
        ):
            raise ValueError(
                "Se necesita la malla métrica de 10 m sin rotación del notebook."
            )
        lectores, alineados = [], []
        indices = [mapa_bandas(s) for s in originales]
        for src in originales:
            misma = (
                src.crs == ref.crs
                and src.transform.almost_equals(ref.transform)
                and src.shape == ref.shape
            )
            lectores.append(
                src
                if misma
                else stack.enter_context(
                    WarpedVRT(
                        src,
                        crs=ref.crs,
                        transform=ref.transform,
                        width=ref.width,
                        height=ref.height,
                        nodata=0,
                        resampling=rasterio.enums.Resampling.nearest,
                    )
                )
            )
            alineados.append(not misma)
        ventana, fraccion = elegir_ventana(
            lectores, indices, lado, parametros["clases_scl_claras"], umbral
        )
        datos = np.stack(
            [
                src.read([idx[b] for b in BANDAS], window=ventana).astype(np.float32)
                for src, idx in zip(lectores, indices)
            ]
        )[None, ...]
        if not np.isfinite(datos).all():
            raise ValueError(
                "El parche tiene NaN o infinitos. No se sustituyen por ceros."
            )
        if float(np.median(np.abs(datos))) > 2 or float(np.max(np.abs(datos))) > 100:
            raise ValueError(
                "Las bandas parecen valores DN/enteros o unidades incompatibles. Se espera reflectancia, ya convertida por el notebook."
            )
        tr = ref.window_transform(ventana)
        crs = ref.crs.to_string()
    dias = int(parametros["dias_hasta_consulta"])
    if not 1 <= dias <= 30:
        raise ValueError(
            "dias_hasta_consulta debe estar entre 1 y 30 para esta prueba."
        )
    origen = fechas[0]
    consulta = fechas[-1] + timedelta(days=dias)
    doy = np.asarray([(f - origen).days for f in fechas], dtype=np.float32)[None, :]
    query = np.array([[(consulta - origen).days]], dtype=np.float32)
    carpeta = Path(cfg["carpeta_salida"]) / "preparados"
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        carpeta / "sentinel2_entrada.npz",
        reflectancia=datos,
        dias_desde_origen=doy,
        dias_consulta=query,
        bandas=np.array(BANDAS),
    )
    info = {
        "huella_config": huella_config(cfg, "sentinel2"),
        "forma": list(datos.shape),
        "dtype": str(datos.dtype),
        "bandas": BANDAS,
        "unidad": "reflectancia adimensional",
        "resolucion_m": 10,
        "archivos": [str(f[1].resolve()) for f in fichas],
        "fechas_utc": fechas.strftime("%Y-%m-%d").tolist(),
        "origen_temporal": origen.strftime("%Y-%m-%d"),
        "consulta_utc": consulta.strftime("%Y-%m-%d"),
        "dias_consulta": int(query[0, 0]),
        "crs": crs,
        "transform": list(tr)[:6],
        "ventana": {
            "fila": int(ventana.row_off),
            "columna": int(ventana.col_off),
            "lado_px": lado,
        },
        "alineacion_aplicada": alineados,
        "fraccion_clara_minima_del_parche": fraccion,
        "tratamiento": "Se elige un parche sin nodata; nubes conservadas en las bandas. SCL solo selecciona la ventana.",
        "alcance": "Ingesta mínima con una fecha"
        if len(fechas) == 1
        else "Ingesta de secuencia multifecha; sin evaluar precisión",
        "origen_datos": "archivos locales del usuario; no se descargaron datos nuevos",
    }
    escribir_json(carpeta / "sentinel2_metadatos.json", info)
    logging.info(
        "Sentinel-2: %s, %d fecha(s), reflectancia sin dividir otra vez entre 10000.",
        datos.shape,
        len(fechas),
    )
    return info


def preparar_sentinel(cfg: dict) -> dict:
    archivos, automatico = archivos_sentinel(cfg)
    if not automatico:
        return preparar_grupo_sentinel(cfg, archivos)
    errores = []
    for archivo in archivos:
        try:
            return preparar_grupo_sentinel(cfg, [archivo])
        except ValueError as error:
            errores.append(f"{archivo.name}: {error}")
    raise ValueError(
        "No se encontró un parche válido entre los candidatos. Cambia archivos o el umbral SCL; no se inventan datos. "
        + " | ".join(errores[:3])
    )


def validar_calendario(fechas) -> pd.DatetimeIndex:
    fechas = pd.DatetimeIndex(pd.to_datetime(fechas, utc=True)).tz_convert(None)
    if len(fechas) < 3:
        raise ValueError("La prueba de Chronos-2 requiere al menos tres días reales.")
    if fechas.has_duplicates or not fechas.equals(fechas.normalize()):
        raise ValueError(
            "Se necesitan días UTC sin duplicados, etiquetados a medianoche."
        )
    if not fechas.equals(pd.date_range(fechas[0], fechas[-1], freq="D")):
        raise ValueError(
            "Faltan días o las fechas no están ordenadas. No se rellenan faltantes con lluvia cero."
        )
    return fechas


def preparar_era5(cfg: dict) -> dict:
    parametros = cfg["era5_land"]
    modo = parametros["modo_series"]
    if modo not in ("regional", "por_celda"):
        raise ValueError("modo_series debe ser regional o por_celda.")
    base = Path(cfg["carpeta_datos"])
    archivo = resolver(parametros["archivo_netcdf"], base)
    descartadas = 0
    if archivo.is_file():
        with xr.open_dataset(archivo) as ds:
            if "precipitacion_diaria_mm" not in ds:
                raise ValueError(
                    "Este adaptador recibe precipitacion_diaria_mm.nc YA interpretado por el notebook; no el tp horario crudo."
                )
            lluvia = ds["precipitacion_diaria_mm"].load()
        if set(lluvia.dims) != {"time", "latitude", "longitude"}:
            raise ValueError(f"Dimensiones de lluvia no esperadas: {lluvia.dims}.")
        if str(lluvia.attrs.get("units", "")).strip().lower() != "mm":
            raise ValueError(
                "La precipitación preparada debe estar en mm. No se aplica otra multiplicación por 1000."
            )
        lluvia = lluvia.transpose("time", "latitude", "longitude").sortby("time")
        fechas = validar_calendario(lluvia.time.values)
        valores = lluvia.values.astype(np.float64)
        if np.isinf(valores).any():
            raise ValueError("Hay infinitos en los datos de lluvia.")
        if np.any(valores[np.isfinite(valores)] < -1e-6):
            raise ValueError("La lluvia observada contiene valores negativos.")
        valores = np.maximum(valores, 0)
        comunes = np.isfinite(valores).all(axis=0)
        con_alguno = np.isfinite(valores).any(axis=0)
        descartadas = int((con_alguno & ~comunes).sum())
        if not comunes.any():
            raise ValueError(
                "No hay celdas con todos los días presentes; no se imputan valores en esta prueba."
            )
        series = valores[:, comunes].T
        lon, lat = np.meshgrid(lluvia.longitude.values, lluvia.latitude.values)
        if modo == "regional":
            pesos = np.cos(np.deg2rad(lat[comunes]))
            series = np.average(series, axis=0, weights=pesos)[None, :]
            ids = ["zona_revision"]
        else:
            ids = [
                f"lat_{a:.4f}_lon_{b:.4f}" for a, b in zip(lat[comunes], lon[comunes])
            ]
        fuente = "netcdf_diario_mm"
    else:
        archivo = resolver(parametros["archivo_csv_alternativo"], base)
        if not archivo.is_file():
            raise FileNotFoundError(
                "Falta el NetCDF diario y también resumen_diario_aoi.csv. Copia la carpeta era5_land generada por el notebook."
            )
        if modo != "regional":
            raise ValueError(
                "El CSV de resumen solo permite modo regional; para series por celda se necesita el NetCDF."
            )
        df = pd.read_csv(archivo)
        columnas = {"fecha_utc", "precipitacion_media_aoi_mm"}
        if not columnas.issubset(df.columns):
            raise ValueError(f"El CSV necesita estas columnas: {columnas}.")
        df = df.sort_values("fecha_utc")
        fechas = validar_calendario(df.fecha_utc)
        series = pd.to_numeric(df.precipitacion_media_aoi_mm, errors="raise").to_numpy(
            dtype=np.float64
        )[None, :]
        ids = ["zona_revision"]
        fuente = "csv_diario_mm"
    if not np.isfinite(series).all() or np.any(series < 0):
        raise ValueError(
            "Las series deben contener lluvia finita y no negativa. No se imputan faltantes como cero."
        )
    pasos = int(parametros["pasos_a_predecir"])
    if not 1 <= pasos <= 7:
        raise ValueError("pasos_a_predecir debe estar entre 1 y 7 días para la prueba.")
    registros = [
        pd.DataFrame(
            {"item_id": nombre, "timestamp": fechas, "target": datos.astype(np.float32)}
        )
        for nombre, datos in zip(ids, series)
    ]
    tabla = pd.concat(registros, ignore_index=True).sort_values(
        ["item_id", "timestamp"]
    )
    carpeta = Path(cfg["carpeta_salida"]) / "preparados"
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "chronos2_entrada.csv", index=False)
    contexto = series[:, None, :].astype(np.float32)
    np.savez_compressed(
        carpeta / "chronos2_entrada.npz", contexto=contexto, ids=np.asarray(ids)
    )
    info = {
        "huella_config": huella_config(cfg, "era5_land"),
        "archivo_fuente": str(archivo.resolve()),
        "fuente": fuente,
        "forma_contexto": list(contexto.shape),
        "numero_series": len(ids),
        "dias_historicos": len(fechas),
        "fecha_inicial_utc": fechas[0].strftime("%Y-%m-%d"),
        "fecha_final_utc": fechas[-1].strftime("%Y-%m-%d"),
        "frecuencia": "D",
        "unidad": "mm/día",
        "modo_series": modo,
        "celdas_incompletas_descartadas": descartadas,
        "pasos_a_predecir": pasos,
        "reescala": "ninguna; Chronos-2 maneja su propia normalización",
        "tratamiento_fechas": "Se preservan las fechas diarias ya corregidas por el notebook; no se resta otro día.",
        "alcance": "Prueba de recepción con datos diarios; la agregación mensual requiere historia y meses completos.",
    }
    escribir_json(carpeta / "chronos2_metadatos.json", info)
    logging.info(
        "ERA5-Land: contexto %s, %s serie(s) de %s días reales.",
        contexto.shape,
        len(ids),
        len(fechas),
    )
    return info
