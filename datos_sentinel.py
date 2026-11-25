"""Sentinel-2 L2A desde Earth Search y COG públicos, sin OAuth ni Process API."""

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from requests.adapters import HTTPAdapter
from shapely.geometry import box, shape
from urllib3.util.retry import Retry
from utilidades import BANDAS, guardar_json, huella, leer_json

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_COLECCION = "sentinel-2-c1-l2a"
EARTH_SEARCH_COLECCIONES = (EARTH_SEARCH_COLECCION, "sentinel-2-l2a")
PROCEDENCIA = "Earth Search / Element 84, Sentinel-2 L2A, COG públicos AWS"
RADIOMETRIA = "max(DN * scale + offset, 0); nodata antes de convertir; NEAREST"
ALL_DOWNLOAD_BANDS = BANDAS + ["SCL", "dataMask"]
ASSETS = dict(
    zip(
        ALL_DOWNLOAD_BANDS[:-1],
        [
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir08",
            "swir16",
            "swir22",
            "scl",
        ],
    )
)
_sesion = None


def _cliente():
    global _sesion
    if _sesion is None:
        _sesion = requests.Session()
        reintentos = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        _sesion.mount("https://", HTTPAdapter(max_retries=reintentos))
        _sesion.headers["User-Agent"] = "aguacate-sentinel-earth-search/1"
    return _sesion


def fecha_escena(item):
    return dt.datetime.fromisoformat(
        item["properties"]["datetime"].replace("Z", "+00:00")
    )


def acquisition_interval(timestamp, margin_min=10):
    timestamp = timestamp.astimezone(dt.timezone.utc)
    margen = dt.timedelta(minutes=margin_min)
    inicio = max(
        timestamp - margen, timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    )
    fin = min(
        timestamp + margen,
        timestamp.replace(hour=23, minute=59, second=59, microsecond=999999),
    )
    return tuple(t.isoformat().replace("+00:00", "Z") for t in [inicio, fin])


def _intervalo_busqueda(intervalo):
    limites = []
    for indice, valor in enumerate(intervalo):
        valor = str(valor)
        if len(valor) == 10:
            valor += "T00:00:00Z" if indice == 0 else "T23:59:59.999999Z"
        fecha = dt.datetime.fromisoformat(valor.replace("Z", "+00:00"))
        if fecha.tzinfo is None:
            raise ValueError("Las fechas de consulta deben indicar UTC.")
        limites.append(fecha.astimezone(dt.timezone.utc))
    if len(limites) != 2 or limites[0] > limites[1]:
        raise ValueError("Intervalo de búsqueda Sentinel no válido.")
    return limites


def buscar_escenas(bbox_wgs84, intervalo, *, max_cloud=None):
    """Consulta todos los metadatos paginados; no descarga bandas ni usa claves."""
    t0, t1 = _intervalo_busqueda(intervalo)
    if len(bbox_wgs84) != 4 or not np.isfinite(bbox_wgs84).all():
        raise ValueError("BBox WGS84 no válido.")
    parametros = {
        "collections": ",".join(EARTH_SEARCH_COLECCIONES),
        "bbox": ",".join(str(float(v)) for v in bbox_wgs84),
        "datetime": "/".join(t.isoformat().replace("+00:00", "Z") for t in [t0, t1]),
        "limit": 100,
    }
    if max_cloud is not None:
        if not 0 <= max_cloud <= 1:
            raise ValueError("max_cloud debe estar entre 0 y 1.")
        parametros["query"] = json.dumps({"eo:cloud_cover": {"lt": max_cloud * 100}})
    url = EARTH_SEARCH_URL + "/search"
    items, vistos, paginas = [], set(), set()
    while url:
        respuesta = _cliente().get(url, params=parametros, timeout=(10, 60))
        respuesta.raise_for_status()
        if respuesta.url in paginas:
            raise ValueError("Earth Search repitió una página; catálogo incompleto.")
        paginas.add(respuesta.url)
        documento = respuesta.json()
        if not isinstance(documento.get("features"), list):
            raise ValueError("Earth Search no devolvió una colección STAC válida.")
        for item in documento["features"]:
            if not t0 <= fecha_escena(item) <= t1:
                raise ValueError("El catálogo devolvió una escena fuera del intervalo.")
            identidad = (item.get("collection", ""), item["id"])
            if identidad not in vistos:
                nube = float(item["properties"].get("eo:cloud_cover", 100))
                if max_cloud is None or nube < max_cloud * 100:
                    items.append(item)
                vistos.add(identidad)
        siguientes = [
            enlace
            for enlace in documento.get("links", [])
            if enlace.get("rel") == "next"
        ]
        if siguientes and siguientes[0].get("method", "GET") != "GET":
            raise ValueError(
                "Paginación STAC no compatible; no se usará un catálogo parcial."
            )
        url = siguientes[0]["href"] if siguientes else None
        if url and not url.startswith(EARTH_SEARCH_URL + "/"):
            raise ValueError(
                "Earth Search devolvió una página fuera del catálogo configurado."
            )
        parametros = None
    # Las colecciones pueden publicar dos procesamientos de la misma captura.
    # Preferir Collection 1; completar con L2A las capturas que allí faltan.
    capturas = {}
    for item in items:
        clave = _identidad_captura(item)
        if clave not in capturas or _prioridad_item(item) < _prioridad_item(
            capturas[clave]
        ):
            capturas[clave] = item
    return sorted(capturas.values(), key=lambda it: (fecha_escena(it), it["id"]))


def _prioridad_item(item):
    coleccion = item.get("collection", "")
    rango = (
        EARTH_SEARCH_COLECCIONES.index(coleccion)
        if coleccion in EARTH_SEARCH_COLECCIONES
        else 2
    )
    baseline = re.findall(
        r"\d+", str(item.get("properties", {}).get("s2:processing_baseline", "0.0"))
    )
    return (rango, tuple(-int(v) for v in baseline), item["id"])


def _identidad_captura(item):
    propiedades = item.get("properties", {})
    producto = str(propiedades.get("s2:product_uri", ""))
    try:
        plataforma, tesela = _clave_escena(item)
    except ValueError:
        return (item.get("collection", ""), item["id"])
    captura = re.search(r"_MSIL2A_(\d{8}T\d{6})_", producto)
    tiempo = captura.group(1) if captura else fecha_escena(item).isoformat()
    return plataforma, tesela, tiempo


def _clave_escena(item):
    """Relaciona los identificadores de los catálogos anterior y actual."""
    texto = str(item["id"])
    producto = item.get("properties", {}).get("s2:product_uri", texto)
    plataforma = re.search(r"S2[ABC]", producto)
    tesela = re.search(r"_T?(\d{2}[A-Z]{3})(?:_|\.)", producto)
    if not plataforma or not tesela:
        raise ValueError(f"No se identifica la plataforma/tesela de {texto}.")
    return plataforma.group(), tesela.group(1)


def _es_cog_earth(item):
    assets = item.get("assets", {})
    return all(k in assets and "href" in assets[k] for k in ASSETS.values())


def resolver_escenas_earth(
    items, bbox_wgs84, timestamp, directorio_cache, *, margin_min=10, max_cloud=None
):
    """Resuelve catálogos antiguos sin cambiar de adquisición ni borrar su caché."""
    intervalo = acquisition_interval(timestamp, margin_min)
    t0, t1 = _intervalo_busqueda(intervalo)
    grupo = [i for i in items if t0 <= fecha_escena(i) <= t1]
    if not grupo:
        raise ValueError("No hay escenas de la adquisición solicitada.")
    if all(_es_cog_earth(i) for i in grupo):
        return grupo
    claves = {_clave_escena(i) for i in grupo}
    consulta = {
        "bbox": list(bbox_wgs84),
        "intervalo": intervalo,
        "claves": sorted(claves),
        "max_cloud": max_cloud,
        "colecciones": list(EARTH_SEARCH_COLECCIONES),
    }
    ruta = Path(directorio_cache) / f"earth_search_{huella(consulta)}.json"
    if ruta.exists():
        encontrados = leer_json(ruta)
    else:
        encontrados = buscar_escenas(bbox_wgs84, intervalo, max_cloud=max_cloud)
        guardar_json(ruta, encontrados)
    encontrados = [
        i
        for i in encontrados
        if t0 <= fecha_escena(i) <= t1 and _clave_escena(i) in claves
    ]
    if {_clave_escena(i) for i in encontrados} != claves or not all(
        _es_cog_earth(i) for i in encontrados
    ):
        raise ValueError(
            "Earth Search no contiene todas las teselas de esa adquisición. "
            "Se conserva lo descargado; no se sustituye la fecha ni se usa Sentinel Hub."
        )
    return encontrados


def _leer_banda(asset, crs, transformacion, ancho, alto, *, reflectancia):
    bandas = asset.get("raster:bands", [])
    if not bandas or "nodata" not in bandas[0]:
        raise ValueError("Faltan los metadatos de nodata del COG.")
    metadata = bandas[0]
    if reflectancia and not {"scale", "offset"} <= set(metadata):
        raise ValueError(
            "Faltan scale/offset; no se supondrá una calibración de reflectancia."
        )
    nodata = metadata["nodata"]
    with rasterio.open(asset["href"]) as src:
        with WarpedVRT(
            src,
            crs=crs,
            transform=transformacion,
            width=ancho,
            height=alto,
            resampling=Resampling.nearest,
            src_nodata=nodata,
            nodata=nodata,
            dtype="float32",
            warp_mem_limit=64,
        ) as vrt:
            arr = vrt.read(1, masked=True)
    valido = ~np.ma.getmaskarray(arr) & np.isfinite(arr.data)
    valor = np.asarray(arr.filled(0), dtype=np.float32)
    if reflectancia:
        escala, offset = float(metadata["scale"]), float(metadata["offset"])
        if not np.isfinite([escala, offset]).all() or escala <= 0:
            raise ValueError("Calibración de reflectancia no válida.")
        valor = np.maximum(valor * escala + offset, 0)
    elif not np.isin(valor[valido], np.arange(12)).all():
        raise ValueError("El COG contiene clases SCL inesperadas.")
    valor[~valido] = 0
    return valor, valido


def leer_escenas(items, bbox, crs, size):
    """Mosaico de una adquisición, doce capas, float32 y reflectancia armonizada.

    Los bloques COG se leen mediante HTTP Range. Todas las bandas y SCL se
    alinean a la cuadrícula indicada; dataMask se deriva de sus máscaras.
    """
    ancho, alto = size
    if any(not isinstance(v, (int, np.integer)) or not 1 <= v <= 2500 for v in size):
        raise ValueError("La ventana debe tener entre 1 y 2500 píxeles por lado.")
    if not items or not all(_es_cog_earth(i) for i in items):
        raise ValueError("Se requieren escenas Earth Search con sus once COG.")
    fechas = [fecha_escena(i) for i in items]
    if (
        len({t.date() for t in fechas}) != 1
        or (max(fechas) - min(fechas)).total_seconds() > 1200
    ):
        raise ValueError("No se pueden mezclar distintas adquisiciones en una lectura.")
    tr = from_bounds(*bbox, ancho, alto)
    wgs = box(*transform_bounds(crs, "EPSG:4326", *bbox, densify_pts=21))
    salida = np.zeros((alto, ancho, 12), dtype=np.float32)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY=3,
        GDAL_HTTP_RETRY_DELAY=1,
        GDAL_CACHEMAX=64 * 1024 * 1024,
    ):
        for item in sorted(
            items,
            key=lambda it: (
                float(it["properties"].get("eo:cloud_cover", 100)),
                it["id"],
            ),
        ):
            if not shape(item["geometry"]).intersects(wgs):
                continue
            if (salida[..., 11] > 0).all():
                break
            valores = np.zeros_like(salida)
            mascara = np.ones((alto, ancho), dtype=bool)
            for i, b in enumerate(ALL_DOWNLOAD_BANDS[:-1]):
                valor, valido = _leer_banda(
                    item["assets"][ASSETS[b]],
                    crs,
                    tr,
                    ancho,
                    alto,
                    reflectancia=b != "SCL",
                )
                valores[..., i] = valor
                mascara &= valido
            tomar = mascara & (salida[..., 11] == 0)
            salida[tomar, :11] = valores[tomar, :11]
            salida[..., 11][tomar] = 1
    if not np.isfinite(salida).all():
        raise ValueError("La lectura Sentinel produjo valores no finitos.")
    return salida


def make_evalscript():
    """Descriptor para reconocer TIFF antiguos; nunca se ejecuta ni se envía."""
    unidades = ["REFLECTANCE"] * 10 + ["DN", "DN"]
    return (
        "//VERSION=3\nfunction setup(){return {input:[{bands:"
        + json.dumps(ALL_DOWNLOAD_BANDS)
        + ",units:"
        + json.dumps(unidades)
        + '}],output:{bands:12,sampleType:"FLOAT32"},mosaicking:"SIMPLE"};}\n'
        + "function evaluatePixel(s){return ["
        + ",".join("s." + b for b in ALL_DOWNLOAD_BANDS)
        + "];}"
    )
