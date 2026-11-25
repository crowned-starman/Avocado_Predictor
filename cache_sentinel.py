"""Reutilización verificable de GeoTIFF Sentinel, incluidas las cachés anteriores."""

import json
import re
from datetime import datetime
from pathlib import Path

import rasterio
from datos_sentinel import ALL_DOWNLOAD_BANDS, make_evalscript
from rasterio.windows import Window
from rasterio.windows import transform as transformar_ventana
from shapely.geometry import shape
from utilidades import guardar_json, huella

PREFIJO = "escenas_completas_"
CLAVES = (
    "version_cobertura",
    "bbox",
    "crs",
    "size",
    "timestamp",
    "interval",
    "bands",
    "evalscript",
    "max_cloud",
    "resampling",
    "harmonizeValues",
    "ids_productos_catalogo",
    "huellas",
    "max_px",
)


def _compactar(script):
    # Solo se admite la expresión conocida; no se equiparan algoritmos distintos.
    return re.sub(r"\s+", "", script)


def descriptor(peticion):
    """Firma estable ante espacios del script y orden de anillos de la geometría."""
    base = {k: peticion[k] for k in CLAVES}
    if _compactar(base["evalscript"]) != _compactar(make_evalscript()):
        raise ValueError(
            "El evalscript de la caché no corresponde a las doce capas actuales."
        )
    base["evalscript"] = _compactar(base["evalscript"])
    base["huellas"] = shape(base["huellas"]).normalize().wkb_hex
    base["bbox"] = [float(v) for v in base["bbox"]]
    base["max_cloud"] = float(base["max_cloud"])
    base["ids_productos_catalogo"] = sorted(base["ids_productos_catalogo"])
    # Convierte tuplas del notebook y listas de JSON a la misma representación.
    return json.loads(json.dumps(base))


def _script_anterior():
    # Texto original exacto: permite reconocer también carpetas sin metadatos.
    unidades = ["REFLECTANCE"] * 10 + ["DN", "DN"]
    return (
        "//VERSION=3\nfunction setup() { return {input:[{bands:"
        + json.dumps(ALL_DOWNLOAD_BANDS)
        + ",units:"
        + json.dumps(unidades)
        + '}],output:{bands:12,sampleType:"FLOAT32"},mosaicking:"SIMPLE"}; }\n'
        + "function evaluatePixel(s) { return ["
        + ",".join("s." + b for b in ALL_DOWNLOAD_BANDS)
        + "]; }"
    )


def firma_recuadro(firma_carpeta, p):
    return huella(
        {
            "peticion": firma_carpeta,
            "ventana": [
                p[k] for k in ["fila_inicio", "columna_inicio", "alto_px", "ancho_px"]
            ],
        }
    )


def _leer(ruta):
    try:
        valor = json.loads(ruta.read_text(encoding="utf-8"))
        return valor if isinstance(valor, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _validar(ruta, p, firma, peticion, transformacion):
    """Inspecciona cabecera y procedencia; no carga el mosaico completo a memoria."""
    try:
        with rasterio.open(ruta) as src:
            r, c, h, w = (
                p[k] for k in ["fila_inicio", "columna_inicio", "alto_px", "ancho_px"]
            )
            esperado = transformar_ventana(Window(c, r, w, h), transformacion)
            if (src.height, src.width, src.count) != (h, w, 12):
                return "Dimensiones o número de capas distintos."
            if list(src.descriptions) != ALL_DOWNLOAD_BANDS:
                return "Nombres u orden de bandas distintos."
            if any(t != "float32" for t in src.dtypes):
                return "Tipo de datos distinto de float32."
            if src.crs != rasterio.crs.CRS.from_string(peticion["crs"]):
                return "CRS distinto."
            if not src.transform.almost_equals(esperado):
                return "Cuadrícula o resolución distintas."
            if src.tags().get("huella_peticion") != firma_recuadro(firma, p):
                return "Firma de petición distinta o ausente."
            fecha = datetime.fromisoformat(peticion["timestamp"]).date().isoformat()
            if src.tags().get("fecha_utc") != fecha:
                return "Fecha distinta o ausente."
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return f"GeoTIFF no legible ({type(error).__name__})."
    return None


def resolver_cache(
    directorio, peticion, plan, transformacion, *, permitir_descargas=False
):
    """Devuelve rutas existentes verificadas; por defecto nunca habilita descargas."""
    directorio = Path(directorio)
    contenido = descriptor(peticion)
    firma_semantica = huella(contenido)
    conocidas = {huella(peticion)}
    conocidas.add(huella({**peticion, "evalscript": _script_anterior()}))
    metadatos = _leer(directorio / "metadatos.json")
    anterior = None
    try:
        if descriptor(metadatos) == contenido:
            # basename funciona también si se movió desde Colab u otro Windows.
            anterior = Path(str(metadatos["carpeta_recuadros"]).replace("\\", "/")).name
            base_anterior = {k: metadatos[k] for k in CLAVES}
            if anterior != PREFIJO + huella(base_anterior):
                anterior = None
    except (KeyError, ValueError, TypeError):
        pass
    esperados = [p for p in plan if p["intersecta_escenas"]]
    candidatos, rechazados = [], []
    for carpeta in sorted(directorio.glob(PREFIJO + "*")):
        if not carpeta.is_dir():
            continue
        firma = carpeta.name[len(PREFIJO) :]
        compatible = firma in conocidas or carpeta.name == anterior
        manifiesto = _leer(carpeta / "peticion_cache.json")
        try:
            compatible |= (
                manifiesto.get("firma_archivos") == firma
                and descriptor(manifiesto["peticion"]) == contenido
            )
        except (KeyError, ValueError, TypeError):
            pass
        if not compatible:
            rechazados.append(
                {
                    "archivo": str(carpeta),
                    "motivo": "Petición no verificable o distinta.",
                }
            )
            continue
        archivos = {}
        for p in esperados:
            archivo = carpeta / f"recuadro_{p['corte']:04d}.tif"
            if not archivo.exists():
                continue
            motivo = _validar(archivo, p, firma, peticion, transformacion)
            if motivo:
                rechazados.append({"archivo": str(archivo), "motivo": motivo})
            else:
                archivos[p["corte"]] = archivo
        candidatos.append((carpeta, firma, archivos))
    candidatos.sort(
        key=lambda item: (-len(item[2]), item[0].name != anterior, item[0].name)
    )
    archivos = {}
    for _, _, disponibles in candidatos:
        for corte, ruta in disponibles.items():
            archivos.setdefault(corte, ruta)
    faltantes = [p["corte"] for p in esperados if p["corte"] not in archivos]
    if candidatos:
        carpeta, firma, _ = candidatos[0]
    else:
        firma = firma_semantica
        carpeta = directorio / (PREFIJO + firma)
    informe = {
        "firma_semantica": firma_semantica,
        "esperados": len(esperados),
        "reutilizables": len(archivos),
        "faltantes": faltantes,
        "descargas_habilitadas": permitir_descargas,
        "carpeta_activa": str(carpeta),
        "archivos_reutilizados": {str(k): str(v) for k, v in archivos.items()},
        "rechazados": rechazados,
    }
    ruta_informe = directorio / "diagnostico_cache.json"
    guardar_json(ruta_informe, informe)
    print(
        f"Caché Sentinel: {len(archivos)}/{len(esperados)} TIFF reutilizables; {len(faltantes)} faltantes."
    )
    print("Diagnóstico:", ruta_informe)
    if faltantes and not permitir_descargas:
        raise ValueError(
            f"Faltan {len(faltantes)} TIFF verificables. No se descargó ningún TIFF. "
            f"Revisa {ruta_informe}. Para solicitar únicamente los faltantes utiliza "
            "revisar-datos --fuente sentinel --descargar-faltantes-sentinel."
        )
    # Nunca reemplaza un archivo existente que no pasó la validación.
    for corte in faltantes:
        destino = carpeta / f"recuadro_{corte:04d}.tif"
        if destino.exists():
            raise ValueError(
                f"Archivo incompatible conservado: {destino}. Revisa el diagnóstico antes de descargar."
            )
    carpeta.mkdir(parents=True, exist_ok=True)
    # Conserva la procedencia de cada carpeta usada antes de regenerar metadatos.json.
    registrar = {c[0]: c[1] for c in candidatos if c[2]}
    registrar[carpeta] = firma
    for destino, firma_archivos in registrar.items():
        guardar_json(
            destino / "peticion_cache.json",
            {
                "version": 1,
                "firma_semantica": firma_semantica,
                "firma_archivos": firma_archivos,
                "peticion": peticion,
            },
        )
    return {
        "carpeta": carpeta,
        "firma": firma,
        "archivos": archivos,
        "informe": informe,
    }
