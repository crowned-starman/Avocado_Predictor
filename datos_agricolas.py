"""Lectura y validación conservadas del notebook de revisión del proyecto."""

import io
import re
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

PORTAL_AGRICOLA = "https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php"


def normalizar_texto(valor):
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", texto.casefold())


CAMPOS_AGRI = [
    "Anio",
    "Idestado",
    "Nomestado",
    "Idddr",
    "Nomddr",
    "Idcader",
    "Nomcader",
    "Idmunicipio",
    "Nommunicipio",
    "Idciclo",
    "Nomcicloproductivo",
    "Idmodalidad",
    "Nommodalidad",
    "Idunidadmedida",
    "Nomunidad",
    "Idcultivo",
    "Nomcultivo",
    "Sembrada",
    "Cosechada",
    "Siniestrada",
    "Volumenproduccion",
    "Rendimiento",
    "Preciomediorural",
    "Valorproduccion",
]
IDS_AGRI = [
    "Anio",
    "Idestado",
    "Idddr",
    "Idcader",
    "Idmunicipio",
    "Idciclo",
    "Idmodalidad",
    "Idunidadmedida",
    "Idcultivo",
]
NUMS_AGRI = [
    "Sembrada",
    "Cosechada",
    "Siniestrada",
    "Volumenproduccion",
    "Rendimiento",
    "Preciomediorural",
    "Valorproduccion",
]
CLAVE_DETALLE = [
    "Anio",
    "Idestado",
    "Idddr",
    "Idcader",
    "Idmunicipio",
    "Idciclo",
    "Idmodalidad",
    "Idunidadmedida",
    "Idcultivo",
]


def descubrir_enlaces_agricolas(html):
    soup = BeautifulSoup(html, "html.parser")
    # Se identifica el bloque por su encabezado, no por años o nombres de archivos inventados.
    tarjetas = soup.select("#dataAccordion > .card")
    municipal = None
    for tarjeta in tarjetas:
        texto = tarjeta.get_text(" ", strip=True)
        if texto.startswith("Municipal "):
            municipal = tarjeta
            break
    if municipal is None:
        raise ValueError(
            "Cambió la estructura del portal municipal. Usa ARCHIVOS_AGRICOLAS_LOCALES."
        )
    enlaces = {}
    for a in municipal.select("a[href]"):
        url = urljoin(PORTAL_AGRICOLA, a["href"])
        q = parse_qs(urlparse(url).query)
        if "ANIO" in q:
            anio = int(q["ANIO"][0])
            if anio in enlaces and enlaces[anio] != url:
                raise ValueError("Dos enlaces distintos para el mismo año.")
            enlaces[anio] = url
    diccionario = next(
        (
            urljoin(PORTAL_AGRICOLA, a["href"])
            for a in municipal.select("a[href]")
            if "ANIO" not in parse_qs(urlparse(a["href"]).query)
        ),
        None,
    )
    if not enlaces:
        raise ValueError("No se encontraron enlaces anuales municipales.")
    return enlaces, diccionario


def leer_csv_agricola(ruta):
    contenido = Path(ruta).read_bytes()
    if contenido.lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise ValueError(
            "El servidor entregó HTML en lugar del CSV. No se intentará limpiarlo como datos."
        )
    try:
        texto = contenido.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        texto = contenido.decode("cp1252")
        encoding = "cp1252"
    # Los archivos oficiales verificados usan coma como separador y punto decimal.
    frame = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
    # Nombres históricos observados en los CSV oficiales de 2020–2022.
    alias = {"nomcultivosinum": "Nomcultivo", "precio": "Preciomediorural"}
    frame = frame.rename(
        columns={c: alias.get(normalizar_texto(c), c) for c in frame.columns}
    )
    esperados = {normalizar_texto(c): c for c in CAMPOS_AGRI}
    frame = frame.rename(
        columns={c: esperados.get(normalizar_texto(c), c) for c in frame.columns}
    )
    faltan = set(CAMPOS_AGRI) - set(frame.columns)
    if faltan:
        raise ValueError(
            f"Esquema diferente: faltan {sorted(faltan)}. Revisa el diccionario de ese año."
        )
    for c in frame.columns:
        frame[c] = frame[c].str.strip()
    return frame, encoding


def limpiar_agricola(frame, anio, estados):
    # Conversión estricta de identificadores antes de filtrar el estado.
    idestado = pd.to_numeric(frame["Idestado"], errors="raise")
    seleccion = frame["Nomcultivo"].map(normalizar_texto).eq("aguacate")
    if estados is not None:
        seleccion &= idestado.isin(estados)
    d = frame.loc[seleccion, CAMPOS_AGRI].copy()
    if d.empty:
        raise ValueError(
            f"No hay aguacate para la selección del año {anio}. No se sustituye por cero."
        )
    for c in IDS_AGRI:
        num = pd.to_numeric(d[c], errors="raise")
        if num.isna().any() or not np.isfinite(num).all() or not (num % 1 == 0).all():
            raise ValueError(f"Identificador inválido en {c}.")
        d[c] = num.astype("int64")
    if not d["Anio"].eq(anio).all():
        raise ValueError("El contenido no corresponde al año solicitado.")
    if (
        not d["Idestado"].between(1, 32).all()
        or not d["Idmunicipio"].between(1, 999).all()
    ):
        raise ValueError(
            "Se detectó una clave de total o una clave territorial fuera de rango. Revisar antes de sumar."
        )
    if not d["Idmodalidad"].isin([1, 2]).all():
        raise ValueError(
            "Hay una modalidad distinta a riego/temporal; revisar posibles totales."
        )
    if not d["Nomcicloproductivo"].map(normalizar_texto).eq("perennes").all():
        raise ValueError(
            "Aguacate contiene ciclos inesperados: revisa el esquema antes de agregar."
        )
    if not d["Nomunidad"].map(normalizar_texto).isin(["tonelada", "toneladas"]).all():
        raise ValueError("La producción no está expresada exclusivamente en toneladas.")
    for c in NUMS_AGRI:
        # Solo ausencias declaradas quedan como NaN; otros textos provocan error.
        texto = d[c].replace({"": np.nan, "N/D": np.nan, "NA": np.nan, "N.A.": np.nan})
        d[c] = pd.to_numeric(texto, errors="raise")
        if np.isinf(d[c].to_numpy(dtype=float)).any():
            raise ValueError(f"Valores infinitos en {c}.")
        if (d[c].dropna() < 0).any():
            raise ValueError(f"Valores negativos en {c}.")
    esenciales = ["Sembrada", "Cosechada", "Volumenproduccion"]
    if d[esenciales].isna().any().any():
        raise ValueError(
            "Faltan superficie o producción; revisa antes de formar etiquetas."
        )
    if d["Idcultivo"].nunique() != 1:
        raise ValueError(
            "Varios códigos de aguacate en un mismo año: revisar posibles desgloses."
        )
    if d.duplicated(CLAVE_DETALLE, keep=False).any():
        raise ValueError(
            "Hay claves de detalle duplicadas; no se sumarán sin revisar el origen."
        )
    if ((d.Cosechada == 0) & (d.Volumenproduccion > 0)).any():
        raise ValueError("Producción positiva con superficie cosechada cero.")
    d["CVEGEO"] = d.Idestado.astype(str).str.zfill(2) + d.Idmunicipio.astype(
        str
    ).str.zfill(3)
    d["anio_archivo"] = anio
    return d


def resumir_agricola(detalle):
    sumables = [
        "Sembrada",
        "Cosechada",
        "Siniestrada",
        "Volumenproduccion",
        "Valorproduccion",
    ]
    # No se mezclan cultivos o unidades, incluso al agrupar varios estados.
    claves = [
        "Anio",
        "Idestado",
        "Nomestado",
        "Idcultivo",
        "Nomcultivo",
        "Idunidadmedida",
        "Nomunidad",
    ]

    def agregar(campos):
        t = detalle.groupby(campos, dropna=False, as_index=False)[sumables].sum(
            min_count=1
        )
        t["Rendimiento_t_ha"] = np.divide(
            t.Volumenproduccion,
            t.Cosechada,
            out=np.full(len(t), np.nan),
            where=t.Cosechada.to_numpy() > 0,
        )
        t["Precio_medio_pesos_t"] = np.divide(
            t.Valorproduccion,
            t.Volumenproduccion,
            out=np.full(len(t), np.nan),
            where=t.Volumenproduccion.to_numpy() > 0,
        )
        return t

    municipal = agregar(claves + ["Idmunicipio", "Nommunicipio", "CVEGEO"])
    if municipal.duplicated(["Anio", "CVEGEO"]).any():
        raise ValueError(
            "Un municipio-año tiene varios nombres/cultivos/unidades. Resolver antes de crear etiquetas."
        )
    estatal = agregar(claves)
    m = municipal.groupby(["Anio", "Idestado"], as_index=False).agg(
        municipios_con_registro=("CVEGEO", "nunique"),
        total_municipal_t=("Volumenproduccion", "sum"),
    )
    estatal = estatal.merge(m, on=["Anio", "Idestado"], validate="one_to_one")
    if not np.allclose(
        estatal.Volumenproduccion, estatal.total_municipal_t, rtol=0, atol=1e-6
    ):
        raise ValueError("No coinciden total estatal y suma municipal.")
    estatal = estatal.drop(columns="total_municipal_t")
    return municipal.sort_values(["Anio", "CVEGEO"]), estatal.sort_values(
        ["Anio", "Idestado"]
    )
