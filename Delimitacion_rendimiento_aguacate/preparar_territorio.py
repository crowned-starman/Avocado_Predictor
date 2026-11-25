"""Etiquetas de rendimiento y cruce municipal con la malla existente. Sin inferencia."""

from utilidades import guardar_json, leer_json, proyectar
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from shapely import make_valid
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from datos_agricolas import resumir_agricola, limpiar_agricola, CAMPOS_AGRI
from .mapa_interactivo import crear_mapa

BASE = Path(__file__).resolve().parent
# Proyección local equivalente para superficies; GeoJSON de intercambio en EPSG:4326.
AREA = CRS.from_proj4(
    "+proj=laea +lat_0=19.2 +lon_0=-101.8 +datum=WGS84 +units=m +no_defs"
)


def capa(municipios, claves, ruta):
    features = [
        dict(
            type="Feature",
            properties={"CVEGEO": c, "municipio": n},
            geometry=mapping(g),
        )
        for c, n, g in municipios
        if c in claves
    ]
    guardar_json(ruta, dict(type="FeatureCollection", features=features), compacto=True)


def parche(g, **kwargs):
    vertices, codigos = [], []
    for anillo in [g.exterior, *g.interiors]:
        coords = list(anillo.coords)
        vertices.extend(coords)
        codigos.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(coords) - 2)
            + [MplPath.CLOSEPOLY]
        )
    return PathPatch(MplPath(vertices, codigos), **kwargs)


def dibujar(municipios, claves, ruta, recuadros=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    for c, n, g in municipios:
        geoms = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for pol in geoms:
            ax.add_patch(
                parche(
                    pol,
                    facecolor="#84a95a" if c in claves else "#eeeeee",
                    edgecolor="#66746a",
                    linewidth=0.45,
                )
            )
    if recuadros:
        for ident, g, elegido in recuadros:
            geoms = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            for pol in geoms:
                x, y = pol.exterior.xy
                ax.plot(x, y, color="#db4b35" if elegido else "#aaaaaa", linewidth=0.65)
            if elegido:
                p = g.representative_point()
                ax.text(p.x, p.y, str(ident), fontsize=6, ha="center")
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(19.2)))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(
        "Michoacán: selección municipal con cosecha de aguacate registrada en 2020–2025\n"
        "Los polígonos representan municipios completos; no huertas individuales"
    )
    leyenda = [
        Patch(facecolor="#84a95a", label=f"Municipios seleccionados ({len(claves)})"),
        Patch(
            facecolor="#eeeeee",
            label="Municipios no seleccionados",
        ),
    ]
    if recuadros:
        leyenda.append(
            Patch(
                facecolor="none", edgecolor="#db4b35", label="Recuadros seleccionados"
            )
        )
    ax.legend(handles=leyenda, loc="lower left", fontsize=8)
    ax.text(
        0.99,
        0.01,
        "Fuente: INEGI, Marco Geoestadístico dic. 2025; DGSIAP, cierres 2020–2025",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def buscar_datos(args, config):
    if args.datos:
        return Path(args.datos).resolve()
    posibles = []
    if args.config_ingesta:
        posibles.append(Path(args.config_ingesta).resolve())
    posibles.extend(
        [BASE.parent / "configuracion.json", Path.cwd() / "configuracion.json"]
    )
    for p in posibles:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8-sig"))
            if "carpeta_datos" in d:
                ruta = Path(d["carpeta_datos"])
                return ruta if ruta.is_absolute() else (p.parent / ruta).resolve()
    for raiz in [BASE.parent, Path.cwd()]:
        for prefijo in ["dato", "datos", ""]:
            p = raiz / prefijo / "revision_datos_aguacate"
            if (p / "sentinel2").exists():
                return p.resolve()
    return None


def cruzar_recuadros(datos, municipios, claves, salida, fraccion_minima=0.85, cortes_preseleccionados=None):
    s2 = datos / "sentinel2"
    ruta = s2 / "cortes_georreferenciados.csv"
    if not ruta.exists():
        ruta = s2 / "plan_recuadros.csv"
    if not ruta.exists():
        return {
            "estado": "pendiente",
            "motivo": f"No existe cortes_georreferenciados.csv ni plan_recuadros.csv en {s2}",
        }
    from cobertura_territorial import cargar_malla, calcular_cobertura, escribir_filtro

    t, crs = cargar_malla(ruta, s2 / "metadatos.json")
    t, geometrias, union = calcular_cobertura(t, crs, municipios, claves, fraccion_minima, cortes_preseleccionados)
    resumen_filtro = escribir_filtro(salida, t, geometrias, fraccion_minima)
    elegidos = t.seleccionado_por_territorio.to_numpy(dtype=bool)
    recuadros = [(int(f.corte), g, bool(sel)) for f, g, sel in zip(t.itertuples(), geometrias, elegidos)]
    seleccion = [(c, n, proyectar(g, 4326, AREA)) for c, n, g in municipios if c in claves]
    intersecciones = []
    for ident, g, elegido in recuadros:
        if not elegido:
            continue
        gm = proyectar(g, 4326, AREA)
        for c, n, m in seleccion:
            area = gm.intersection(m).area
            if area > 1:
                intersecciones.append(dict(corte=ident, CVEGEO=c, municipio=n,
                    interseccion_geometrica_ha=area / 10000, fraccion_rectangulo=area / gm.area))
    pd.DataFrame(intersecciones, columns=["corte", "CVEGEO", "municipio",
        "interseccion_geometrica_ha", "fraccion_rectangulo"]).to_csv(
            salida / "recuadros_por_municipio.csv", index=False, encoding="utf-8-sig")
    # Cobertura de rectángulos planeados, no disponibilidad de píxeles ni de huertas.
    huella = unary_union([proyectar(g, 4326, AREA) for _, g, sel in recuadros if sel])
    cobertura = [
        dict(
            CVEGEO=c,
            municipio=n,
            fraccion_municipio_en_malla_planeada=m.intersection(huella).area / m.area,
            fraccion_municipio_en_recuadros_aceptados=m.intersection(huella).area / m.area,
        )
        for c, n, m in seleccion
    ]
    pd.DataFrame(cobertura).to_csv(
        salida / "cobertura_geometrica_municipal.csv", index=False, encoding="utf-8-sig"
    )
    dibujar(municipios, claves, salida / "02_municipios_y_recuadros.png", recuadros)
    capa_recuadros = dict(
        type="FeatureCollection",
        features=[
            dict(
                type="Feature",
                properties={"corte": ident, "seleccionado": sel},
                geometry=mapping(g),
            )
            for ident, g, sel in recuadros
        ],
    )
    guardar_json(salida / "recuadros.geojson", capa_recuadros)
    return dict(
        estado="realizado",
        archivo=str(ruta),
        crs_original=str(crs),
        posiciones=len(t),
        seleccionadas=int(sum(elegidos)),
        descartadas=int(len(t) - sum(elegidos)),
        fraccion_minima_aoi=float(fraccion_minima),
        comparacion_umbrales=resumen_filtro["comparacion_umbrales"],
        nota="Cobertura mínima por unión municipal. No confirma huertas ni píxeles sin nubes. No borra TIFF ni renumera recuadros.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datos",
        help="Carpeta revision_datos_aguacate que contiene sentinel2 y era5_land",
    )
    p.add_argument(
        "--config-ingesta", help="Ruta al configuracion.json de tu prueba de ingesta"
    )
    p.add_argument("--config", default=str(BASE.parent / "configuracion.json"))
    p.add_argument("--salida", help="Directorio de salida opcional")
    args = p.parse_args()
    if not args.config_ingesta:
        args.config_ingesta = args.config
    global_cfg = leer_json(args.config)
    cfg = global_cfg.get("territorio", global_cfg)
    if not isinstance(cfg.get("cvegeo_seleccionados", []), list) or any(
        not isinstance(c, str) or len(c) != 5 or not c.isdigit()
        for c in cfg.get("cvegeo_seleccionados", [])
    ):
        raise ValueError(
            'CVEGEO debe ser una lista de textos de cinco dígitos, por ejemplo ["16102"].'
        )
    if not cfg.get("meses_emision_propuestos") or any(
        not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= 12
        for m in cfg["meses_emision_propuestos"]
    ):
        raise ValueError("Los meses de emisión deben ser enteros entre 1 y 12.")
    if len(set(cfg["meses_emision_propuestos"])) != len(
        cfg["meses_emision_propuestos"]
    ):
        raise ValueError("Los meses de emisión no pueden repetirse.")
    salida = (
        Path(args.salida).resolve() if args.salida else BASE / "resultados_territorio"
    )
    salida.mkdir(parents=True, exist_ok=True)
    # Elimina únicamente salidas de este módulo que podrían quedar desactualizadas.
    for nombre in [
        "informe_territorio.json",
        "plan_territorial_completo.csv",
        "recuadros_seleccionados.csv",
        "recuadros_descartados.csv",
        "informe_filtro.json",
        "recuadros_por_municipio.csv",
        "cobertura_geometrica_municipal.csv",
        "02_municipios_y_recuadros.png",
        "recuadros.geojson",
        "cobertura_aoi_lluvia.csv",
    ]:
        (salida / nombre).unlink(missing_ok=True)
    fuente = BASE / "fuentes"
    detalle = pd.read_csv(
        fuente / "aguacate_detalle_2020_2025.csv", dtype=str, keep_default_na=False
    )
    faltan = set(CAMPOS_AGRI) - set(detalle.columns)
    if faltan:
        raise ValueError(f"Faltan campos agrícolas: {faltan}")
    partes = [
        limpiar_agricola(detalle.loc[pd.to_numeric(detalle.Anio) == a].copy(), a, [16])
        for a in range(2020, 2026)
    ]
    municipal, estatal = resumir_agricola(pd.concat(partes, ignore_index=True))
    geo = json.loads(
        (fuente / "michoacan_inegi_original.json").read_text(encoding="utf-8")
    )
    municipios = []
    for f in geo["features"]:
        props = f["properties"]
        g = make_valid(shape(f["geometry"]))
        crs = f["geometry"].get("crs", {}).get("properties", {}).get("name")
        if crs != "EPSG:6365":
            raise ValueError("CRS de INEGI inesperado.")
        municipios.append((props["cvegeo"], props["nomgeo"], proyectar(g, crs, 4326)))
    catalogo = pd.DataFrame(
        [{"CVEGEO": c, "municipio_inegi": n} for c, n, g in municipios]
    )
    municipal = municipal.merge(
        catalogo, on="CVEGEO", how="left", validate="many_to_one"
    )
    sin_geo = municipal.loc[municipal.municipio_inegi.isna()]
    sin_geo.to_csv(
        salida / "claves_sin_geometria.csv", index=False, encoding="utf-8-sig"
    )
    if len(sin_geo):
        raise ValueError(
            "Hay claves agrícolas sin geometría; revisar claves_sin_geometria.csv."
        )
    validas = municipal.loc[municipal.Cosechada > 0].copy()
    candidatos = set(validas.CVEGEO)
    manual = cfg.get("cvegeo_seleccionados", [])
    if manual and not set(manual) <= candidatos:
        raise ValueError("Hay claves seleccionadas sin cosecha positiva en 2020–2025.")
    claves = set(manual) if manual else candidatos
    etiquetas = validas.loc[validas.CVEGEO.isin(claves)].copy()
    etiquetas["uso_temporal"] = "desarrollo_2020_2025"
    etiquetas.to_csv(
        salida / "rendimiento_municipio_anio_2020_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    estatal.to_csv(
        salida / "rendimiento_estatal_2020_2025.csv", index=False, encoding="utf-8-sig"
    )
    resumen = validas.groupby(["CVEGEO", "municipio_inegi"], as_index=False).agg(
        anios_con_etiqueta=("Anio", "nunique"),
        superficie_cosechada_ha_anio_promedio=("Cosechada", "mean"),
        produccion_acumulada_2020_2025_t=("Volumenproduccion", "sum"),
    )
    resumen["seleccionado"] = resumen.CVEGEO.isin(claves)
    resumen.to_csv(
        salida / "municipios_candidatos.csv", index=False, encoding="utf-8-sig"
    )
    todas = validas.groupby(["CVEGEO", "municipio_inegi"], as_index=False).agg(
        anios_con_etiqueta=("Anio", "nunique"),
        superficie_cosechada_ha_anio_promedio=("Cosechada", "mean"),
    )
    crear_mapa(
        municipios,
        todas,
        claves,
        salida / "mapa_seleccion.html",
        cfg["meses_emision_propuestos"],
    )
    presentes = set(zip(etiquetas.Anio, etiquetas.CVEGEO))
    pd.DataFrame(
        [
            dict(Anio=a, CVEGEO=c)
            for c in sorted(claves)
            for a in range(2020, 2026)
            if (a, c) not in presentes
        ],
        columns=["Anio", "CVEGEO"],
    ).to_csv(salida / "municipio_anio_sin_etiqueta.csv", index=False)
    capa(
        municipios,
        {c for c, n, g in municipios},
        salida / "michoacan_municipios.geojson",
    )
    capa(municipios, claves, salida / "municipios_seleccionados.geojson")
    dibujar(municipios, claves, salida / "01_municipios_candidatos.png")
    # Plantilla de emisión: mismo objetivo anual, cuatro fechas propuestas, no cuatro etiquetas distintas.
    filas = [
        dict(
            CVEGEO=c,
            anio_objetivo=2026,
            fecha_emision=f"2026-{mes:02d}-01",
            ultima_observacion_permitida=(
                dt.date(2026, mes, 1) - dt.timedelta(days=1)
            ).isoformat(),
            rendimiento_real_t_ha="",
            estado="pendiente_datos_y_etiqueta_2026",
        )
        for c in sorted(claves)
        for mes in cfg["meses_emision_propuestos"]
    ]
    pd.DataFrame(filas).to_csv(
        salida / "plantilla_emisiones_2026.csv", index=False, encoding="utf-8-sig"
    )
    datos = buscar_datos(args, cfg)
    from cobertura_territorial import RECUADROS_MAPA
    cruce = (
        cruzar_recuadros(datos, municipios, claves, salida, cfg.get("fraccion_minima_aoi", 0.85),
                         cfg.get("cortes_preseleccionados") or RECUADROS_MAPA)
        if datos
        else dict(
            estado="pendiente",
            motivo="No se encontró la carpeta local de tus recuadros. Usa --datos o --config-ingesta.",
        )
    )
    # La prueba previa solicitaba lluvia solo para el rectángulo de búsqueda.
    clima = dict(
        estado="pendiente_revision",
        nota="No se asigna una serie regional única a todos los municipios.",
    )
    if datos and (datos / "era5_land/metadatos.json").exists():
        em = json.loads(
            (datos / "era5_land/metadatos.json").read_text(encoding="utf-8")
        )
        if em.get("aoi_geojson"):
            ge = proyectar(shape(em["aoi_geojson"]), 4326, AREA)
            tabla = [
                dict(
                    CVEGEO=c,
                    municipio=n,
                    fraccion_territorio_en_aoi_lluvia=proyectar(g, 4326, AREA)
                    .intersection(ge)
                    .area
                    / proyectar(g, 4326, AREA).area,
                )
                for c, n, g in municipios
                if c in claves
            ]
            pd.DataFrame(tabla).to_csv(
                salida / "cobertura_aoi_lluvia.csv", index=False, encoding="utf-8-sig"
            )
            clima = dict(
                estado="aoi_revisado",
                nota="Fracción del AOI solicitado, no celdas válidas ni cobertura de huertas.",
                dias_disponibles=em.get("dias_representados_utc", []),
            )
    informe = dict(
        objetivo="Rendimiento de aguacate: producción (t) / superficie cosechada (ha)",
        unidad_territorial="municipio",
        municipios_oficiales=len(municipios),
        municipios_candidatos=len(candidatos),
        municipios_seleccionados=len(claves),
        etiquetas_municipio_anio=len(etiquetas),
        periodo_desarrollo=[2020, 2025],
        periodo_prueba=2026,
        seleccion="Con cosecha positiva en al menos un año 2020–2025; lista candidata, no máscara de huertas.",
        metadatos_inegi=geo.get("metadatos"),
        cruce_recuadros=cruce,
        lluvia=clima,
        advertencias_datos=[
            "No se imputan ceros a municipios-año sin registro o sin superficie cosechada.",
            "Los cierres anuales no proporcionan etiquetas mensuales de rendimiento.",
            "La cifra definitiva 2025 se publicó en agosto de 2026; usar disponibilidad real en cada corte.",
            "2026 se reserva para prueba; no se usa para elegir municipios o ajustar modelos.",
            "No se han entrenado ni evaluado modelos mediante este módulo.",
        ],
    )
    guardar_json(salida / "informe_territorio.json", informe)
    print(
        f"Listo: {len(claves)} municipios candidatos; {len(etiquetas)} etiquetas anuales reales."
    )
    print("Cruce de recuadros:", cruce["estado"])
    if cruce["estado"] == "pendiente":
        print(cruce["motivo"])
    print("Resultados:", salida)


if __name__ == "__main__":
    main()
