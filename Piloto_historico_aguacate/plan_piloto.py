"""Planifica muestras constantes dentro de municipios y de la malla local del usuario."""

from utilidades import proyectar
from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import CRS
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from shapely import make_valid
from utilidades import guardar_json, leer_json, firma_objeto, sha256
from .comun_piloto import VERSION_PLAN, ubicaciones


def crear_plan(cfg, p):
    raiz = Path(cfg["_raiz"])
    territorio = raiz / "Delimitacion_rendimiento_aguacate" / "resultados_territorio"
    archivo_geo = territorio / "municipios_seleccionados.geojson"
    archivo_etiquetas = territorio / "rendimiento_municipio_anio_2020_2025.csv"
    if not archivo_geo.exists() or not archivo_etiquetas.exists():
        raise FileNotFoundError(
            "Primero guarda tu selección municipal y ejecuta preparar_territorio.py. Faltan los resultados territoriales locales."
        )
    geo = leer_json(archivo_geo)
    municipios = {f["properties"]["CVEGEO"]: f for f in geo["features"]}
    if not set(p["municipios_piloto"]) <= set(municipios):
        raise ValueError(
            "Algún municipio piloto no está en tu selección. Ajusta municipios_piloto o revisa el resultado territorial."
        )
    etiquetas = pd.read_csv(archivo_etiquetas, dtype={"CVEGEO": str})
    if etiquetas.duplicated(["Anio", "CVEGEO"]).any():
        raise ValueError("Etiquetas agrícolas municipio-año duplicadas.")
    s2 = Path(cfg["carpeta_datos"]) / "sentinel2"
    archivo_malla = s2 / "cortes_georreferenciados.csv"
    if not archivo_malla.exists():
        archivo_malla = s2 / "plan_recuadros.csv"
    if not archivo_malla.exists():
        raise FileNotFoundError(
            f"Falta la tabla de recuadros georreferenciados en {s2}. No se deduce la malla de una imagen."
        )
    malla = pd.read_csv(archivo_malla)
    if not {"corte", "xmin", "ymin", "xmax", "ymax"} <= set(malla):
        raise ValueError("La tabla de recuadros no contiene sus límites y números.")
    if malla.corte.duplicated().any():
        raise ValueError("Números de recuadro duplicados.")
    meta = leer_json(s2 / "metadatos.json") if (s2 / "metadatos.json").exists() else {}
    crss = malla.crs.dropna().astype(str).unique() if "crs" in malla else []
    if len(crss) > 1:
        raise ValueError("La malla combina varios CRS.")
    crs = str(crss[0]) if len(crss) else meta.get("crs")
    if (
        not crs
        or not CRS.from_user_input(crs).is_projected
        or CRS.from_user_input(crs).axis_info[0].unit_name != "metre"
    ):
        raise ValueError("Se necesita el CRS métrico de la malla original.")
    cuadros = [box(r.xmin, r.ymin, r.xmax, r.ymax) for r in malla.itertuples()]
    malla_union = unary_union(cuadros)
    origen_x = float(malla.xmin.min())
    origen_y = float(malla.ymax.max())
    h = p["tamanio_parche_px"]
    lado = h * 10
    lugares = []
    casos = []
    anios = sorted(p["anios_contexto"] + [p["anio_consulta"]])
    for cve in p["municipios_piloto"]:
        f = municipios[cve]
        g = make_valid(shape(f["geometry"]))
        gm = proyectar(g, 4326, crs)
        interior = gm.intersection(malla_union).buffer(-lado)
        if interior.is_empty:
            raise ValueError(f"No cabe una muestra interior de {lado} m en {cve}.")
        centro = interior.representative_point()
        x = origen_x + round((centro.x - lado / 2 - origen_x) / 10) * 10
        y = origen_y + round((centro.y + lado / 2 - origen_y) / 10) * 10
        bb = [x, y - lado, x + lado, y]
        ventana = box(*bb)
        if not gm.covers(ventana) or not malla_union.covers(ventana):
            raise ValueError("La muestra sale del municipio o de la malla.")
        ids = [
            int(row.corte)
            for row, cuadro in zip(malla.itertuples(), cuadros)
            if cuadro.intersection(ventana).area > 1
        ]
        nombre = f["properties"].get("municipio", cve)
        lugares.append(
            dict(
                CVEGEO=cve,
                municipio=nombre,
                crs=crs,
                bbox=bb,
                lado_px=h,
                transform=[10, 0, x, 0, -10, y],
                recuadros_originales=ids,
                geometria_municipal=f["geometry"],
                muestra_geojson=mapping(proyectar(ventana, crs, 4326)),
                alcance="Muestra fija de 640×640 m dentro del municipio; no representa todas sus huertas.",
            )
        )
        for anio in anios:
            etiqueta = etiquetas.loc[
                (etiquetas.CVEGEO == cve) & (etiquetas.Anio == anio)
            ]
            if (
                len(etiqueta) != 1
                or not np.isfinite(etiqueta.Rendimiento_t_ha.iloc[0])
                or etiqueta.Cosechada.iloc[0] <= 0
            ):
                raise ValueError(
                    f"Falta etiqueta anual real con hectáreas cosechadas positivas para {cve}, {anio}."
                )
            casos.append(
                dict(
                    id=f"{cve}_{anio}",
                    CVEGEO=cve,
                    municipio=nombre,
                    anio=anio,
                    papel="consulta" if anio == p["anio_consulta"] else "contexto",
                    inicio_contexto=f"{anio}-11-01",
                    fin_contexto=f"{anio}-11-25",
                    corte_observaciones_utc=f"{anio}-11-26T00:00:00Z",
                    fechas_lluvia_futura=pd.date_range(f"{anio}-11-26", periods=5)
                    .strftime("%Y-%m-%d")
                    .tolist(),
                    fecha_consulta_sentinel=f"{anio}-11-30",
                )
            )
    plan = dict(
        version=VERSION_PLAN,
        configuracion=p,
        lugares=lugares,
        casos=casos,
        fuente_etiquetas=str(archivo_etiquetas),
        sha256_etiquetas=sha256(archivo_etiquetas),
        fuente_malla=str(archivo_malla),
        alcance="Prueba retrospectiva de conexiones; no evaluación de precisión ni pronóstico operativo emitido en el pasado.",
        reserva="No se utilizan datos ni etiquetas 2026.",
    )
    plan["firma_plan"] = firma_objeto(plan)
    datos, salida = ubicaciones(cfg, plan)
    datos.mkdir(parents=True, exist_ok=True)
    salida.mkdir(parents=True, exist_ok=True)
    guardar_json(datos / "plan.json", plan)
    guardar_json(salida / "plan.json", plan)
    pd.DataFrame(casos).drop(columns="fechas_lluvia_futura").to_csv(
        salida / "casos_piloto.csv", index=False, encoding="utf-8-sig"
    )
    guardar_json(
        salida / "muestras_piloto.geojson",
        dict(
            type="FeatureCollection",
            features=[
                dict(
                    type="Feature",
                    properties={
                        k: l[k]
                        for k in [
                            "CVEGEO",
                            "municipio",
                            "recuadros_originales",
                            "alcance",
                        ]
                    },
                    geometry=l["muestra_geojson"],
                )
                for l in lugares
            ],
        ),
    )
    raiz_salida = raiz / "resultados_piloto"
    guardar_json(
        raiz_salida / "ultimo_plan.json",
        {"firma_plan": plan["firma_plan"], "directorio": str(salida)},
    )
    return plan
