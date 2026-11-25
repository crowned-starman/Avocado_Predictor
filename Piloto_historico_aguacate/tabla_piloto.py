"""Mismas características pronosticadas para contexto y consulta, sin etiqueta futura en X."""

import numpy as np
import pandas as pd
from utilidades import BANDAS, guardar_json, leer_json, sha256, firma_objeto
from .comun_piloto import firma_entradas, ubicaciones, carpeta_caso, validar_lluvia


def verificar_dos_modelos(cfg, plan):
    _, salida = ubicaciones(cfg, plan)
    m = leer_json(salida / "dos_modelos_completos.json")
    if m["firma_plan"] != plan["firma_plan"] or m["firma_entradas"] != firma_entradas(
        cfg, plan
    ):
        raise ValueError(
            "Las entradas cambiaron o las salidas pertenecen a otro plan. Ejecuta dos-modelos."
        )
    if any(sha256(salida / n) != h for n, h in m["sha256_salidas"].items()):
        raise ValueError(
            "Cambió una salida de TAMRF o Chronos-2. Ejecuta dos-modelos antes de construir la tabla."
        )
    return m


def indice_promedio(a, b):
    den = a + b
    validos = np.abs(den) > 1e-6
    if not validos.any():
        raise ValueError(
            "No se puede calcular un índice espectral: todos los denominadores son cero."
        )
    valor = float(np.mean((a[validos] - b[validos]) / den[validos]))
    if not np.isfinite(valor):
        raise ValueError("Índice espectral no finito.")
    return valor


def construir_tabla(cfg, plan):
    verificar_dos_modelos(cfg, plan)
    _, salida = ubicaciones(cfg, plan)
    if sha256(plan["fuente_etiquetas"]) != plan["sha256_etiquetas"]:
        raise ValueError("Las etiquetas cambiaron desde la planificación.")
    etiquetas = pd.read_csv(plan["fuente_etiquetas"], dtype={"CVEGEO": str})
    filas = []
    caracteristicas = None
    for caso in plan["casos"]:
        datos, out = carpeta_caso(cfg, plan, caso)
        with np.load(out / "tamrf.npz", allow_pickle=False) as z:
            if (
                z["bandas"].tolist() != BANDAS
                or str(z["fecha"].item()) != caso["fecha_consulta_sentinel"]
            ):
                raise ValueError(
                    "Bandas o fecha de la salida TAMRF diferentes del plan."
                )
            pred = z["reflectancia_estimada"]
            if pred.shape != (1, 1, 10, 64, 64) or not np.isfinite(pred).all():
                raise ValueError("Salida TAMRF incompatible con la tabla.")
            a = pred[0, 0]
        lluvia = pd.read_csv(out / "chronos2.csv")
        if len(lluvia) != 5 or set(lluvia.item_id) != {caso["id"]}:
            raise ValueError("Salida de lluvia no corresponde al caso.")
        fechas = (
            pd.DatetimeIndex(pd.to_datetime(lluvia.timestamp))
            .strftime("%Y-%m-%d")
            .tolist()
        )
        if fechas != caso["fechas_lluvia_futura"]:
            raise ValueError("Fechas de lluvia fuera de consulta.")
        historico = pd.read_csv(datos / "lluvia.csv")
        _, valores = validar_lluvia(historico, caso)
        f = {}
        for i, b in enumerate(BANDAS):
            f[f"{b}_estimada_media"] = float(a[i].mean())
            f[f"{b}_estimada_std"] = float(a[i].std())
        f["ndvi_estimado_medio"] = indice_promedio(
            a[BANDAS.index("B08")], a[BANDAS.index("B04")]
        )
        f["ndmi_estimado_medio"] = indice_promedio(
            a[BANDAS.index("B8A")], a[BANDAS.index("B11")]
        )
        f["lluvia_observada_25dias_mm"] = float(valores.sum())
        f["lluvia_observada_7dias_mm"] = float(valores[-7:].sum())
        f["lluvia_estimada_5dias_mm"] = float(lluvia.prediccion_mm_no_negativa.sum())
        f["lluvia_estimada_media_diaria_mm"] = float(
            lluvia.prediccion_mm_no_negativa.mean()
        )
        for cve in sorted(plan["configuracion"]["municipios_piloto"]):
            f[f"municipio_{cve}"] = int(cve == caso["CVEGEO"])
        if not np.isfinite(list(f.values())).all():
            raise ValueError("Hay características no finitas; no se imputan ceros.")
        if caracteristicas is None:
            caracteristicas = list(f)
        y = etiquetas.loc[
            (etiquetas.CVEGEO == caso["CVEGEO"]) & (etiquetas.Anio == caso["anio"]),
            "Rendimiento_t_ha",
        ]
        if len(y) != 1 or not np.isfinite(y.iloc[0]):
            raise ValueError("Etiqueta anual ausente o duplicada.")
        filas.append(
            dict(
                CVEGEO=caso["CVEGEO"],
                municipio=caso["municipio"],
                anio=caso["anio"],
                papel=caso["papel"],
                fecha_corte_observaciones=caso["fin_contexto"],
                fecha_consulta_sentinel=caso["fecha_consulta_sentinel"],
                **f,
                rendimiento_t_ha=float(y.iloc[0]),
            )
        )
    tabla = pd.DataFrame(filas).sort_values(["anio", "CVEGEO"])
    contexto = tabla.loc[tabla.papel == "contexto"].copy()
    consulta = (
        tabla.loc[tabla.papel == "consulta"].drop(columns="rendimiento_t_ha").copy()
    )
    reservadas = tabla.loc[
        tabla.papel == "consulta", ["CVEGEO", "anio", "rendimiento_t_ha"]
    ].copy()
    if len(contexto) != len(plan["configuracion"]["municipios_piloto"]) * len(
        plan["configuracion"]["anios_contexto"]
    ):
        raise ValueError(
            "Faltan filas históricas; no se sustituirán por otras fechas o lugares."
        )
    # Los nombres indican los años fijos del piloto incluido; configurar otro año no cambia la separación temporal.
    tabla.to_csv(salida / "tabla_piloto.csv", index=False, encoding="utf-8-sig")
    contexto.to_csv(
        salida / "tabla_contexto_2020_2024.csv", index=False, encoding="utf-8-sig"
    )
    consulta.to_csv(
        salida / "tabla_consulta_2025.csv", index=False, encoding="utf-8-sig"
    )
    reservadas.to_csv(
        salida / "etiquetas_consulta_reservadas.csv", index=False, encoding="utf-8-sig"
    )
    archivos = {
        n: sha256(salida / n)
        for n in [
            "tabla_piloto.csv",
            "tabla_contexto_2020_2024.csv",
            "tabla_consulta_2025.csv",
            "etiquetas_consulta_reservadas.csv",
        ]
    }
    esquema = dict(
        firma_plan=plan["firma_plan"],
        caracteristicas=caracteristicas,
        objetivo="rendimiento_t_ha",
        columnas_excluidas_de_X=[
            "CVEGEO",
            "municipio",
            "anio",
            "papel",
            "fecha_corte_observaciones",
            "fecha_consulta_sentinel",
            "rendimiento_t_ha",
        ],
        filas_contexto=len(contexto),
        filas_consulta=len(consulta),
        numero_caracteristicas=len(caracteristicas),
        sha256_archivos=archivos,
        firma_tabla=firma_objeto(archivos),
        nota="Características satelitales de una muestra fija por municipio; no promedio de todas las huertas.",
        produccion_o_superficie_final_del_anio_usadas_en_X=False,
    )
    guardar_json(salida / "esquema_tabla.json", esquema)
    return esquema


def comprobar_tabla(cfg, plan):
    verificar_dos_modelos(cfg, plan)
    _, salida = ubicaciones(cfg, plan)
    e = leer_json(salida / "esquema_tabla.json")
    if e["firma_plan"] != plan["firma_plan"] or any(
        sha256(salida / n) != h for n, h in e["sha256_archivos"].items()
    ):
        raise ValueError(
            "La tabla cambió o pertenece a otro plan. Ejecuta tabla otra vez."
        )
    return e
