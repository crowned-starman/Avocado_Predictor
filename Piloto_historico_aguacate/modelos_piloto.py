"""Inferencia con pesos oficiales: secuencias pequeñas y tabla municipal piloto."""

from recursos_modelos import obtener_tabicl
import gc
import logging
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.transform import Affine
from utilidades import BANDAS, guardar_json, leer_json, sha256
from recursos_modelos import TABICL_REVISION, TABICL_CHECKPOINT, TABICL_SHA256
from .comun_piloto import carpeta_caso, ubicaciones, firma_entradas, validar_lluvia
from .descargas_piloto import evaluar_tiff


def descargar_modelos(cfg):
    from recursos_modelos import obtener_tamrf, obtener_chronos

    return dict(
        tamrf=obtener_tamrf(cfg),
        chronos2=obtener_chronos(cfg),
        tabiclv2=obtener_tabicl(cfg),
    )


def preparar_datos_caso(cfg, plan, caso):
    from preparacion import mapa_bandas

    datos, _ = carpeta_caso(cfg, plan, caso)
    m = leer_json(datos / "sentinel2" / "secuencia.json")
    if m["firma_plan"] != plan["firma_plan"]:
        raise ValueError("La secuencia Sentinel pertenece a otro plan.")
    fechas = pd.DatetimeIndex(pd.to_datetime(m["fechas"]))
    if (
        fechas.has_duplicates
        or not fechas.is_monotonic_increasing
        or len(fechas) != plan["configuracion"]["observaciones_sentinel_por_caso"]
    ):
        raise ValueError("Número, orden o duplicados de fechas Sentinel incompatibles.")
    if len(m["archivos"]) != len(fechas) or any(
        not caso["inicio_contexto"] <= d.strftime("%Y-%m-%d") <= caso["fin_contexto"]
        for d in fechas
    ):
        raise ValueError("Secuencia Sentinel fuera del corte temporal.")
    lugar = next(l for l in plan["lugares"] if l["CVEGEO"] == caso["CVEGEO"])
    arreglos = []
    for fecha, nombre in zip(fechas, m["archivos"]):
        ruta = datos / "sentinel2" / nombre
        if sha256(ruta) != m["sha256"][nombre]:
            raise ValueError(
                "Cambió una muestra Sentinel: vuelve a ejecutar descargar."
            )
        bien, _ = evaluar_tiff(ruta, lugar, plan["configuracion"])
        if not bien:
            raise ValueError(
                "La muestra no satisface los umbrales de calidad del plan."
            )
        with rasterio.open(ruta) as src:
            if pd.Timestamp(src.tags().get("fecha_utc")) != fecha:
                raise ValueError("Fecha GeoTIFF distinta a la secuencia.")
            idx = mapa_bandas(src)
            arreglos.append(src.read([idx[b] for b in BANDAS]).astype(np.float32))
    df = pd.read_csv(datos / "lluvia.csv")
    validar_lluvia(df, caso)
    return np.stack(arreglos)[None], fechas, df, lugar


def liberar(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def ejecutar_dos_modelos(cfg, plan):
    from modelos import cargar_tamrf, dispositivo, predecir_tamrf, cargar_chronos
    from recursos_modelos import CHRONOS_REVISION, TAMRF_ARCHIVOS

    dev = dispositivo(cfg)
    _, salida = ubicaciones(cfg, plan)
    firma = firma_entradas(cfg, plan)
    # La marca se escribe solo al terminar ambos modelos, nunca ante una ejecución parcial.
    (salida / "dos_modelos_completos.json").unlink(missing_ok=True)
    for ruta in [
        salida / "tabla_piloto.csv",
        salida / "informe_tabiclv2.json",
        salida / "predicciones_rendimiento_2025.csv",
    ]:
        ruta.unlink(missing_ok=True)
    encoder, decoder, media, std, canales_lr = cargar_tamrf(cfg, dev)
    todas_lluvias = []
    registros = []
    for caso in plan["casos"]:
        x, fechas, lluvia, lugar = preparar_datos_caso(cfg, plan, caso)
        consulta = pd.Timestamp(caso["fecha_consulta_sentinel"])
        origen = fechas[0]
        if consulta <= fechas.max():
            raise ValueError(
                "TAMRF requiere una consulta posterior a todas las entradas."
            )
        tiempos = torch.tensor(
            ((fechas - origen).days.to_numpy()).astype(np.float32)[None], device=dev
        )
        query = torch.tensor(
            [[(consulta - origen).days]], dtype=torch.float32, device=dev
        )
        tensor = torch.from_numpy(x).to(dev)
        pred = (
            predecir_tamrf(
                encoder, decoder, media, std, canales_lr, tensor, tiempos, query
            )
            .cpu()
            .numpy()
        )
        if pred.shape != (1, 1, 10, 64, 64) or not np.isfinite(pred).all():
            raise ValueError("Salida TAMRF de forma o valores incorrectos.")
        _, out = carpeta_caso(cfg, plan, caso)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "tamrf.npz",
            reflectancia_estimada=pred,
            ultima_observacion=x[:, -1:],
            bandas=np.asarray(BANDAS),
            fecha=np.asarray(caso["fecha_consulta_sentinel"]),
        )
        with rasterio.open(
            out / "tamrf_10_bandas.tif",
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=10,
            dtype="float32",
            crs=lugar["crs"],
            transform=Affine(*lugar["transform"]),
            compress="deflate",
            predictor=3,
        ) as dst:
            dst.write(pred[0, 0])
            for i, b in enumerate(BANDAS, 1):
                dst.set_band_description(i, b)
            dst.update_tags(
                fecha_utc=caso["fecha_consulta_sentinel"],
                tipo="estimacion piloto sin validar precision",
            )
        info = dict(
            id=caso["id"],
            forma_entrada=list(x.shape),
            forma_salida=list(pred.shape),
            fechas_entrada=fechas.strftime("%Y-%m-%d").tolist(),
            fecha_consulta=caso["fecha_consulta_sentinel"],
            pesos_preentrenados=True,
            carga_estricta=True,
            sha256_checkpoint=TAMRF_ARCHIVOS["968639_s1t3.ckpt"][1],
            reflectancia_minima=float(pred.min()),
            reflectancia_maxima=float(pred.max()),
            firma_entradas=firma,
        )
        guardar_json(out / "informe_tamrf.json", info)
        registros.append(info)
        todas_lluvias.append(
            pd.DataFrame(
                {
                    "item_id": caso["id"],
                    "timestamp": pd.to_datetime(lluvia.timestamp),
                    "target": lluvia.precipitacion_mm.astype(np.float32),
                }
            )
        )
        logging.info("TAMRF %s: %s → %s", caso["id"], x.shape, pred.shape)
        del tensor, pred
    del encoder, decoder, media, std
    liberar(dev)
    historial = pd.concat(todas_lluvias, ignore_index=True).sort_values(
        ["item_id", "timestamp"]
    )
    historial.to_csv(salida / "entrada_chronos2.csv", index=False)
    modelo = cargar_chronos(cfg, dev)
    pred = modelo.predict_df(
        historial,
        prediction_length=5,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )
    cols = ["predictions", "0.1", "0.5", "0.9"]
    if (
        not set(cols) <= set(pred)
        or len(pred) != 5 * len(plan["casos"])
        or not np.isfinite(pred[cols].values).all()
    ):
        raise ValueError("Salida Chronos-2 incorrecta.")
    if set(pred.item_id) != set(historial.item_id):
        raise ValueError("Identificadores de Chronos-2 diferentes de los casos.")
    for caso in plan["casos"]:
        g = pred.loc[pred.item_id == caso["id"]].sort_values("timestamp").copy()
        if (
            pd.DatetimeIndex(pd.to_datetime(g.timestamp)).strftime("%Y-%m-%d").tolist()
            != caso["fechas_lluvia_futura"]
        ):
            raise ValueError("Chronos-2 no devolvió las fechas futuras esperadas.")
        g["prediccion_mm_no_negativa"] = g.predictions.clip(lower=0)
        _, out = carpeta_caso(cfg, plan, caso)
        g.to_csv(out / "chronos2.csv", index=False)
    pred.to_csv(salida / "salida_chronos2_cruda.csv", index=False)
    salidas = {
        str((out / n).relative_to(salida)): sha256(out / n)
        for caso in plan["casos"]
        for out in [carpeta_caso(cfg, plan, caso)[1]]
        for n in ["tamrf.npz", "chronos2.csv"]
    }
    info = dict(
        firma_plan=plan["firma_plan"],
        firma_entradas=firma,
        sha256_salidas=salidas,
        casos=len(plan["casos"]),
        tamrf_pesos_oficiales=True,
        chronos2_pesos_oficiales=True,
        chronos2_revision=CHRONOS_REVISION,
        filas_lluvia_futura=len(pred),
        alcance="Comprobación de entradas y salidas; no se calcularon métricas de precisión.",
    )
    guardar_json(salida / "dos_modelos_completos.json", info)
    del modelo
    liberar(dev)
    return info


def probar_tercero(cfg, plan):
    from tabicl import TabICLRegressor
    from modelos import dispositivo
    from .tabla_piloto import comprobar_tabla

    dev = dispositivo(cfg)
    _, salida = ubicaciones(cfg, plan)
    esquema = comprobar_tabla(cfg, plan)
    contexto = pd.read_csv(
        salida / "tabla_contexto_2020_2024.csv", dtype={"CVEGEO": str}
    )
    consulta = pd.read_csv(salida / "tabla_consulta_2025.csv", dtype={"CVEGEO": str})
    if (
        contexto.anio >= plan["configuracion"]["anio_consulta"]
    ).any() or not consulta.anio.eq(plan["configuracion"]["anio_consulta"]).all():
        raise ValueError("La separación temporal del tercer modelo no es correcta.")
    columnas = esquema["caracteristicas"]
    if "rendimiento_t_ha" in consulta or any(
        n in columnas
        for n in ["rendimiento_t_ha", "Volumenproduccion", "Cosechada", "Anio", "anio"]
    ):
        raise ValueError(
            "La consulta o sus características incluyen la etiqueta o columnas no autorizadas."
        )
    x = contexto[columnas].astype(np.float32)
    y = contexto.rendimiento_t_ha.to_numpy(dtype=np.float32)
    q = consulta[columnas].astype(np.float32)
    if (
        not np.isfinite(x).all().all()
        or not np.isfinite(q).all().all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Tabla con valores no finitos.")
    obtener_tabicl(cfg)
    checkpoint = Path(cfg["carpeta_modelos"]) / "tabiclv2" / TABICL_CHECKPOINT
    modelo = TabICLRegressor(
        model_path=checkpoint,
        allow_auto_download=False,
        checkpoint_version=TABICL_CHECKPOINT,
        n_estimators=plan["configuracion"]["tabicl_estimadores"],
        device=str(dev),
        n_jobs=int(cfg.get("hilos_cpu", 2)),
        random_state=42,
        batch_size=1,
        use_amp=False,
        use_fa3=False,
    )
    modelo.fit(x, y)
    pred = np.asarray(modelo.predict(q))
    if pred.shape != (len(consulta),) or not np.isfinite(pred).all():
        raise ValueError("TabICLv2 no devolvió una estimación finita por consulta.")
    resultado = consulta[
        ["CVEGEO", "municipio", "anio", "fecha_consulta_sentinel"]
    ].copy()
    resultado["rendimiento_estimado_t_ha"] = pred
    resultado["rendimiento_estimado_t_ha_no_negativo"] = np.maximum(pred, 0)
    resultado.to_csv(
        salida / "predicciones_rendimiento_2025.csv", index=False, encoding="utf-8-sig"
    )
    info = dict(
        modelo="TabICLv2 Regressor",
        checkpoint=TABICL_CHECKPOINT,
        revision=TABICL_REVISION,
        sha256_checkpoint=TABICL_SHA256,
        pesos_preentrenados=True,
        forma_contexto=list(x.shape),
        forma_consulta=list(q.shape),
        filas_predichas=len(pred),
        unidad="t/ha",
        anios_contexto=sorted(contexto.anio.unique().tolist()),
        anio_consulta=plan["configuracion"]["anio_consulta"],
        etiqueta_2025_usada_en_fit=False,
        datos_2026_utilizados=False,
        actualizacion_de_pesos=False,
        firma_plan=plan["firma_plan"],
        firma_tabla=esquema["firma_tabla"],
        alcance="Prueba técnica conectada: muestras espaciales pequeñas y cinco días pronosticados; no rendimiento municipal validado.",
    )
    guardar_json(salida / "informe_tabiclv2.json", info)
    del modelo
    liberar(dev)
    return info
