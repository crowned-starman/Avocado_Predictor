#!/usr/bin/env python3
"""Pipeline reproducible SIAP para estimar rendimiento municipal de aguacate (t/ha).

Entrena candidatos únicamente con variables rezagadas, realiza backtesting
temporal y guarda un modelo sklearn entrenado junto con métricas y pronósticos.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLUMNAS = {"Anio", "CVEGEO", "Nommunicipio", "Cosechada", "Volumenproduccion"}
NUMERICAS = [
    "rendimiento_lag1", "rendimiento_lag2", "rendimiento_media_3",
    "rendimiento_desv_3", "superficie_lag1_ha", "produccion_lag1_t",
    "anios_desde_ultimo_registro", "anio_objetivo",
]
CATEGORICAS = ["CVEGEO"]
VARIABLES = CATEGORICAS + NUMERICAS


def cargar_panel(ruta: Path, territorio: Path | None) -> pd.DataFrame:
    raw = pd.read_csv(ruta, encoding="utf-8-sig", dtype={"CVEGEO": str})
    faltan = COLUMNAS - set(raw.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias del SIAP: {sorted(faltan)}")
    for col in ["Anio", "Cosechada", "Volumenproduccion"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    raw["CVEGEO"] = raw["CVEGEO"].str.strip().str.zfill(5)
    if raw[["Anio", "Cosechada", "Volumenproduccion"]].isna().any().any():
        raise ValueError("Hay años, superficies o volúmenes nulos; revisa el origen.")
    if (raw[["Cosechada", "Volumenproduccion"]] < 0).any().any():
        raise ValueError("Se encontraron superficies o volúmenes negativos.")
    if ((raw.Cosechada == 0) & (raw.Volumenproduccion > 0)).any():
        raise ValueError("Producción positiva con superficie cosechada cero.")
    if raw.duplicated(["Anio", "CVEGEO", "Idmodalidad"] if "Idmodalidad" in raw else ["Anio", "CVEGEO"]).any():
        # Las filas de riego/temporal son válidas si cada modalidad es única.
        if "Idmodalidad" not in raw or raw.duplicated(["Anio", "CVEGEO", "Idmodalidad"]).any():
            raise ValueError("Se encontraron filas SIAP duplicadas para la misma modalidad.")
    if territorio is not None:
        config = json.loads(territorio.read_text(encoding="utf-8"))
        seleccionados = config.get("territorio", {}).get("cvegeo_seleccionados", [])
        if seleccionados:
            raw = raw[raw.CVEGEO.isin([str(c).zfill(5) for c in seleccionados])].copy()
            if raw.empty:
                raise ValueError("La selección territorial no coincide con el CSV.")
    panel = (
        raw.groupby(["Anio", "CVEGEO", "Nommunicipio"], as_index=False)
        .agg(superficie_cosechada_ha=("Cosechada", "sum"),
             produccion_t=("Volumenproduccion", "sum"))
    )
    panel["rendimiento_t_ha"] = np.divide(
        panel.produccion_t, panel.superficie_cosechada_ha,
        out=np.full(len(panel), np.nan), where=panel.superficie_cosechada_ha.to_numpy() > 0,
    )
    panel = panel[panel.rendimiento_t_ha.notna()].sort_values(["CVEGEO", "Anio"])
    if panel.duplicated(["Anio", "CVEGEO"]).any():
        raise ValueError("La agregación no produjo una sola fila por municipio-año.")
    return panel.reset_index(drop=True)


def crear_fila_objetivo(grupo: pd.DataFrame, anio_objetivo: int) -> dict | None:
    hist = grupo[grupo.Anio < anio_objetivo].sort_values("Anio")
    if hist.empty:
        return None
    rend = hist.rendimiento_t_ha.to_numpy(dtype=float)
    ultimo = hist.iloc[-1]
    ultimos3 = rend[-3:]
    return {
        "Anio": int(anio_objetivo),
        "CVEGEO": str(ultimo.CVEGEO),
        "Nommunicipio": str(ultimo.Nommunicipio),
        "rendimiento_lag1": float(rend[-1]),
        "rendimiento_lag2": float(rend[-2]) if len(rend) >= 2 else float(rend[-1]),
        "rendimiento_media_3": float(np.mean(ultimos3)),
        "rendimiento_desv_3": float(np.std(ultimos3, ddof=0)) if len(ultimos3) > 1 else 0.0,
        "superficie_lag1_ha": float(ultimo.superficie_cosechada_ha),
        "produccion_lag1_t": float(ultimo.produccion_t),
        "anios_desde_ultimo_registro": int(anio_objetivo - int(ultimo.Anio)),
        "anio_objetivo": int(anio_objetivo),
    }


def crear_muestras(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clave, grupo in panel.groupby("CVEGEO", sort=True):
        for anio in sorted(grupo.Anio.unique()):
            if anio <= int(grupo.Anio.min()):
                continue
            f = crear_fila_objetivo(grupo, int(anio))
            etiqueta = grupo.loc[grupo.Anio.eq(anio), "rendimiento_t_ha"]
            if f and len(etiqueta) == 1:
                f["y"] = float(etiqueta.iloc[0])
                filas.append(f)
    if not filas:
        raise ValueError("No hay pares de años previos y etiqueta para entrenar.")
    return pd.DataFrame(filas).sort_values(["Anio", "CVEGEO"]).reset_index(drop=True)


def construir_modelo(estimador) -> Pipeline:
    prep = ColumnTransformer(
        [("municipio", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
         ("numericas", "passthrough", NUMERICAS)],
        remainder="drop",
    )
    return Pipeline([("preparacion", prep), ("modelo", estimador)])


def candidatos() -> dict:
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=7, l2_regularization=1.0, random_state=42),
    }


def metricas(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    denom = float(np.abs(y_real).sum())
    return {"n": int(len(y_real)), "MAE_t_ha": mae, "RMSE_t_ha": rmse,
            "R2": float(r2_score(y_real, y_pred)) if len(y_real) > 1 else None,
            "WAPE_pct": float(100 * np.abs(y_real - y_pred).sum() / denom) if denom else None}


def entrenar_y_evaluar(panel: pd.DataFrame, muestras: pd.DataFrame, salida: Path, anio_pred: int) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    anios = sorted(muestras.Anio.unique().astype(int))
    cortes = [int(a) for a in anios if a >= anios[-3] and (muestras.Anio < a).nunique() >= 2]
    if not cortes:
        raise ValueError("Se requieren al menos tres años objetivo para backtesting temporal.")
    resultados = []
    resumen_candidatos = []
    for nombre, estimador in candidatos().items():
        scores = []
        for corte in cortes:
            train = muestras[muestras.Anio < corte]
            test = muestras[muestras.Anio == corte]
            modelo = construir_modelo(estimador.__class__(**estimador.get_params()))
            modelo.fit(train[VARIABLES], train.y)
            pred = np.maximum(0, modelo.predict(test[VARIABLES]))
            m = metricas(test.y, pred)
            base = metricas(test.y, test.rendimiento_lag1)
            resultados.append({"modelo": nombre, "anio_test": int(corte), **m,
                               "baseline_persistencia_MAE_t_ha": base["MAE_t_ha"],
                               "baseline_persistencia_RMSE_t_ha": base["RMSE_t_ha"]})
            scores.append(m["MAE_t_ha"])
        resumen_candidatos.append({"modelo": nombre, "MAE_promedio_folds_t_ha": float(np.mean(scores))})
    elegido = min(resumen_candidatos, key=lambda x: x["MAE_promedio_folds_t_ha"])["modelo"]
    specs = candidatos()
    modelos_finales = {}
    for nombre, estimador in specs.items():
        modelo = construir_modelo(estimador)
        modelo.fit(muestras[VARIABLES], muestras.y)
        modelos_finales[nombre] = modelo
    joblib.dump(modelos_finales, salida / "modelos_entrenados.joblib")
    score_baseline = float(np.mean([
        r["baseline_persistencia_MAE_t_ha"] for r in resultados if r["modelo"] == elegido
    ]))
    score_ia = next(x["MAE_promedio_folds_t_ha"] for x in resumen_candidatos if x["modelo"] == elegido)
    usar_ia = score_ia < score_baseline
    pred_filas = []
    for _, grupo in panel.groupby("CVEGEO", sort=True):
        f = crear_fila_objetivo(grupo, anio_pred)
        if f:
            x = pd.DataFrame([f])[VARIABLES]
        preds = {nombre: max(0.0, float(modelo.predict(x)[0])) for nombre, modelo in modelos_finales.items()}
        estimacion_ia = preds[elegido]
        pred_filas.append({"Anio": anio_pred, "CVEGEO": f["CVEGEO"],
                           "Nommunicipio": f["Nommunicipio"],
                           **{f"prediccion_{k}_t_ha": v for k, v in preds.items()},
                           "baseline_ultimo_rendimiento_t_ha": f["rendimiento_lag1"],
                           "estimacion_recomendada_t_ha": estimacion_ia if usar_ia else f["rendimiento_lag1"],
                           "metodo_recomendado": elegido if usar_ia else "Persistencia (baseline)",
                           "ultimo_anio_observado": anio_pred - f["anios_desde_ultimo_registro"]})
    pd.DataFrame(resultados).to_csv(salida / "backtesting_temporal.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_filas).to_csv(salida / f"predicciones_{anio_pred}.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(salida / "panel_municipal.csv", index=False, encoding="utf-8-sig")
    informe = {
        "objetivo": "Rendimiento de aguacate en toneladas por hectárea cosechada",
        "fuente": "SIAP, detalle agrícola municipal 2020-2025 incluido en el repositorio",
        "entorno_entrenamiento": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                                  "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "territorio_municipios": int(panel.CVEGEO.nunique()),
        "registros_municipio_anio": int(len(panel)),
        "periodo_observado": [int(panel.Anio.min()), int(panel.Anio.max())],
        "anio_predicho": int(anio_pred),
        "muestras_supervisadas": int(len(muestras)),
        "cortes_temporales": cortes,
        "seleccion_modelo_IA_por_MAE_promedio": elegido,
        "comparacion_candidatos": resumen_candidatos,
        "MAE_promedio_baseline_persistencia_t_ha": score_baseline,
        "recomendacion_operativa": elegido if usar_ia else "Persistencia (baseline); la IA se conserva para evaluación y mejora",
        "advertencia_baseline": "La evaluación usa tres cortes temporales. El pronóstico recomendado usa persistencia cuando su MAE promedio supera al del baseline.",
        "archivos": ["modelos_entrenados.joblib", "backtesting_temporal.csv",
                     f"predicciones_{anio_pred}.csv", "panel_municipal.csv"],
        "limitaciones": [
            "Modelo v1 con historial SIAP municipal; no incorpora clima ERA5-Land ni Sentinel-2.",
            "Validación retrospectiva por año; no es una validación independiente con cosecha futura.",
            "Pronóstico municipal; no representa variabilidad entre huertas del mismo municipio."
        ],
    }
    (salida / "metricas.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    return informe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("Delimitacion_rendimiento_aguacate/fuentes/aguacate_detalle_2020_2025.csv"))
    parser.add_argument("--configuracion", type=Path, default=Path("configuracion.json"))
    parser.add_argument("--sin-filtro-territorial", action="store_true", help="Usa todos los municipios presentes en el CSV en vez de los seleccionados en configuracion.json.")
    parser.add_argument("--salida", type=Path, default=Path("resultados/emergencia_rendimiento"))
    parser.add_argument("--anio-prediccion", type=int, default=None, help="Por defecto, año posterior al último año SIAP.")
    args = parser.parse_args()
    territorio = None if args.sin_filtro_territorial else args.configuracion
    panel = cargar_panel(args.datos, territorio)
    muestras = crear_muestras(panel)
    anio_pred = args.anio_prediccion or int(panel.Anio.max()) + 1
    if anio_pred <= int(panel.Anio.max()):
        parser.error("--anio-prediccion debe ser posterior al último año observado para evitar fuga temporal.")
    informe = entrenar_y_evaluar(panel, muestras, args.salida, anio_pred)
    print(json.dumps(informe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

