# MVP de emergencia: rendimiento de aguacate

Pipeline ejecutable para estimar el rendimiento municipal de aguacate en **toneladas por hectárea cosechada (t/ha)** con datos agrícolas observados del SIAP.

## Qué entrega

- Consolida los registros de riego y temporal por municipio y año.
- Calcula la etiqueta como suma de toneladas producidas / suma de hectáreas cosechadas.
- Entrena tres regresores supervisados: ExtraTrees, Random Forest y HistGradientBoosting.
- Evalúa cada modelo mediante backtesting temporal en los últimos tres años disponibles y lo compara con el baseline de persistencia (repetir el último rendimiento observado).
- Reentrena los tres modelos con el historial disponible y guarda el artefacto sklearn entrenado.
- Genera la estimación para el siguiente año después del último año SIAP disponible. En la versión actual, el año objetivo es 2026.

## Datos y alcance

La fuente incluida en el repositorio es el detalle municipal de aguacate SIAP para Michoacán, años 2020–2025. La configuración territorial actual limita la corrida a los 32 municipios seleccionados para el proyecto. Son resultados agregados por municipio, no predicciones de huertas individuales.

Las variables de entrada son exclusivamente históricas: rendimientos previos, media y desviación de hasta tres rendimientos observados, superficie cosechada y volumen de producción del último registro disponible, municipio y año objetivo. No se usan producción ni superficie del mismo año que se intenta predecir.

La versión de emergencia **no incorpora todavía precipitación ERA5-Land ni Sentinel-2**. No debe describirse como un modelo multimodal ni como pronóstico con clima futuro. El siguiente paso de investigación es probar si esas fuentes aportan valor en validación temporal y ampliar el historial de entrenamiento.

## Instalación y ejecución

Requiere Python 3.10 o posterior.

```bash
python -m pip install -r requirements-emergencia.txt
python pipeline_rendimiento_aguacate.py
```

La ejecución por defecto lee el CSV SIAP, usa los municipios de `configuracion.json` y escribe los resultados en `resultados/emergencia_rendimiento/`. Para pronosticar otro año posterior al último dato observado:

```bash
python pipeline_rendimiento_aguacate.py --anio-prediccion 2027
```

Para incluir todos los municipios de Michoacán presentes en el archivo, sin el filtro de 32 municipios:

```bash
python pipeline_rendimiento_aguacate.py --sin-filtro-territorial
```

## Archivos producidos

- `modelos_entrenados.joblib`: los tres modelos ya ajustados. Cargar solo artefactos confiables creados por esta ejecución.
- `backtesting_temporal.csv`: MAE, RMSE, R², WAPE y comparación con persistencia por cada año de prueba.
- `predicciones_2026.csv` (o el año indicado): las predicciones de cada modelo, el baseline y el método recomendado por la validación.
- `panel_municipal.csv`: etiquetas municipales agregadas.
- `metricas.json`: resumen reproducible del ajuste y limitaciones.

## Lectura de la validación inicial

La corrida de emergencia utilizó 192 municipio-año del territorio configurado (32 municipios × 6 años), 160 ejemplos supervisados y cortes temporales 2023, 2024 y 2025. ExtraTrees fue el mejor de los tres modelos de IA en MAE promedio, pero obtuvo aproximadamente **0.498 t/ha**, frente a **0.495 t/ha** de persistencia. Por tanto, la salida también entrega los pronósticos de IA, pero recomienda operacionalmente el baseline hasta que un modelo supere la referencia de forma consistente.

Estos valores son una evaluación retrospectiva pequeña con solo tres años de prueba. No acreditan desempeño en una cosecha futura. Es necesario actualizar SIAP cuando se publique el siguiente año y recalcular las métricas antes de usar la estimación como decisión productiva.

## Reproducibilidad

La semilla de los modelos de árboles está fijada en 42. Los cortes respetan el orden temporal: para probar un año solo se entrena con muestras cuyo año objetivo es anterior. El archivo de salida conserva tanto los modelos entrenados como las tablas y métricas que permiten revisar la predicción.

