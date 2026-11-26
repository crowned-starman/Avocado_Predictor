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

## Clima y Sentinel-2

La descarga de precipitación diaria no requiere una cuenta CDS. Obtén los datos ERA5 desde Open-Meteo:

```powershell
python lluvia_open_meteo.py --anios 2020 2021 2022 2023 2024 2025
```

El descargador usa un punto representativo por cada uno de los 32 municipios seleccionados. ERA5 tiene una resolución aproximada de 25 km, así que la lluvia es un contexto regional, no un promedio municipal ni de huertas.

### Generar la malla Sentinel-2

El CSV de la cuadrícula original no está incluido. Para no depender de ese archivo, genera una malla regular nueva de 10 × 10 km usando el CRS de área del proyecto y los polígonos municipales INEGI:

```powershell
python generar_malla_sentinel.py
```

Esto crea `datos/sentinel2/malla_municipal_10km.csv` y sus metadatos. La malla sirve para localizar escenas y crear ventanas de lectura; el extractor aplica los índices dentro de los polígonos exactos de los 32 municipios. No se reutilizan los 49 números de recuadro de la selección anterior, porque sus coordenadas originales no están disponibles. Los índices representan vegetación general municipal, no huertas individuales.

Primero procesa la muestra piloto: el año 2025 y los tres municipios enumerados en `piloto.municipios_piloto` dentro de `configuracion.json`:

```powershell
python extraer_features_sentinel.py --anio 2025 --malla datos/sentinel2/malla_municipal_10km.csv --piloto
```

La salida de la muestra queda aislada en `datos/sentinel_piloto/`. Sirve para revisar que se consulten escenas y se generen NDVI, EVI, SAVI y NDWI; no es suficiente para entrenar ni evaluar el modelo temporal.

Cuando la muestra termine correctamente, para el conjunto de desarrollo completo ejecuta todos los años 2020–2025 sin `--piloto`:

```powershell
for ($anio = 2020; $anio -le 2025; $anio++) {
    python extraer_features_sentinel.py --anio $anio --malla datos/sentinel2/malla_municipal_10km.csv --malla-municipal
    if ($LASTEXITCODE -ne 0) { break }
}
```

El extractor lee COG públicos de Earth Search por HTTP Range, aplica el filtro SCL y calcula los índices con reflectancia L2A. Conserva catálogos y cachés SQLite locales para reanudar. No guarda en un bucket la estimación de imágenes nativas sin comprimir; esta es una cota de volumen de ventanas raster, no la ocupación de disco de la extracción.

Integra la lluvia y los resúmenes satelitales y luego ejecuta el pipeline:

```powershell
python preparar_features_ambientales.py --lluvia-root datos/lluvia_open_meteo --sentinel-anual datos/sentinel --salida datos/features_ambientales.csv
python pipeline_rendimiento_aguacate.py --features-ambientales datos/features_ambientales.csv
```

El CSV integrado contiene, por municipio-año: precipitación total anual, días lluviosos, racha seca máxima, cobertura de lluvia, mediana de NDVI/EVI/SAVI/NDWI y escenas válidas. El modelo crea rezagos y promedios de hasta tres años; al predecir 2026 solo usa mediciones hasta 2025. Si faltan filas ambientales para los municipios requeridos, el pipeline se detiene y no produce un resultado parcial.

Las predicciones incluidas en el repositorio se entrenaron solo con SIAP. El modelo con lluvia y Sentinel-2 se entrenará después de descargar, integrar y revisar ambas fuentes en tu equipo.
