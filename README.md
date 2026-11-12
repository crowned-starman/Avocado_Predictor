# Predicción de la producción de aguacate en Michoacán

Proyecto académico de Big Data que busca integrar imágenes satelitales, información climática y registros históricos de producción agrícola para estudiar la evolución del cultivo de aguacate y desarrollar estimaciones de su producción futura.

La propuesta parte de una pregunta: **¿cómo pueden los cambios observados en la vegetación y las condiciones ambientales ayudar a anticipar variaciones en la producción de aguacate?**

## Propósito del proyecto

Se busca aprovechar distintas fuentes de datos para conocer el comportamiento del terreno a lo largo del tiempo, explorar su relación con la producción y generar información que pueda apoyar la planeación agrícola.


## Objetivo general

Desarrollar una propuesta de predicción de la producción de aguacate en una zona de Michoacán mediante la integración de información satelital, climática y agrícola, y evaluar su capacidad para anticipar variaciones en periodos posteriores.

## Objetivos específicos

- Reunir y organizar datos históricos que correspondan a la misma zona y a periodos comparables.
- Analizar cambios en la cobertura vegetal y en indicadores de vigor y humedad de la vegetación.
- Explorar la generación de imágenes e indicadores futuros estimados del terreno.
- Estudiar la relación entre los cambios ambientales y los registros de producción de aguacate.

## Zona de estudio

El estudio se centrará en una **zona de interés delimitada dentro de Michoacán, México**, vinculada al cultivo de aguacate.

Se buscará que la información satelital, las condiciones ambientales y los registros agrícolas representen un territorio comparable. E

## Datos que se prevé utilizar

| Fuente | Información prevista | Utilidad en el proyecto |
|---|---|---|
| **Sentinel-2, a través de Copernicus Data Space** | Imágenes multiespectrales de la zona de estudio. | Examinar cambios en la superficie y obtener indicadores relacionados con la vegetación y su contenido de agua. |
| **ERA5-Land, a través del Climate Data Store de Copernicus** | Registros históricos de variables como precipitación, temperatura y humedad del suelo, según su disponibilidad y pertinencia. | Complementar las imágenes con información ambiental de los mismos periodos. |
| **Estadísticas oficiales de producción agrícola, DGSIAP/SIAP** | Producción de aguacate en toneladas y, cuando estén disponibles, superficie cosechada, rendimiento. | Aportar los valores agrícolas con los que se estudiaría la relación entre las condiciones del terreno y la producción. |

## Periodo de análisis

Se considera **2015–2025 como periodo histórico de referencia**, sujeto a la disponibilidad, calidad y compatibilidad de las fuentes para la zona seleccionada.

La frecuencia de análisis y los horizontes de predicción se definirán conforme al detalle temporal de los datos y a los objetivos del estudio. Se buscará respetar la diferencia entre observaciones frecuentes del terreno y registros de producción que pueden estar agrupados por año o ciclo agrícola.


## Fuentes de referencia

- [Sentinel-2 — Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2).
- [ERA5-Land — Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land).
- [Datos abiertos de estadística de producción agrícola — DGSIAP](https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php).
- [Anuario Estadístico de la Producción Agrícola](https://nube.agricultura.gob.mx/cierre_agricola/).
