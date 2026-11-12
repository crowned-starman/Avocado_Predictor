# Predicción de la producción de aguacate en Michoacán

Proyecto académico de Big Data que busca integrar imágenes satelitales, información climática y registros históricos de producción agrícola para estudiar la evolución del cultivo de aguacate y desarrollar estimaciones de su producción futura.

La propuesta parte de una pregunta: **¿cómo pueden los cambios observados en la vegetación y las condiciones ambientales ayudar a anticipar variaciones en la producción de aguacate?**

## Propósito del proyecto

Se busca aprovechar distintas fuentes de datos para conocer el comportamiento del terreno a lo largo del tiempo, explorar su relación con la producción y generar información que pueda apoyar la planeación agrícola.

Además de analizar lo ocurrido en el pasado, se pretende explorar la estimación de condiciones futuras de la vegetación, como su vigor y los indicadores relacionados con su contenido de agua. Estas estimaciones servirían como información adicional para estudiar posibles cambios en la producción.

## Objetivo general

Desarrollar una propuesta de predicción de la producción de aguacate en una zona de Michoacán mediante la integración de información satelital, climática y agrícola, y evaluar su capacidad para anticipar variaciones en periodos posteriores.

## Objetivos específicos

- Reunir y organizar datos históricos que correspondan a la misma zona y a periodos comparables.
- Analizar cambios en la cobertura vegetal y en indicadores de vigor y humedad de la vegetación.
- Explorar la generación de imágenes e indicadores futuros estimados del terreno.
- Estudiar la relación entre los cambios ambientales y los registros de producción de aguacate.
- Evaluar si la combinación de estas fuentes aporta información útil para estimar la producción futura.

## Zona de estudio

El estudio se centrará en una **zona de interés delimitada dentro de Michoacán, México**, vinculada al cultivo de aguacate.

Se buscará que la información satelital, las condiciones ambientales y los registros agrícolas representen un territorio comparable. El nivel de análisis —por zona, municipio o entidad— se ajustará al detalle geográfico disponible en los datos de producción.

## Datos que se prevé utilizar

| Fuente | Información prevista | Utilidad en el proyecto |
|---|---|---|
| **Sentinel-2, a través de Copernicus Data Space** | Imágenes multiespectrales de la zona de estudio. | Examinar cambios en la superficie y obtener indicadores relacionados con la vegetación y su contenido de agua. |
| **ERA5-Land, a través del Climate Data Store de Copernicus** | Registros históricos de variables como precipitación, temperatura y humedad del suelo, según su disponibilidad y pertinencia. | Complementar las imágenes con información ambiental de los mismos periodos. |
| **Estadísticas oficiales de producción agrícola, DGSIAP/SIAP** | Producción de aguacate en toneladas y, cuando estén disponibles, superficie cosechada, rendimiento y modalidad de riego o temporal. | Aportar los valores agrícolas con los que se estudiaría la relación entre las condiciones del terreno y la producción. |

ERA5-Land es un conjunto de datos de reanálisis: combina información de modelos y observaciones para describir condiciones históricas. Su función en esta propuesta sería aportar contexto ambiental; las imágenes de la superficie procederían de Sentinel-2.

## Enfoque propuesto

La información se organizaría como un historial de la zona de estudio. A partir de ese historial se analizarían patrones de cambio en la vegetación y las condiciones ambientales, y se exploraría su relación con los registros agrícolas.

Posteriormente, se buscaría estimar la evolución futura de las imágenes o de sus indicadores y utilizar esas estimaciones, junto con la información histórica pertinente, para desarrollar pronósticos de producción.

Las imágenes futuras generadas representarían **estimaciones del estado del terreno**, no nuevas observaciones realizadas por el satélite. Su utilidad tendría que comprobarse comparándolas con lo observado posteriormente.

## Periodo de análisis

Se considera **2015–2025 como periodo histórico de referencia**, sujeto a la disponibilidad, calidad y compatibilidad de las fuentes para la zona seleccionada.

La frecuencia de análisis y los horizontes de predicción se definirán conforme al detalle temporal de los datos y a los objetivos del estudio. Se buscará respetar la diferencia entre observaciones frecuentes del terreno y registros de producción que pueden estar agrupados por año o ciclo agrícola.

## Resultados que se buscan

- Un conjunto de datos integrado para estudiar la producción de aguacate en la zona seleccionada.
- Mapas y series temporales que permitan interpretar cambios en la vegetación y las condiciones ambientales.
- Estimaciones de condiciones futuras del terreno y de la producción agrícola, acompañadas de una evaluación de sus errores e incertidumbre.
- Evidencia sobre qué información resulta útil para anticipar variaciones productivas y cuáles son sus limitaciones.

La identificación de relaciones entre variables se distinguirá de la demostración de efectos causales. La interpretación de los resultados también deberá considerar factores como el manejo agrícola y la escala de los registros utilizados.

## Alcance de la propuesta

La selección final de variables, la delimitación del análisis, los periodos de estudio y los métodos se ajustarán conforme se definan los datos y las necesidades del proyecto. Las estimaciones deberán evaluarse con información posterior que no se haya utilizado para construir los modelos.

## Fuentes de referencia

- [Sentinel-2 — Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2).
- [ERA5-Land — Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land).
- [Datos abiertos de estadística de producción agrícola — DGSIAP](https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php).
- [Anuario Estadístico de la Producción Agrícola](https://nube.agricultura.gob.mx/cierre_agricola/).
