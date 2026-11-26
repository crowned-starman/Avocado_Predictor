## 1. Propósito

Proponer una interfaz web clara e intuitiva que permita consultar estimaciones de **producción de aguacate en toneladas** mediante un filtro por **municipio** (y superficie en hectáreas), eligiendo entre consulta **por año** o por **temporadas ya determinadas**, y presentando tanto los datos principales como una **descripción en lenguaje claro generada por un LLM**.

Esta propuesta está diseñada para avanzar en la definición visual y de flujo mientras se completa la integración con el modelo y los servicios del proyecto.

## 2. Alcance y definiciones del sistema

- **Selección territorial:** filtro por **municipio** (sin selección libre ni dibujo de polígonos en mapa); el usuario podrá indicar las **hectáreas** de referencia para su consulta.
- **Horizontes de consulta:** selección **por año** o entre **temporadas predefinidas** establecidas por el proyecto.
- **Narrativa para el usuario:** integración de un **LLM** para redactar descripciones comprensibles a partir de los resultados y variables del periodo consultado.
- **Estado actual de datos:** durante la fase de interfaz/propuesta se utilizan **datos simulados (*mocks*)** para validar flujos, estados visuales y presentación de información antes de conectar el modelo real.

## 3. Flujo principal del usuario

1. **Elegir ubicación y superficie:** seleccionar el **municipio** e ingresar el número de **hectáreas**.
2. **Elegir tipo de predicción:** alternar entre consulta **por año** o **temporada predefinida**, y seleccionar el valor correspondiente.
3. **Consultar resultados:** visualizar la estimación de producción en **toneladas** (y referencia por hectárea), indicadores de contexto y la **descripción generada por el LLM**.

## 4. Estructura visual propuesta

### A. Panel de parámetros de consulta

- **Selector de municipio:** lista desplegable o selector de municipios participantes.
- **Campo de hectáreas:** entrada numérica para ajustar la superficie de interés.
- **Selector de periodo:**
  - Opción **Por año:** selector del año a consultar.
  - Opción **Por temporada:** selector con las **temporadas ya determinadas**.
- **Botón de consulta:** dispara la obtención de la estimación (simulada o conectada al modelo).

### B. Panel de resultados

- **Tarjeta principal de producción:** muestra la estimación en **toneladas de aguacate** y su equivalencia/referencia por hectárea, junto con un indicador de rango o confianza cuando aplique.
- **Indicadores de contexto:** tarjetas o resumen visual con variables de apoyo vinculadas al enfoque del proyecto (indicadores de vegetación, condiciones climáticas y referencia agrícola del periodo).
- **Bloque de descripción para el usuario (LLM):** espacio dedicado a mostrar el texto explicativo redactado por el LLM para facilitar la interpretación del resultado, con una etiqueta visible que indique si el contenido proviene de simulación o del sistema conectado.

## 5. Estados de la interfaz

- **Estado inicial / vacío:** invita al usuario a elegir municipio, hectáreas y periodo antes de consultar.
- **Estado de carga:** muestra indicadores visuales de espera mientras se procesa la consulta.
- **Estado con resultados:** despliega las toneladas estimadas, indicadores y la descripción del LLM junto con la etiqueta de origen del dato (*simulado* o *modelo*).
- **Estado de error o sin datos:** informa con claridad si ocurrió un fallo o si no hay información disponible para la combinación elegida.

## 6. Ejemplo de datos simulados (contrato preliminar de UI)

```json
{
  "consulta": {
    "municipio": "Uruapan",
    "hectareas": 10,
    "modo": "temporada",
    "anio": 2026,
    "temporada": "Temporada 1 (predefinida)"
  },
  "resultado": {
    "produccion_estimada_toneladas": 115.0,
    "toneladas_por_hectarea": 11.5,
    "rango_estimado_toneladas": [102.0, 128.0],
    "indicadores_contexto": {
      "vigor_vegetacion": "Alto",
      "humedad_suelo": "Moderada-Alta",
      "precipitacion_mm": 395
    },
    "descripcion_llm": "Para la consulta en Uruapan (10 ha) durante la temporada seleccionada, se estima una producción cercana a 115 toneladas, respaldada por un vigor vegetal alto y condiciones de humedad favorables en el periodo.",
    "origen_dato": "simulacion_propuesta_ui"
  }
}
