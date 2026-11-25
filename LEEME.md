# Rendimiento de aguacate — ejecución desde cero

El objetivo es estimar el **rendimiento anual de aguacate por municipio de Michoacán, en toneladas por hectárea cosechada**. Sentinel-2 aporta información del terreno y ERA5-Land aporta lluvia. Las estimaciones de TAMRF y Chronos-2 se resumen en una tabla que recibe TabICLv2 Regressor.

**2020–2025** es el periodo de desarrollo y **2026** queda reservado para evaluación externa. El piloto actual comprueba la conexión de los modelos; todavía no mide precisión ni ajusta sus pesos por gradientes.

**Punto de partida:** tienes el código del proyecto limpio y conservaste únicamente la carpeta `datos`. No se da por creado el entorno virtual, descargados los modelos ni ejecutada ninguna etapa. Se generarán nuevamente los resultados.

Esta guía cubre la instalación local y el piloto. El entorno completo usa **`requirements.txt`**, la configuración principal **`configuracion.json`** y la entrada **`ejecutar.py`**. Los notebooks utilizan los módulos compartidos. Para el histórico Sentinel desde Colab hacia S3, consulta **`GUIA_AMAZON_S3.md`**; allí se utiliza una instalación mínima que conserva el entorno GPU de Colab.

## 1. Preparar la carpeta del proyecto

1. Extrae el ZIP del proyecto limpio en una carpeta nueva.
2. Copia dentro de esa carpeta la carpeta **`datos` que conservaste**, con todo su contenido.
3. Abre esa carpeta del proyecto en VS Code con **Archivo → Abrir carpeta**.
4. Abre **Terminal → Nueva terminal**. Usa PowerShell.

En esa misma raíz deben estar `ejecutar.py`, `configuracion.json`, `requirements.txt`, `datos`, `Delimitacion_rendimiento_aguacate` y `Piloto_historico_aguacate`. Las dos últimas carpetas vienen en el ZIP y contienen código y fuentes necesarias; conserva el proyecto limpio completo.

Para comprobar que la terminal está situada correctamente:

```powershell
Get-Location
Get-ChildItem
```

En la lista debe aparecer **`ejecutar.py`**. Trabaja siempre desde esa carpeta para ejecutar los comandos de esta guía.

Usaremos la carpeta nueva directamente. No necesitas el proyecto anterior, su configuración, sus resultados ni su entorno virtual. No ejecutes el comando `actualizar`.

## 2. Crear el entorno e instalar todo

Usa **Python 3.12 de 64 bits**, con las extensiones Python y Jupyter de VS Code. La instalación incluida utiliza PyTorch CPU para Windows/Linux.

Comprueba que Python 3.12 esté disponible y crea el entorno nuevo:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
```

Si el primer comando no encuentra Python 3.12, instala esa versión antes de continuar.

Instala todas las dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Cuando termine la instalación, en VS Code pulsa **Ctrl+Shift+P → Python: Select Interpreter** y selecciona `.venv\Scripts\python.exe`. Si abres el notebook, selecciona también ese entorno como kernel.

No necesitas activar el entorno con `Activate.ps1`: los comandos llaman directamente a su Python. No necesitas instalar otros requisitos desde las subcarpetas ni desde el notebook. Todos los comandos siguientes se ejecutan desde la **raíz del proyecto**, donde están `.venv`, `ejecutar.py` y `configuracion.json`.

## 3. Conectar la carpeta datos y preparar los accesos

| Sección de `configuracion.json` | Qué controla |
|---|---|
| `carpeta_datos` | Ruta de `revision_datos_aguacate`, con `sentinel2`, `era5_land` y `agricola`. |
| `carpeta_salida`, `carpeta_modelos` | Resultados locales y pesos descargados. |
| `dispositivo`, `hilos_cpu` | Dispositivo y recursos de inferencia. |
| `sentinel2`, `era5_land` | Prueba local de ingesta con datos ya descargados. |
| `territorio` | Selección municipal y emisiones propuestas. |
| `piloto` | Municipios, años, muestras y calidad del historial pequeño. |
| `revision` | Fuentes iniciales revisadas por el notebook: fechas, AOI y malla. |

Primero revisa la estructura de la carpeta que conservaste:

```powershell
Get-ChildItem .\datos
```

Ajusta **solo `carpeta_datos`** en `configuracion.json` según esta tabla:

| Contenido que tienes | Valor de `carpeta_datos` |
|---|---|
| `datos/revision_datos_aguacate/sentinel2` y `datos/revision_datos_aguacate/era5_land` | `"datos/revision_datos_aguacate"` |
| `datos/sentinel2` y `datos/era5_land`, directamente | `"datos"` |

La ruta debe apuntar a la carpeta **que contiene** `sentinel2` y `era5_land`, no a una de esas dos subcarpetas.

Por ejemplo, para la estructura incluida en la configuración:

```json
"carpeta_datos": "datos/revision_datos_aguacate"
```

Las rutas relativas parten de la carpeta de configuración. Se respeta que tus datos estén en `dato`, `datos` u otra ubicación; modifica `carpeta_datos` cuando corresponda. En las rutas Windows del JSON utiliza `/` o `\\`.

### Accesos: Sentinel sin claves y lluvia con una clave local

**Sentinel-2 ahora se obtiene de Earth Search, de Element 84, leyendo los COG públicos de AWS.** No se necesita cuenta AWS, Client ID, Client Secret ni unidades de procesamiento de Sentinel Hub. La revisión inicial y el historial del piloto utilizan este mismo acceso.

Para ERA5-Land, abre **`credenciales.json`**, junto a `ejecutar.py`, y completa únicamente el token CDS:

```json
{
  "CDS_API_KEY": ""
}
```

Escribe el valor entre comillas y guarda con **Ctrl+S**. Si tu archivo anterior contiene `CDSE_CLIENT_ID` y `CDSE_CLIENT_SECRET`, puedes conservarlos: Sentinel ya no los lee ni los solicita. La actualización no reemplaza tu archivo de claves.

La revisión, el notebook y el piloto leen automáticamente `CDS_API_KEY`. El orden es: valor no vacío de `credenciales.json`, variable de entorno y configuración local compatible de CDS; si falta, se solicita con entrada oculta. La cuenta CDS debe tener aceptadas las condiciones de ERA5-Land.

Las claves no se añaden a planes, tablas ni informes. `credenciales.json` está excluido de Git. Mantén este archivo local fuera de los paquetes que compartas.


## 4. Ejecutar todas las etapas desde cero

Ejecuta **cada paso por separado y en este orden**. Espera a que termine antes de iniciar el siguiente. Si aparece un error, resuélvelo antes de continuar con las etapas que dependen de él.

### Paso 1. Revisar y preparar las fuentes de entrada

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py revisar-datos --fuente todas
```

Se ejecuta la revisión de Sentinel-2, ERA5-Land y agricultura y se generan nuevamente sus tablas y figuras. La carpeta `datos` se utiliza como entrada y caché. Sentinel busca también en las carpetas anteriores `escenas_completas_*` y valida fecha, bandas, cuadrícula y procedencia antes de reutilizar los TIFF. **Este comando no descarga TIFF Sentinel por defecto**: si faltan archivos verificables se detiene e informa, sin sustituir los guardados. ERA5-Land y agricultura mantienen su descarga o reutilización habitual. Solo la parte de lluvia utiliza `CDS_API_KEY` de `credenciales.json`; Sentinel no solicita claves.

Comprueba que dentro de la ruta `carpeta_datos` queden:

- `sentinel2/cortes_georreferenciados.csv` y `sentinel2/metadatos.json`.
- Los GeoTIFF `recuadro_*.tif`, dentro de la carpeta de escenas indicada en los metadatos.
- `era5_land/precipitacion_diaria_mm.nc` o `era5_land/resumen_diario_aoi.csv`.

Las imágenes PNG por sí solas no sustituyen los GeoTIFF ni el archivo de lluvia.

Al empezar Sentinel verás `Caché Sentinel: X/Y TIFF reutilizables`. Si están todos, seguirá con `Reutilizando recuadro...` y regenerará tablas y figuras leyendo tus TIFF. El diagnóstico queda en `sentinel2/diagnostico_cache.json`; los dos huecos de la malla que no intersectan escenas no cuentan como archivos faltantes.

Si estás empezando sin TIFF, o faltan algunos y quieres obtenerlos, habilita expresamente su descarga:

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py revisar-datos --fuente todas --descargar-faltantes-sentinel
```

Se descargan solamente los faltantes de la petición actual **desde Earth Search**. No hay retorno automático a Sentinel Hub. Un TIFF existente incompatible se conserva y se informa; no se sobrescribe automáticamente. No borres ni renombres las carpetas anteriores para corregir la caché.


### Paso 2. Generar el territorio y las etiquetas agrícolas

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py territorio
```

Se generan las geometrías de los municipios seleccionados, las etiquetas de rendimiento de 2020–2025 y el cruce con tu malla de recuadros. La configuración del proyecto limpio incluye una selección de 32 municipios; este paso la aplica sin necesitar archivos de resultados de otra ejecución.

Abre el mapa para revisar la selección:

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py mapa
```

Comprueba que `informe_territorio.json` indique **`cruce_recuadros.estado: "realizado"`**. Si indica `pendiente`, revisa la ruta de `carpeta_datos` y los CSV de Sentinel del paso 1 antes de seguir con el piloto.

### Paso 3. Preparar la ingesta y ejecutar TAMRF y Chronos-2

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py ingesta todo
```

Esta etapa prepara las entradas a partir de `datos`, descarga los pesos de TAMRF y Chronos-2 y ejecuta la prueba local. **No necesitas tener una carpeta de modelos previa:** `modelos_descargados` se creará durante la ejecución.

Comprueba `resultados/informe_ingesta.json`: el campo `resultado` debe ser **`"correcto"`**.

### Paso 4. Ver los resultados de esa prueba

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py ver-resultados
```

El visor genera los gráficos en `resultados/visualizaciones`. Muestra la imagen de entrada, las diez bandas estimadas y la lluvia futura. Revisa las imágenes que acaba de crear esta ejecución.

### Paso 5. Generar el plan del historial piloto

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py piloto planificar
```

Se comprueban las etiquetas, los municipios y la malla y se genera el plan de **18 casos municipio-año**. Esta acción no descarga datos ni pide credenciales. Revisa la ruta de salida que indique la terminal.

### Paso 6. Ejecutar el historial y conectar los tres modelos

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py piloto todo
```

El programa ejecuta, en orden:

1. Descarga del pequeño historial coincidente de Sentinel-2 y lluvia, de noviembre de 2020–2025.
2. Comprobación de los checkpoints y descarga de TabICLv2 y de cualquier otro peso que aún falte.
3. Inferencia de TAMRF y Chronos-2 para cada caso.
4. Construcción de la tabla y separación del contexto 2020–2024 y las consultas 2025.
5. Ejecución de TabICLv2 para devolver tres estimaciones de rendimiento anual en t/ha.

Los pesos de los dos primeros modelos, descargados en el paso 3 de **esta misma instalación**, pueden reutilizarse. La carpeta `datos` conservada no tiene por qué incluir el historial de noviembre: el piloto lo obtiene y lo guarda en `datos_piloto`.

### Paso 7. Consultar el resultado final del piloto

Abre `resultados_piloto/ultimo_plan.json` y busca la ruta de la subcarpeta activa. Dentro de ella:

- `informe_piloto.json` debe indicar **`resultado: "correcto"`**.
- `informe_tabiclv2.json` debe indicar **tres filas predichas** y que la etiqueta 2025 no se usó en `fit`.
- `predicciones_rendimiento_2025.csv` contiene las tres estimaciones técnicas en **t/ha**.

La prueba local del paso 3 y el piloto conectado del paso 6 tienen resultados distintos. El archivo de rendimiento del tercer modelo está en **`resultados_piloto`**, no en la carpeta `resultados` de la primera ingesta.

## 5. Carpetas que se crean al ejecutar

| Carpeta | Qué contiene y cuándo aparece |
|---|---|
| `.venv` | Entorno nuevo, creado en la instalación. |
| `datos` | La única carpeta conservada; contiene las entradas iniciales. |
| `modelos_descargados` | Checkpoints y código externo descargados en las pruebas. |
| `resultados` | Entradas preparadas, salidas y gráficos de la ingesta local. |
| `Delimitacion_rendimiento_aguacate/resultados_territorio` | Geometrías, etiquetas, mapas y cruce generados al ejecutar `territorio`. |
| `datos_piloto` | Historial pequeño descargado para los casos del piloto. |
| `resultados_piloto` | Plan, salidas de los tres modelos y tablas del historial piloto. |

Estas carpetas de trabajo se generarán a partir de los comandos. No necesitas recuperar pesos ni resultados de una ejecución anterior.

## 6. Referencia de comandos y resultados

La tabla siguiente reúne todos los comandos. El prefijo común es `.\.venv\Scripts\python.exe .\ejecutar.py`.

| Comando | Para qué sirve |
|---|---|
| `revisar-datos --fuente todas` | Revisión de las tres fuentes; reutiliza Sentinel local sin descargar TIFF. |
| `revisar-datos --fuente todas --descargar-faltantes-sentinel` | Revisa la caché y permite descargar solo los TIFF Sentinel faltantes. |
| `revisar-datos --fuente sentinel` | Solo Sentinel; también admite `lluvia` y `agricola`. |
| `territorio` | Etiquetas, geometría municipal y cruce con la malla local. |
| `mapa` | Abre el mapa interactivo municipal. |
| `aplicar-seleccion --archivo "RUTA_DEL_JSON"` | Importa la selección descargada del mapa. |
| `ingesta todo` | Prueba local con TAMRF y Chronos-2. |
| `ingesta preparar` | Solo prepara los archivos locales. |
| `ingesta descargar-modelos` | Verifica/descarga los modelos de esa prueba. |
| `ingesta probar` | Ejecuta ambos modelos con lo ya preparado. |
| `ver-resultados` | Genera gráficos de la prueba local guardada. |
| `piloto planificar` | Prepara el plan sin descargar ni pedir credenciales. |
| `piloto descargar` | Descarga o reutiliza las muestras históricas. |
| `piloto descargar-modelos` | Verifica/descarga los tres checkpoints. |
| `piloto dos-modelos` | Ejecuta TAMRF y Chronos-2 sobre el historial. |
| `piloto tabla` | Construye y separa contexto y consulta. |
| `piloto tercero` | Prueba TabICLv2 con la tabla. |
| `piloto todo` | Ejecuta las cinco etapas anteriores en orden. |
| `actualizar-fuentes` | Actualiza las fuentes públicas agrícolas y municipales, solo cuando se necesite. |

Los comandos de trabajo admiten `--config "RUTA_DE_CONFIGURACION"`. El selector de perfiles de **Ejecutar y depurar (F5)** ofrece las mismas etapas y utiliza `ejecutar.py`.

### Revisión inicial y notebook

Puedes abrir `Revision_fuentes_aguacate_VSCode.ipynb`, elegir el kernel `.venv` y ejecutar sus celdas. Lee `revision` del JSON y usa los mismos lectores que los scripts. La descarga de TIFF queda deshabilitada por defecto también en el notebook; para una descarga inicial añade `"DESCARGAR_FALTANTES_SENTINEL": true` dentro de `revision`. La terminal solo la habilita mediante `--descargar-faltantes-sentinel`. La ejecución de terminal utiliza ese mismo notebook, sin mantener otra copia de su procesamiento.

Se conservan las fechas originales: marzo de 2026 para revisar una adquisición Sentinel y tres días de lluvia. Esos datos de revisión no entran al contexto del piloto histórico. Cambia los parámetros en `revision` si necesitas otras fechas. La creación de un ZIP adicional de resultados queda desactivada; puedes habilitarla con `revision.CREAR_ZIP_RESULTADOS` si la necesitas.

### Earth Search, radiometría y archivos anteriores

El acceso compartido usa el catálogo STAC `https://earth-search.aws.element84.com/v1` y las colecciones `sentinel-2-c1-l2a` y `sentinel-2-l2a`. Se prefiere Collection 1 para una misma captura y se conservan las fechas que solo están en la otra colección L2A. No se cuentan distintos procesamientos de una captura como fechas adicionales. Se pagina todo el catálogo y se respetan las fechas solicitadas. No se sustituye una adquisición ausente por otro día automáticamente.

Las diez bandas conservan su orden: B02, B03, B04, B05, B06, B07, B08, B8A, B11 y B12. Cada COG se convierte usando su propio `raster:bands.scale` y `offset`; se excluye nodata antes de convertir y se limitan los valores negativos a cero, como en la reflectancia armonizada usada antes. Las bandas de 20 m y SCL se alinean a la cuadrícula de salida de 10 m mediante vecino más próximo. SCL mantiene clases enteras y `dataMask` se calcula a partir de las máscaras válidas de las bandas y SCL. Se leen ventanas de los COG, sin descargar cada archivo de escena completo.

Se conservan los TIFF, catálogos y secuencias anteriores compatibles. Si un catálogo antiguo no contiene enlaces COG, se buscan las mismas plataformas, teselas y ventana de adquisición en Earth Search. La revisión conserva el catálogo anterior para mantener su extensión y numeración de recuadros. Un cambio de proveedor no garantiza igualdad píxel a píxel entre procesamientos: los TIFF nuevos registran proveedor, escenas de origen y radiometría. La tabla georreferenciada indica la procedencia de cada TIFF de revisión.

El texto `evalscript` conservado en los metadatos sirve únicamente para reconocer las cachés antiguas. No se envía ni se ejecuta en Sentinel Hub. La dependencia `sentinelhub` se mantiene solo por sus utilidades locales de geometría del notebook.

Si el piloto se había detenido con «faltan fechas distintas», repite `python ejecutar.py piloto todo` después de actualizar el código. Las secuencias completas se reutilizan antes de consultar la red. Los catálogos anteriores se actualizan una vez con ambas colecciones, sin borrar TIFF, cambiar el terreno, ampliar el intervalo temporal ni bajar el umbral de calidad. El catálogo nuevo guarda los parámetros de consulta y sus escenas. Si siguen faltando fechas, el error indica cuántas hay y sus fechas.

El aviso informativo `boto3 not available, falling back to a DummySession` corresponde a la apertura anónima de archivos públicos por Rasterio; no indica que falten imágenes y no requiere instalar boto3. El piloto oculta esos avisos informativos, manteniendo advertencias y errores.

La descarga pública no incluye créditos de cómputo ni almacenamiento. Siguen siendo necesarios espacio local y conexión; los fallos de red o los límites del servicio se informan sin cambiar a una fuente de pago.

### Territorio y selección

La configuración del proyecto limpio incluye **32 municipios** y mantiene Tangancícuaro. Si quieres modificar esa selección, utiliza el mapa y el comando de importación que se indica a continuación.

Para modificarla: ejecuta `mapa`, elige municipios y descarga `seleccion_municipios.json`. Después:

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py aplicar-seleccion --archivo "C:\RUTA\seleccion_municipios.json"
.\.venv\Scripts\python.exe .\ejecutar.py territorio
```

El JSON descargado es una importación temporal. La configuración permanente sigue siendo `configuracion.json`.

La carpeta `Delimitacion_rendimiento_aguacate/resultados_territorio` contiene el mapa, `municipios_seleccionados.geojson`, las etiquetas `rendimiento_municipio_anio_2020_2025.csv`, el cruce `recuadros_seleccionados.csv`, `02_municipios_y_recuadros.png` y `cobertura_geometrica_municipal.csv`.

Sin tus CSV georreferenciados se generan etiquetas y mapa municipal, pero el cruce queda pendiente. La malla no se reconstruye de una captura. Para esta ejecución, utiliza el cruce y el CSV de cobertura generados al completar el paso 2.

El rendimiento es **producción total / superficie cosechada total**, después de sumar riego y temporal. Las ausencias no se rellenan con cero. La cobertura geométrica no garantiza imagen válida ni despejada, y el municipio incluye bosques, ciudades y otros cultivos. Los polígonos geoestadísticos tampoco delimitan predios ni certifican límites legales.

### Ingesta local y visualización

Esta prueba rápida usa los datos iniciales ya descargados. No sustituye al piloto multifecha. Los resultados quedan en `carpeta_salida`; el visor añade una subcarpeta en `visualizaciones`.

Sus gráficos muestran la última entrada, las diez bandas estimadas y la lluvia futura con cuantiles. El cambio entre última entrada y salida futura **no mide error**, porque corresponden a fechas distintas.

### Piloto conectado

| Elemento | Configuración conservada |
|---|---|
| Municipios | Uruapan, Tancítaro y Ario, dentro de la selección. |
| Observaciones | Noviembre 1–25 de cada año, 2020–2025. |
| TAMRF | Tres fechas reales del mismo terreno: `[1, 3, 10, 64, 64]`; diez bandas estimadas del 30 de noviembre: `[1, 1, 10, 64, 64]`. |
| Chronos-2 | 25 días por municipio-año; cinco días futuros, noviembre 26–30. |
| TabICLv2, contexto | 15 filas de 2020–2024, 29 características y etiquetas reales. |
| TabICLv2, consulta | Tres filas de 2025, sin entregar sus etiquetas al modelo. |

Las características son medias y desviaciones de diez bandas estimadas, NDVI/NDMI medios, cuatro resúmenes de lluvia y tres indicadores municipales. Contexto y consulta se construyen con el mismo proceso de predicción. Producción, superficie final, año, identificador y etiqueta no entran automáticamente a X.

Cada muestra Sentinel es una **ventana fija de 64×64 píxeles a 10 m**, no una imagen completa reducida a 64. Requiere tres fechas distintas sin nodata y al menos 80 % de clases claras. Si faltan fechas válidas, el programa informa y se detiene. Una muestra geométrica no identifica una huerta ni representa el municipio completo.

La lluvia se solicita sobre los municipios completos a su resolución nativa de 0.1°. El acumulado original 00 UTC del día siguiente se convierte de metros a mm y se desplaza al día correcto **una sola vez**. Se calcula una media municipal por áreas de intersección con celdas completas, en vez de utilizar la misma serie del AOI pequeño para todos.

Las descargas tienen caché y los pesos se verifican por SHA256. No hay sustitución por datos simulados o pesos aleatorios al fallar una descarga. CDS puede dejar solicitudes en cola. Repite `piloto todo` o ejecuta la etapa pendiente para reanudar.

`resultados_piloto/ultimo_plan.json` señala la subcarpeta activa. Allí encontrarás:

| Archivo | Qué revisar |
|---|---|
| `casos_piloto.csv` | Lugares, años y cortes del plan. |
| `tabla_contexto_2020_2024.csv` | Características y etiquetas del contexto. |
| `tabla_consulta_2025.csv` | Consultas sin etiqueta objetivo. |
| `etiquetas_consulta_reservadas.csv` | Etiquetas reales separadas para comparación posterior. |
| `predicciones_rendimiento_2025.csv` | Estimaciones técnicas en t/ha. |
| `informe_tabiclv2.json` | Checkpoint, formas y separación temporal. |
| `informe_piloto.json` | Estado correcto/error de cada etapa. |

Una respuesta finita comprueba la conexión, no la precisión agronómica. La etiqueta es anual, aunque aquí se usan muestras pequeñas y cinco días pronosticados. La evaluación posterior ampliará y comprobará muestreo, ventanas y características.

## 7. Organización y comprobaciones

| Componente | Responsabilidad |
|---|---|
| `ejecutar.py` | Entrada principal. |
| `utilidades.py` | Rutas, JSON, firmas, lectura de claves locales y proyección compartidas. |
| `credenciales.json` | Claves de acceso que completas una sola vez; archivo local excluido de Git. |
| `datos_sentinel.py`, `datos_era5.py`, `datos_agricolas.py` | Earth Search público, interpretación de lluvia y agricultura, compartidos por las interfaces. |
| `cache_sentinel.py` | Identifica y valida los TIFF anteriores y controla los faltantes. |
| `preparacion.py` | Adaptación de archivos locales. |
| `recursos_modelos.py`, `modelos.py` | Descarga, verificación y pases de inferencia compartidos. |
| `Delimitacion_rendimiento_aguacate` | Municipios, etiquetas y mapas. |
| `Piloto_historico_aguacate` | Historial, tabla y conexión del tercer modelo. |
| `revision_datos.py`, notebook | Interfaz terminal y visual del mismo procesamiento. |
| `ver_resultados.py` | Gráficos guardados. |
| `tests` | Verificación de preparación; no se necesita para la ejecución normal. |

Las carpetas están separadas por su función. La instalación se hace con `requirements.txt`, los ajustes con `configuracion.json` y las etapas se ejecutan mediante `ejecutar.py`.

Las pruebas verifican bandas, malla, fechas, nodata, unidades y reutilización de cachés anteriores:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La migración a Earth Search se comprobó con lecturas reales de Michoacán de 2020 y 2025: diez bandas, SCL y dataMask en ventanas de 64×64. También se ejecutó el adaptador real del piloto para una secuencia de tres fechas de noviembre de 2020 y se verificó que la segunda ejecución reutiliza los archivos sin red. Las pruebas automáticas verifican escala/offset, nodata, vecino más próximo, mosaico, paginación y conservación de cachés. Se comprobó también la recuperación del caso Uruapan 2022 desde un catálogo vacío y una secuencia real de tres fechas, además de la disponibilidad de al menos tres fechas en los 18 catálogos municipio-año. La disponibilidad en catálogo no sustituye la evaluación de calidad por píxel. Estas pruebas técnicas no miden precisión agronómica.

La corrección de caché se comprobó con TIFF pequeños de prueba y el procesamiento real del notebook: reutilización completa sin descargas, detención ante faltantes y una sola solicitud al habilitar el único faltante. No fue necesario acceder a tus TIFF ni a tus cuentas.

La cadena se comprobó en CPU con checkpoints oficiales, imágenes y lluvia simuladas para verificar el software y etiquetas DGSIAP reales. No se presentan esas entradas como tus observaciones satelitales. El historial Sentinel se obtiene públicamente de Earth Search; únicamente la descarga de lluvia requiere tu cuenta CDS.

El piloto es retrospectivo con pesos actuales: no demuestra que cada reanálisis, publicación o checkpoint estuviera disponible en aquellos años. Las etiquetas definitivas 2025 se publicaron en 2026 y no se entregan como contexto de su consulta 2025. La evaluación operativa deberá incorporar las fechas reales de disponibilidad y el cierre anual 2026 cuando esté publicado.

Se conserva el orden acordado: definir el problema; comprobar la ingesta histórica; conectar TabICLv2; evaluar los pesos actuales; ajustar solo lo que necesite mejorar.

## Fuentes técnicas

- TAMRF: https://github.com/Evoland-Land-Monitoring-Evolution/tamrfsits — pesos: https://zenodo.org/records/17474541
- Chronos-2: https://huggingface.co/amazon/chronos-2
- TabICLv2: https://github.com/soda-inria/tabicl — API: https://tabicl.readthedocs.io/en/latest/api.html
- Sentinel-2 en Earth Search: https://github.com/Element84/earth-search/blob/main/docs/collections/sentinel-2-l2a.md
- COG públicos: https://registry.opendata.aws/sentinel-2-l2a-cogs/
- ERA5-Land: https://confluence.ecmwf.int/pages/viewpage.action?pageId=143061770
- INEGI: https://www.inegi.org.mx/servicios/catalogounico.html
- DGSIAP: https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php
- PyTorch CPU: https://docs.pytorch.org/get-started/previous-versions/


## 8. Histórico Sentinel en S3 y filtro de cobertura

La nueva guía **GUIA_AMAZON_S3.md** explica la creación de la cuenta y del bucket, el filtro mínimo del 85 % de cobertura municipal, y la ingesta desde Colab sin guardar TIFF en el disco del equipo. Abre **Sentinel_a_S3_Colab.ipynb** en Colab.

La selección usa la unión de los municipios, no huertas. El campo `territorio.fraccion_minima_aoi` controla el umbral; 0.85 acepta también recuadros con 90–100 %. `territorio` y `sentinel-s3 preparar` usan el mismo cálculo y conservan la numeración original.

Los comandos nuevos son `sentinel-s3 preparar`, `sentinel-s3 planificar` y `sentinel-s3 transferir`. El primero solo necesita geometrías y el CSV de malla; el segundo consulta metadatos; el tercero lee las ventanas nativas en RAM y las guarda como COG en S3. Ejecuta la transferencia desde Colab o un servidor si no quieres que los datos pasen por tu computadora.

En Colab instala únicamente `requirements-sentinel-s3.txt`; el `requirements.txt` completo corresponde al entorno CPU de VS Code y no se debe instalar sobre el entorno GPU de Colab para esta tarea. Esta entrega prepara datos y almacenamiento; no entrena aún nuevas capas.

La selección inicial de **49 recuadros rojos** del mapa está fijada en `territorio.cortes_preseleccionados` de `configuracion.json`. El filtro del 85 % solo puede excluir elementos de esa lista, nunca añadir recuadros ajenos. Los números se conservan; las coordenadas se leen del CSV original de la malla. Si cambias la lista, repite preparar y planificar.
