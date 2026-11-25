"""Lectura y conversión de ERA5-Land compartidas; unidades y calendario se corrigen una vez."""

from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from utilidades import credencial_configurada, leer_credencial

DATASET_ERA5 = "reanalysis-era5-land"


def crear_cliente_cds():
    import cdsapi

    token = credencial_configurada("CDS_API_KEY")
    opciones = dict(quiet=True, debug=False, timeout=60, retry_max=2)
    if token:
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api", key=token, **opciones
        )
    try:
        return cdsapi.Client(**opciones)
    except Exception:
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=leer_credencial("CDS_API_KEY"),
            **opciones,
        )


def abrir_netcdf(ruta):
    ruta = Path(ruta)
    if zipfile.is_zipfile(ruta):
        with zipfile.ZipFile(ruta) as z:
            miembros = [
                n
                for n in z.namelist()
                if n.lower().endswith((".nc", ".nc4", ".netcdf"))
            ]
            if len(miembros) != 1:
                raise ValueError("Se esperaba un NetCDF en la respuesta CDS.")
            destino = ruta.with_suffix(".extraido.nc")
            with z.open(miembros[0]) as entrada, destino.open("wb") as salida:
                shutil.copyfileobj(entrada, salida)
        ruta = destino
    with xr.open_dataset(ruta, engine="netcdf4") as ds:
        return ds.load()


def interpretar_precipitacion(ds):
    nombres = [v for v in ["tp", "total_precipitation"] if v in ds.data_vars]
    if len(nombres) != 1:
        raise ValueError("No se identifica la variable de lluvia ERA5-Land.")
    tp = ds[nombres[0]]
    if "valid_time" in tp.dims:
        tp = tp.rename({"valid_time": "time"})
    if not {"time", "latitude", "longitude"} <= set(tp.dims):
        raise ValueError("Dimensiones ERA5-Land inesperadas.")
    for dim in set(tp.dims) - {"time", "latitude", "longitude"}:
        if tp.sizes[dim] != 1:
            raise ValueError(f"Dimensión adicional no resuelta: {dim}.")
        tp = tp.isel({dim: 0}, drop=True)
    if str(tp.attrs.get("units", "")).strip().lower() not in [
        "m",
        "metres",
        "meters",
        "m of water equivalent",
    ]:
        raise ValueError(
            "El archivo ERA5-Land original debe estar en metros; no se reconvierten archivos ya expresados en mm."
        )
    fechas = pd.DatetimeIndex(tp.time.values)
    if fechas.has_duplicates or not fechas.equals(fechas.normalize()):
        raise ValueError("Se requieren acumulados a las 00 UTC, sin duplicados.")
    if not np.isfinite(tp.values).any() or float(tp.min(skipna=True)) < -1e-6:
        raise ValueError("Precipitación ausente o negativa fuera de tolerancia.")
    diario = (tp.clip(min=0) * 1000).assign_coords(
        time=tp.time - np.timedelta64(1, "D")
    )
    diario = (
        diario.assign_coords(longitude=(diario.longitude + 180) % 360 - 180)
        .sortby("longitude")
        .sortby("latitude")
        .sortby("time")
    )
    for coord in ["latitude", "longitude"]:
        valores = np.asarray(diario[coord].values, dtype=float)
        if len(valores) > 1 and not np.allclose(
            np.diff(valores), 0.1, rtol=0, atol=1e-4
        ):
            raise ValueError(
                "Se esperaba la malla ERA5-Land nativa de 0.1 grados, sin remuestreo."
            )
    # Nombre del contrato compartido por tablas, NetCDF y preparación de Chronos-2.
    diario = diario.transpose("time", "latitude", "longitude").rename(
        "precipitacion_diaria_mm"
    )
    diario.attrs = {
        "units": "mm",
        "convencion": "00 UTC del día siguiente representa el día etiquetado",
    }
    return diario
