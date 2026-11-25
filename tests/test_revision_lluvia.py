"""Revisión completa desde la caché CDS hasta el archivo que recibe la ingesta."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from datos_era5 import interpretar_precipitacion
from preparacion import preparar_era5
from revision_datos import ejecutar_revision
from utilidades import cargar_configuracion


class RevisionLluviaTest(unittest.TestCase):
    @staticmethod
    def respuesta(nombre="tp"):
        valores = np.broadcast_to(
            np.asarray([0.001, 0.002, 0.003], dtype=np.float32)[:, None, None],
            (3, 6, 8),
        ).copy()
        return xr.Dataset(
            {
                nombre: (
                    ("valid_time", "latitude", "longitude"),
                    valores,
                    {"units": "m"},
                )
            },
            coords={
                "valid_time": pd.date_range("2020-03-02", periods=3),
                "latitude": np.linspace(19.7, 19.2, 6),
                "longitude": np.linspace(-102.3, -101.6, 8),
                "expver": ("valid_time", ["0001"] * 3),
                "number": 0,
            },
        )

    def test_nombre_unidades_y_fechas_de_ambas_variables_de_origen(self):
        for nombre in ["tp", "total_precipitation"]:
            with self.subTest(nombre=nombre):
                diario = interpretar_precipitacion(self.respuesta(nombre))
                self.assertEqual(diario.name, "precipitacion_diaria_mm")
                self.assertEqual(diario.attrs["units"], "mm")
                self.assertEqual(
                    pd.DatetimeIndex(diario.time.values).strftime("%Y-%m-%d").tolist(),
                    ["2020-03-01", "2020-03-02", "2020-03-03"],
                )
                np.testing.assert_allclose(diario.values[:, 0, 0], [1, 2, 3])
                self.assertIn("precipitacion_diaria_mm", diario.to_dataframe())

    def test_revision_desde_cache_exporta_figuras_tablas_y_netcdf_compatible(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            proyecto = Path(__file__).resolve().parents[1]
            cfg = json.loads((proyecto / "configuracion.json").read_text())
            cfg.update(
                carpeta_datos=str(raiz / "datos"),
                carpeta_salida=str(raiz / "resultados"),
                carpeta_modelos=str(raiz / "modelos"),
            )
            cfg["revision"]["ERA5_FECHA_INICIAL"] = "2020-03-01"
            cfg["revision"]["CREAR_ZIP_RESULTADOS"] = False
            archivo_cfg = raiz / "configuracion.json"
            archivo_cfg.write_text(json.dumps(cfg))
            directorio = raiz / "datos/era5_land"
            directorio.mkdir(parents=True)
            peticion = {
                "variable": ["total_precipitation"],
                "year": ["2020"],
                "month": ["03"],
                "day": ["02", "03", "04"],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [19.7, -102.3, 19.2, -101.6],
            }
            firma = hashlib.sha256(
                json.dumps(peticion, sort_keys=True).encode()
            ).hexdigest()[:12]
            original = directorio / f"precipitacion_{firma}.nc"
            self.respuesta().to_netcdf(original, engine="netcdf4")
            antes = original.read_bytes()
            with patch(
                "datos_era5.crear_cliente_cds",
                side_effect=AssertionError("CDS no necesario"),
            ):
                with patch(
                    "datos_sentinel.buscar_escenas",
                    side_effect=AssertionError("Sentinel no necesario"),
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        ejecutar_revision(archivo_cfg, fuente="lluvia")
            self.assertEqual(original.read_bytes(), antes)
            for nombre in [
                "01_mapas_tres_dias.png",
                "02_serie_diaria.png",
                "resumen_diario_aoi.csv",
                "precipitacion_por_celda.csv",
                "precipitacion_diaria_mm.nc",
                "metadatos.json",
            ]:
                self.assertTrue((directorio / nombre).is_file(), nombre)
            tabla = pd.read_csv(directorio / "precipitacion_por_celda.csv")
            self.assertIn("precipitacion_diaria_mm", tabla)
            self.assertEqual(
                len(tabla), 72
            )  # 24 celdas del AOI por cada uno de tres días.
            with xr.open_dataset(directorio / "precipitacion_diaria_mm.nc") as ds:
                self.assertIn("precipitacion_diaria_mm", ds)
                self.assertEqual(ds.precipitacion_diaria_mm.attrs["units"], "mm")
            info = preparar_era5(cargar_configuracion(archivo_cfg))
            self.assertEqual(info["dias_historicos"], 3)
            entrada = pd.read_csv(raiz / "resultados/preparados/chronos2_entrada.csv")
            np.testing.assert_allclose(entrada.target, [1, 2, 3])
            self.assertEqual(
                entrada.timestamp.tolist(), ["2020-03-01", "2020-03-02", "2020-03-03"]
            )


if __name__ == "__main__":
    unittest.main()
