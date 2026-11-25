"""Comprueba contratos de datos; las muestras se crean en directorios temporales."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preparacion import (
    BANDAS,
    preparar_era5,
    preparar_grupo_sentinel,
    validar_calendario,
)
from prueba_ingesta import cargar_configuracion


class PreparacionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "configuracion.json").read_text()
        )
        cfg["carpeta_datos"] = str(self.raiz / "datos")
        cfg["carpeta_salida"] = str(self.raiz / "salida")
        cfg["carpeta_modelos"] = str(self.raiz / "modelos")
        cfg["sentinel2"]["tamanio_parche_px"] = 32
        ruta = self.raiz / "configuracion.json"
        ruta.write_text(json.dumps(cfg))
        self.cfg = cargar_configuracion(ruta)

    def tearDown(self):
        self.tmp.cleanup()

    def tif(
        self,
        nombre="a.tif",
        fecha="2026-03-19",
        sin_datos=False,
        desplazado=False,
        escala=1,
    ):
        ruta = self.raiz / nombre
        nombres = BANDAS[::-1] + ["SCL", "dataMask"]  # Orden intencionalmente distinto.
        datos = np.zeros((12, 96, 96), dtype=np.float32)
        for i, n in enumerate(nombres[:10]):
            datos[i] = ((BANDAS.index(n) + 1) * 0.02) * escala
        datos[10] = 4
        datos[10, :16, :16] = 9
        datos[11] = 0 if sin_datos else 1
        datos[11, :8, :] = 0
        with rasterio.open(
            ruta,
            "w",
            driver="GTiff",
            width=96,
            height=96,
            count=12,
            dtype="float32",
            crs="EPSG:32613",
            transform=from_origin(600010 if desplazado else 600000, 2150000, 10, 10),
        ) as dst:
            dst.write(datos)
            for i, n in enumerate(nombres, 1):
                dst.set_band_description(i, n)
            dst.update_tags(fecha_utc=fecha)
        return ruta

    def lluvia(self, valores=None, units="mm", variable="precipitacion_diaria_mm"):
        ruta = (
            Path(self.cfg["carpeta_datos"]) / "era5_land" / "precipitacion_diaria_mm.nc"
        )
        ruta.parent.mkdir(parents=True, exist_ok=True)
        if valores is None:
            valores = np.array(
                [
                    [[1.0, np.nan], [3.0, np.nan]],
                    [[2.0, np.nan], [4.0, np.nan]],
                    [[3.0, np.nan], [5.0, np.nan]],
                ]
            )
        xr.DataArray(
            valores,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": pd.date_range("2026-03-01", periods=3),
                "latitude": [19.0, 20.0],
                "longitude": [-102.0, -101.0],
            },
            name=variable,
            attrs={"units": units},
        ).to_dataset().to_netcdf(ruta)
        return ruta

    def test_orden_real_de_bandas_y_fechas_sin_duplicarlas(self):
        p = self.tif()
        info = preparar_grupo_sentinel(self.cfg, [p])
        self.assertEqual(info["forma"], [1, 1, 10, 32, 32])
        self.assertEqual(info["consulta_utc"], "2026-03-24")
        with np.load(
            Path(self.cfg["carpeta_salida"]) / "preparados" / "sentinel2_entrada.npz"
        ) as z:
            np.testing.assert_allclose(
                z["reflectancia"][0, 0, :, 0, 0], np.arange(1, 11) * 0.02, rtol=1e-6
            )
            self.assertEqual(z["dias_desde_origen"].tolist(), [[0.0]])
            self.assertEqual(z["dias_consulta"].tolist(), [[5.0]])

    def test_alineacion_espacial_y_dos_fechas_reales(self):
        a = self.tif()
        b = self.tif("b.tif", fecha="2026-03-24", desplazado=True)
        info = preparar_grupo_sentinel(self.cfg, [b, a])
        self.assertEqual(info["forma"][1], 2)
        self.assertEqual(info["alineacion_aplicada"], [False, True])
        self.assertEqual(info["consulta_utc"], "2026-03-29")

    def test_rechaza_fechas_repetidas(self):
        with self.assertRaisesRegex(ValueError, "misma fecha"):
            preparar_grupo_sentinel(self.cfg, [self.tif(), self.tif("b.tif")])

    def test_rechaza_nodata_y_escala_dn(self):
        with self.assertRaisesRegex(ValueError, "nodata"):
            preparar_grupo_sentinel(self.cfg, [self.tif(sin_datos=True)])
        with self.assertRaisesRegex(ValueError, "DN"):
            preparar_grupo_sentinel(self.cfg, [self.tif("dn.tif", escala=10000)])

    def test_lluvia_conserva_mm_y_fecha_sin_doble_conversion(self):
        self.lluvia()
        info = preparar_era5(self.cfg)
        self.assertEqual(info["forma_contexto"], [1, 1, 3])
        self.assertEqual(info["fecha_inicial_utc"], "2026-03-01")
        tabla = pd.read_csv(
            Path(self.cfg["carpeta_salida"]) / "preparados" / "chronos2_entrada.csv"
        )
        pesos = np.cos(np.deg2rad([19.0, 20.0]))
        esperado = np.average(np.array([[1, 2, 3], [3, 4, 5]]), axis=0, weights=pesos)
        np.testing.assert_allclose(tabla.target, esperado, rtol=1e-6)

    def test_celdas_incompletas_no_se_vuelven_lluvia_cero(self):
        v = np.array(
            [
                [[1.0, 10.0], [np.nan, np.nan]],
                [[2.0, np.nan], [np.nan, np.nan]],
                [[3.0, 30.0], [np.nan, np.nan]],
            ]
        )
        self.lluvia(v)
        info = preparar_era5(self.cfg)
        self.assertEqual(info["celdas_incompletas_descartadas"], 1)
        tabla = pd.read_csv(
            Path(self.cfg["carpeta_salida"]) / "preparados" / "chronos2_entrada.csv"
        )
        np.testing.assert_allclose(tabla.target, [1, 2, 3])

    def test_rechaza_tp_crudo_y_unidades_incorrectas(self):
        self.lluvia(variable="tp", units="m")
        with self.assertRaisesRegex(ValueError, "YA interpretado"):
            preparar_era5(self.cfg)
        self.lluvia(units="m")
        with self.assertRaisesRegex(ValueError, "mm"):
            preparar_era5(self.cfg)

    def test_rechaza_huecos_temporales(self):
        with self.assertRaisesRegex(ValueError, "Faltan días"):
            validar_calendario(["2026-03-01", "2026-03-02", "2026-03-04"])

    def test_alternativa_csv_y_series_por_celda(self):
        ruta = Path(self.cfg["carpeta_datos"]) / "era5_land" / "resumen_diario_aoi.csv"
        ruta.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "fecha_utc": pd.date_range("2026-03-01", periods=3),
                "precipitacion_media_aoi_mm": [0.0, 2.0, 4.0],
            }
        ).to_csv(ruta, index=False)
        info = preparar_era5(self.cfg)
        self.assertEqual(info["fuente"], "csv_diario_mm")
        self.lluvia()
        self.cfg["era5_land"]["modo_series"] = "por_celda"
        info = preparar_era5(self.cfg)
        self.assertEqual(info["forma_contexto"], [2, 1, 3])


if __name__ == "__main__":
    unittest.main()
