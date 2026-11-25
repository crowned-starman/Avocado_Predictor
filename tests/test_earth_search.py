"""Calibración, cuadrícula, mosaico y caché de Earth Search sin red en las pruebas."""

import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from datos_sentinel import (
    ALL_DOWNLOAD_BANDS,
    ASSETS,
    EARTH_SEARCH_COLECCIONES,
    EARTH_SEARCH_URL,
    buscar_escenas,
    leer_escenas,
    resolver_escenas_earth,
)
from Piloto_historico_aguacate.descargas_piloto import descargar_sentinel
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from utilidades import guardar_json


class EarthSearchTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.fecha = dt.datetime(2020, 11, 1, 17, 28, tzinfo=dt.timezone.utc)
        self.bbox = [100000, 2099960, 100040, 2100000]
        self.item = {
            "id": "S2A_T13QGB_20201101T171700_L2A",
            "geometry": mapping(box(-180, -90, 180, 90)),
            "properties": {"datetime": self.fecha.isoformat(), "eo:cloud_cover": 1},
            "assets": {},
        }
        for i, banda in enumerate(ALL_DOWNLOAD_BANDS[:-1]):
            resolucion = 10 if banda in ["B02", "B03", "B04", "B08"] else 20
            lado = 40 // resolucion
            archivo = self.raiz / f"{banda}.tif"
            valor = 4 if banda == "SCL" else 2000 + i * 100
            datos = np.full((lado, lado), valor, dtype=np.uint16)
            with rasterio.open(
                archivo,
                "w",
                driver="GTiff",
                height=lado,
                width=lado,
                count=1,
                dtype="uint16",
                crs="EPSG:32613",
                nodata=0,
                transform=from_origin(100000, 2100000, resolucion, resolucion),
            ) as dst:
                dst.write(datos, 1)
            meta = {"nodata": 0}
            if banda != "SCL":
                meta.update(scale=0.0001, offset=-0.1)
            self.item["assets"][ASSETS[banda]] = {
                "href": str(archivo),
                "raster:bands": [meta],
            }

    def leer(self, items=None, bbox=None):
        return leer_escenas(
            items or [self.item], bbox or self.bbox, "EPSG:32613", (4, 4)
        )

    def test_calibra_una_vez_y_conserva_orden_y_clases(self):
        a = self.leer()
        self.assertEqual(a.shape, (4, 4, 12))
        self.assertEqual(a.dtype, np.float32)
        np.testing.assert_allclose(
            a[0, 0, :10], [0.1 + i * 0.01 for i in range(10)], atol=1e-7
        )
        np.testing.assert_array_equal(a[..., 10], 4)
        np.testing.assert_array_equal(a[..., 11], 1)

    def test_valor_negativo_valido_se_armoniza_sin_volverse_nodata(self):
        with rasterio.open(self.raiz / "B02.tif", "r+") as dst:
            a = dst.read(1)
            a[0, 0], a[0, 1] = 500, 0
            dst.write(a, 1)
        a = self.leer()
        self.assertEqual(a[0, 0, 0], 0)
        self.assertEqual(a[0, 0, 11], 1)
        self.assertEqual(a[0, 1, 11], 0)
        self.assertTrue((a[0, 1] == 0).all())

    def test_banda_20m_y_scl_se_remuestrean_por_vecino(self):
        for nombre, valores in [
            ("B05", [[2000, 3000], [4000, 5000]]),
            ("SCL", [[4, 5], [6, 9]]),
        ]:
            with rasterio.open(self.raiz / (nombre + ".tif"), "r+") as dst:
                dst.write(np.asarray(valores, dtype=np.uint16), 1)
        a = self.leer()
        np.testing.assert_allclose(
            a[..., 3],
            np.repeat(np.repeat([[0.1, 0.2], [0.3, 0.4]], 2, 0), 2, 1),
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            a[..., 10], np.repeat(np.repeat([[4, 5], [6, 9]], 2, 0), 2, 1)
        )

    def test_fuera_del_raster_es_nodata(self):
        a = self.leer(bbox=[100020, 2099960, 100060, 2100000])
        self.assertTrue((a[:, 2:, :] == 0).all())
        self.assertTrue((a[:, :2, 11] == 1).all())

    def test_no_supone_calibracion_ni_acepta_clases_invalidas(self):
        item = copy.deepcopy(self.item)
        del item["assets"]["blue"]["raster:bands"][0]["offset"]
        with self.assertRaisesRegex(ValueError, "scale/offset"):
            self.leer([item])
        with rasterio.open(self.raiz / "SCL.tif", "r+") as dst:
            dst.write(np.full((2, 2), 100, dtype=np.uint16), 1)
        with self.assertRaisesRegex(ValueError, "SCL inesperadas"):
            self.leer()

    def test_no_mezcla_otro_dia(self):
        item = copy.deepcopy(self.item)
        item["properties"]["datetime"] = "2020-11-02T17:28:00Z"
        with self.assertRaisesRegex(ValueError, "adquisiciones"):
            self.leer([self.item, item])

    def test_mosaico_con_prioridad_y_sin_mezclar_bandas_de_escenas(self):
        segundo = copy.deepcopy(self.item)
        segundo["id"] += "_otra"
        segundo["properties"]["eo:cloud_cover"] = 50
        # Crear otro juego de bandas; debe llenar solo el píxel faltante del primero.
        for b in ALL_DOWNLOAD_BANDS[:-1]:
            asset = segundo["assets"][ASSETS[b]]
            archivo = self.raiz / ("otro_" + b + ".tif")
            with rasterio.open(asset["href"]) as src:
                datos, perfil = src.read(1), src.profile
            perfil.pop("blockxsize", None)
            perfil.pop("blockysize", None)
            datos[:] = 5 if b == "SCL" else 5000
            with rasterio.open(archivo, "w", **perfil) as dst:
                dst.write(datos, 1)
            asset["href"] = str(archivo)
        with rasterio.open(self.raiz / "B02.tif", "r+") as dst:
            datos = dst.read(1)
            datos[0, 0] = 0
            dst.write(datos, 1)
        a = self.leer([segundo, self.item])
        np.testing.assert_allclose(a[0, 0, :10], 0.4, atol=1e-7)
        self.assertEqual(a[0, 0, 10], 5)
        self.assertAlmostEqual(float(a[1, 1, 0]), 0.1)
        self.assertEqual(a[1, 1, 10], 4)

    def test_pagina_y_elimina_duplicados_sin_autenticacion(self):
        segundo = copy.deepcopy(self.item)
        segundo["id"] += "_2"
        segundo["properties"]["datetime"] = "2020-11-06T17:28:00Z"
        siguiente = EARTH_SEARCH_URL + "/search?token=pagina2"
        r1 = Mock(url=EARTH_SEARCH_URL + "/search?primera")
        r1.json.return_value = {
            "features": [self.item],
            "links": [{"rel": "next", "href": siguiente}],
        }
        r2 = Mock(url=siguiente)
        r2.json.return_value = {"features": [self.item, segundo], "links": []}
        cliente = Mock()
        cliente.get.side_effect = [r1, r2]
        with patch("datos_sentinel._cliente", return_value=cliente):
            items = buscar_escenas([-103, 19, -102, 20], ["2020-11-01", "2020-11-25"])
        self.assertEqual(len(items), 2)
        self.assertEqual(cliente.get.call_count, 2)
        self.assertEqual(
            cliente.get.call_args_list[0].kwargs["params"]["collections"],
            ",".join(EARTH_SEARCH_COLECCIONES),
        )
        for llamada in cliente.get.call_args_list:
            self.assertTrue(llamada.args[0].startswith(EARTH_SEARCH_URL))
            self.assertNotIn("auth", llamada.kwargs)

    def test_resuelve_catalogo_anterior_y_reutiliza_la_consulta(self):
        anterior = copy.deepcopy(self.item)
        anterior["assets"] = {}
        anterior["id"] = "S2A_MSIL2A_20201101T171700_N0500_R112_T13QGB_20201101T200000"
        with patch(
            "datos_sentinel.buscar_escenas", return_value=[self.item]
        ) as consulta:
            for _ in range(2):
                items = resolver_escenas_earth(
                    [anterior], [-103, 19, -102, 20], self.fecha, self.raiz
                )
                self.assertEqual(items[0]["id"], self.item["id"])
        self.assertEqual(consulta.call_count, 1)
        otra = copy.deepcopy(self.item)
        otra["id"] = otra["id"].replace("S2A_", "S2B_")
        with patch("datos_sentinel.buscar_escenas", return_value=[otra]):
            with self.assertRaisesRegex(ValueError, "no contiene todas"):
                resolver_escenas_earth(
                    [anterior], [-103.1, 19, -102, 20], self.fecha, self.raiz
                )

    def test_prefiere_collection1_y_conserva_fechas_exclusivas_de_l2a(self):
        principal = copy.deepcopy(self.item)
        principal["collection"] = "sentinel-2-c1-l2a"
        principal["properties"]["s2:product_uri"] = (
            "S2A_MSIL2A_20201101T171700_N0500_R112_T13QGB_20231001T000000.SAFE"
        )
        legado = copy.deepcopy(principal)
        legado["collection"] = "sentinel-2-l2a"
        legado["id"] = "S2A_13QGB_20201101_0_L2A"
        legado["properties"]["s2:product_uri"] = (
            "S2A_MSIL2A_20201101T171700_N0200_R112_T13QGB_20201101T200000.SAFE"
        )
        exclusiva = copy.deepcopy(legado)
        exclusiva["id"] = "S2A_13QGB_20201106_0_L2A"
        exclusiva["properties"]["datetime"] = "2020-11-06T17:28:00Z"
        exclusiva["properties"]["s2:product_uri"] = (
            "S2A_MSIL2A_20201106T171700_N0200_R112_T13QGB_20201106T200000.SAFE"
        )
        respuesta = Mock(url=EARTH_SEARCH_URL + "/search")
        respuesta.json.return_value = {
            "features": [legado, exclusiva, principal],
            "links": [],
        }
        cliente = Mock()
        cliente.get.return_value = respuesta
        with patch("datos_sentinel._cliente", return_value=cliente):
            items = buscar_escenas([-103, 19, -102, 20], ["2020-11-01", "2020-11-25"])
        self.assertEqual(
            [item["id"] for item in items], [principal["id"], exclusiva["id"]]
        )

    def test_catalogo_vacio_anterior_se_refresca_y_conserva_tiff(self):
        cfg = {"_raiz": str(self.raiz)}
        lugar = {
            "crs": "EPSG:32613",
            "bbox": [100000, 2099360, 100640, 2100000],
            "transform": [10, 0, 100000, 0, -10, 2100000],
        }
        parametros = {
            "observaciones_sentinel_por_caso": 3,
            "max_descargas_sentinel_por_caso": 7,
            "clases_scl_claras": [4, 5, 6],
            "fraccion_clara_minima": 0.8,
        }
        plan = {"firma_plan": "prueba_cache_2022", "configuracion": parametros}
        caso = {
            "id": "16102_2022",
            "inicio_contexto": "2022-11-01",
            "fin_contexto": "2022-11-25",
        }
        destino = (
            self.raiz
            / "datos_piloto"
            / plan["firma_plan"][:12]
            / caso["id"]
            / "sentinel2"
        )
        destino.mkdir(parents=True)
        guardar_json(destino / "catalogo.json", [])
        items = []
        for dia in ["05", "15", "25"]:
            item = copy.deepcopy(self.item)
            item["id"] = f"S2A_13QGB_202211{dia}_0_L2A"
            item["collection"] = "sentinel-2-l2a"
            item["properties"]["datetime"] = f"2022-11-{dia}T17:28:00Z"
            items.append(item)
        a = np.full((64, 64, 12), 0.1, dtype=np.float32)
        a[..., 10], a[..., 11] = 4, 1
        archivo = destino / "2022-11-05.tif"
        with rasterio.open(
            archivo,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=12,
            dtype="float32",
            crs=lugar["crs"],
            transform=from_origin(100000, 2100000, 10, 10),
        ) as dst:
            dst.write(a.transpose(2, 0, 1))
            dst.descriptions = tuple(ALL_DOWNLOAD_BANDS)
        antes = archivo.read_bytes()
        with patch(
            "Piloto_historico_aguacate.descargas_piloto.buscar_escenas",
            return_value=items,
        ) as consulta:
            with patch(
                "Piloto_historico_aguacate.descargas_piloto.leer_escenas",
                return_value=a,
            ) as lectura:
                descargar_sentinel(cfg, plan, caso, lugar)
        self.assertEqual(consulta.call_count, 1)
        self.assertEqual(lectura.call_count, 2)
        self.assertEqual(antes, archivo.read_bytes())
        catalogo = json.loads((destino / "catalogo.json").read_text())
        self.assertEqual(
            catalogo["consulta"]["colecciones"], list(EARTH_SEARCH_COLECCIONES)
        )
        self.assertEqual(len(catalogo["items"]), 3)
        secuencia = json.loads((destino / "secuencia.json").read_text())
        self.assertEqual(
            secuencia["fechas"], ["2022-11-05", "2022-11-15", "2022-11-25"]
        )

    def test_piloto_guardado_reutiliza_tiff_y_no_consulta_red(self):
        cfg = {"_raiz": str(self.raiz)}
        lugar = {
            "crs": "EPSG:32613",
            "bbox": [100000, 2099360, 100640, 2100000],
            "transform": [10, 0, 100000, 0, -10, 2100000],
        }
        p = {
            "observaciones_sentinel_por_caso": 3,
            "max_descargas_sentinel_por_caso": 3,
            "clases_scl_claras": [4, 5, 6],
            "fraccion_clara_minima": 0.8,
        }
        plan = {"firma_plan": "prueba_earth_search", "configuracion": p}
        caso = {
            "id": "16102_2020",
            "inicio_contexto": "2020-11-01",
            "fin_contexto": "2020-11-25",
        }
        # ubicaciones utiliza los doce primeros caracteres de firma_plan.
        destino = (
            self.raiz
            / "datos_piloto"
            / plan["firma_plan"][:12]
            / caso["id"]
            / "sentinel2"
        )
        destino.mkdir(parents=True)
        items = []
        for dia in ["01", "13", "25"]:
            item = copy.deepcopy(self.item)
            item["id"] = item["id"].replace("20201101", "202011" + dia)
            item["properties"]["datetime"] = f"2020-11-{dia}T17:28:00Z"
            items.append(item)
        guardar_json(destino / "catalogo.json", items)
        a = np.full((64, 64, 12), 0.1, dtype=np.float32)
        a[..., 10], a[..., 11] = 4, 1
        with patch(
            "Piloto_historico_aguacate.descargas_piloto.buscar_escenas",
            return_value=items,
        ) as consulta:
            with patch(
                "Piloto_historico_aguacate.descargas_piloto.leer_escenas",
                return_value=a,
            ) as lectura:
                descargar_sentinel(cfg, plan, caso, lugar)
                self.assertEqual(lectura.call_count, 3)
            self.assertEqual(consulta.call_count, 1)
        hashes = {f.name: f.read_bytes() for f in destino.glob("*.tif")}
        with patch(
            "Piloto_historico_aguacate.descargas_piloto.buscar_escenas",
            side_effect=AssertionError("red innecesaria"),
        ):
            with patch(
                "Piloto_historico_aguacate.descargas_piloto.leer_escenas",
                side_effect=AssertionError("descarga repetida"),
            ):
                descargar_sentinel(cfg, plan, caso, lugar)
        self.assertEqual(
            hashes, {f.name: f.read_bytes() for f in destino.glob("*.tif")}
        )
        m = json.loads((destino / "secuencia.json").read_text())
        self.assertEqual(m["fechas"], ["2020-11-01", "2020-11-13", "2020-11-25"])
        with rasterio.open(destino / m["archivos"][0]) as src:
            self.assertIn("Earth Search", src.tags()["proveedor"])


if __name__ == "__main__":
    unittest.main()
