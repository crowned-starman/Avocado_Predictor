"""Geometría, conservación de píxeles, interrupciones y reanudación S3 sin cuenta real."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
from botocore.exceptions import ClientError
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

import sentinel_s3 as ss
from cobertura_territorial import cargar_malla, calcular_cobertura
from datos_sentinel import ASSETS
from utilidades import guardar_json, proyectar, sha256


class S3Memoria:
    def __init__(self):
        self.objetos = {}
        self.subidas = 0
        self.fallar_banda = None

    def get_bucket_location(self, **kwargs):
        return {"LocationConstraint": "us-west-2"}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys, **kwargs):
        return {"IsTruncated": False, "Contents": [{"Key": k, "Size": len(v[0])}
                for k, v in self.objetos.items() if k.startswith(Prefix)]}

    def head_object(self, Bucket, Key):
        if Key not in self.objetos:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        b, meta = self.objetos[Key]
        return {"ContentLength": len(b), "Metadata": meta}

    def upload_fileobj(self, Fileobj, Bucket, Key, ExtraArgs, Config):
        if self.fallar_banda and Key.endswith(self.fallar_banda + ".tif"):
            raise IOError("interrupción simulada")
        self.objetos[Key] = (Fileobj.read(), ExtraArgs["Metadata"])
        self.subidas += 1

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objetos[Key] = (Body, {})


class SentinelS3Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.bbox = [770000, 2129960, 770040, 2130000]
        g = proyectar(box(*self.bbox), 32613, 4326)
        self.corte = {"corte": 47, "crs_original": "EPSG:32613", "bbox_original": self.bbox,
                      "fraccion_aoi": 1.0, "geometry": mapping(g)}
        self.item = {"id": "S2A_T13QGB_20201101T171700_L2A", "collection": "sentinel-2-c1-l2a",
                     "geometry": mapping(g), "properties": {"datetime": "2020-11-01T17:17:00Z"}, "assets": {}}
        for banda, clave in ASSETS.items():
            res = ss.RESOLUCIONES[banda]
            lado = 40 // res
            datos = np.arange(lado * lado, dtype="uint16").reshape(lado, lado) + 500
            if banda == "SCL":
                datos = np.array([[4, 5], [9, 0]], dtype="uint8")
            datos[0, 0] = 0 if banda != "SCL" else 4
            ruta = self.raiz / f"{banda}.tif"
            with rasterio.open(ruta, "w", driver="GTiff", height=lado, width=lado,
                               count=1, dtype=datos.dtype, nodata=0, crs="EPSG:32613",
                               transform=from_origin(770000, 2130000, res, res)) as dst:
                dst.write(datos, 1)
            rb = {"nodata": 0, "spatial_resolution": res}
            if banda != "SCL":
                rb.update(scale=0.0001, offset=-0.1)
            self.item["assets"][clave] = {"href": str(ruta), "raster:bands": [rb]}

    def cfg_y_plan(self):
        seleccion = {"version": ss.VERSION, "tipo_aoi": ss.TIPO_AOI, "cvegeo": ["16102"],
                     "fraccion_minima_aoi": 0.85, "aoi": self.corte["geometry"], "recuadros": [self.corte]}
        seleccion["firma"] = ss.firma(seleccion)
        guardar_json(self.raiz / "seleccion_s3.json", seleccion)
        parte = self.raiz / "mes.json"
        guardar_json(parte, {"items": [self.item], "tareas": [{"corte": 47, "item": 0}]})
        guardar_json(self.raiz / "plan_descarga_s3.json", {
            "version": ss.VERSION, "catalogo_completo": True, "plan_id": "prueba",
            "seleccion": seleccion, "consulta": {"fecha_inicial": "2020-01-01", "fecha_final": "2025-12-31"},
            "partes": [{"archivo": "mes.json", "sha256": sha256(parte)}]})
        return {"_trabajo": self.raiz, "territorio": {"cvegeo_seleccionados": ["16102"], "fraccion_minima_aoi": 0.85},
                "sentinel_s3": {"bucket": "prueba-aguacate", "prefijo": "aguacate/sentinel2", "region": "us-west-2",
                                "fecha_inicial": "2020-01-01", "fecha_final": "2025-12-31"}}

    def test_conserva_dn_mascara_crs_y_calibracion_como_cog(self):
        for banda in ASSETS:
            with self.subTest(banda=banda):
                memoria, info = ss.construir_tiff(self.item, self.corte, banda)
                with memoria, memoria.open() as salida, rasterio.open(self.item["assets"][ASSETS[banda]]["href"]) as origen:
                    np.testing.assert_array_equal(salida.read(1), origen.read(1))
                    np.testing.assert_array_equal(salida.read_masks(1), origen.read_masks(1))
                    self.assertEqual(salida.transform, origen.transform)
                    self.assertEqual(salida.res[0], ss.RESOLUCIONES[banda])
                    self.assertEqual(salida.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"), "COG")
                    if banda != "SCL":
                        self.assertEqual(salida.scales, (0.0001,))
                        self.assertEqual(salida.offsets, (-0.1,))
                        self.assertEqual(salida.read(1)[0, 1], 501)  # No recorta reflectancia negativa.
                    self.assertGreater(info["bytes"], 0)
                if banda == "SCL":
                    self.assertEqual(info["fraccion_clara_recuadro"], 0.5)
        self.assertFalse(list(self.raiz.glob("*.msk")))

    def test_reanuda_caso_completo_sin_leer_red_ni_subir_tiff(self):
        cfg, s3 = self.cfg_y_plan(), S3Memoria()
        a = ss.transferir(cfg, s3=s3)
        self.assertTrue(a["terminado"])
        self.assertEqual(s3.subidas, 11)
        with patch.object(ss, "construir_tiff", side_effect=AssertionError("No debe leer las imágenes otra vez")):
            b = ss.transferir(cfg, s3=s3)
        self.assertTrue(b["terminado"])
        self.assertEqual(b["casos_reutilizados"], 1)
        self.assertEqual(s3.subidas, 11)

    def test_interrupcion_no_publica_completo_y_reanuda_bandas_faltantes(self):
        cfg, s3 = self.cfg_y_plan(), S3Memoria()
        s3.fallar_banda = "B04"
        a = ss.transferir(cfg, s3=s3)
        self.assertFalse(a["terminado"])
        self.assertEqual(s3.subidas, 2)
        self.assertFalse(any(k.endswith("_completo.json") for k in s3.objetos))
        s3.fallar_banda = None
        b = ss.transferir(cfg, s3=s3)
        self.assertTrue(b["terminado"])
        self.assertEqual(b["bandas_reutilizadas"], 2)
        self.assertEqual(s3.subidas, 11)

    def test_no_sobrescribe_objeto_incompatible(self):
        s3 = S3Memoria()
        s3.objetos["a.tif"] = (b"datos ajenos", {})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            ss.transferir_banda(s3, "prueba-aguacate", "a.tif", self.item, self.corte, "B02")
        self.assertEqual(s3.subidas, 0)

    def test_limite_bytes_se_detiene_sin_subir(self):
        with self.assertRaises(ss.LimiteTransferencia):
            ss.transferir_banda(S3Memoria(), "prueba-aguacate", "a.tif", self.item, self.corte, "B02", limite_bytes=1)

    def test_rechaza_cambio_de_umbral_despues_del_plan(self):
        cfg = self.cfg_y_plan()
        cfg["territorio"]["fraccion_minima_aoi"] = 0.9
        with self.assertRaisesRegex(ValueError, "umbral"):
            ss.transferir(cfg, s3=S3Memoria())

    def test_2026_no_entra_en_desarrollo(self):
        p = self.raiz / "config.json"
        guardar_json(p, {"sentinel_s3": {"carpeta_trabajo": "trabajo", "fecha_inicial": "2020-01-01", "fecha_final": "2026-01-01"}})
        with self.assertRaisesRegex(ValueError, "2026"):
            ss.configuracion(p)

    def test_planifica_meses_y_reutiliza_catalogos_sin_descargar_bandas(self):
        cfg = self.cfg_y_plan()
        cfg["sentinel_s3"].update(fecha_inicial="2020-11-01", fecha_final="2020-12-31")
        with patch.object(ss, "buscar_escenas", side_effect=[[self.item], []]) as buscar:
            plan = ss.planificar(cfg)
            self.assertEqual(buscar.call_count, 2)
            self.assertEqual(plan["casos_escena_recuadro"], 1)
            self.assertEqual(plan["archivos_banda_previstos"], 11)
            self.assertTrue(plan["catalogo_completo"])
        with patch.object(ss, "buscar_escenas", side_effect=AssertionError("No se repite catálogo")):
            segundo = ss.planificar(cfg)
        self.assertEqual(segundo["plan_id"], plan["plan_id"])
        self.assertEqual(len(plan["partes"]), 2)


class CoberturaTest(unittest.TestCase):
    def test_no_agrega_casillas_ajenas_a_la_seleccion_previa(self):
        tabla = pd.DataFrame([
            {"corte": 27, "xmin": 770000, "ymin": 2130000, "xmax": 771000, "ymax": 2131000},
            {"corte": 28, "xmin": 771000, "ymin": 2130000, "xmax": 772000, "ymax": 2131000}])
        municipios = [("a", "A", proyectar(box(769000, 2129000, 773000, 2132000), 32613, 4326))]
        t, _, _ = calcular_cobertura(tabla, 32613, municipios, ["a"], 0.85, [27])
        self.assertTrue((t.fraccion_aoi > 0.99).all())
        self.assertEqual(t.loc[t.seleccionado_por_territorio, "corte"].tolist(), [27])
        self.assertEqual(t.loc[t.corte == 28, "motivo_filtro"].iloc[0], "fuera_de_seleccion_previa")
        with self.assertRaisesRegex(ValueError, "no contiene"):
            calcular_cobertura(tabla, 32613, municipios, ["a"], 0.85, [27, 49])

    def test_comparacion_umbrales_solo_dentro_de_preseleccion(self):
        tabla = pd.DataFrame([
            {"corte": 27, "xmin": 770000, "ymin": 2130000, "xmax": 771000, "ymax": 2131000},
            {"corte": 28, "xmin": 771000, "ymin": 2130000, "xmax": 772000, "ymax": 2131000}])
        municipios = [("a", "A", proyectar(box(769000, 2129000, 773000, 2132000), 32613, 4326))]
        t, g, _ = calcular_cobertura(tabla, 32613, municipios, ["a"], 0.85, [27])
        from cobertura_territorial import escribir_filtro
        with tempfile.TemporaryDirectory() as d:
            r = escribir_filtro(d, t, g, 0.85)
        self.assertEqual(r["comparacion_umbrales"], {"80": 1, "85": 1, "90": 1})
        self.assertEqual(r["cantidad_seleccion_previa"], 1)

    def test_usa_union_no_suma_solapamientos_y_no_renumera(self):
        tabla = pd.DataFrame([{"corte": 121, "xmin": 770000, "ymin": 2130000, "xmax": 780000, "ymax": 2140000},
                              {"corte": 130, "xmin": 790000, "ymin": 2130000, "xmax": 800000, "ymax": 2140000}])
        # La unión cubre 86 % del primer recuadro y 100 % del segundo.
        municipios = [("a", "A", proyectar(box(770000, 2130000, 778600, 2140000), 32613, 4326)),
                      ("b", "B", proyectar(box(770000, 2130000, 778000, 2140000), 32613, 4326)),
                      ("c", "C", proyectar(box(790000, 2130000, 800000, 2140000), 32613, 4326))]
        a, _, _ = calcular_cobertura(tabla, 32613, municipios, ["a", "b", "c"], 0.85)
        self.assertTrue(a.seleccionado_por_territorio.all())
        self.assertAlmostEqual(a.fraccion_aoi.iloc[0], 0.86, places=3)
        b, _, _ = calcular_cobertura(tabla, 32613, municipios, ["a", "b", "c"], 0.90)
        self.assertEqual(b.loc[b.seleccionado_por_territorio, "corte"].tolist(), [130])
        self.assertEqual(a.corte.tolist(), [121, 130])

    def test_sin_crs_o_identificadores_duplicados_se_detiene(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "malla.csv"
            pd.DataFrame([{"corte": 1, "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}]).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "CRS"):
                cargar_malla(p)


if __name__ == "__main__":
    unittest.main()
