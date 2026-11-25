import copy
import datetime as dt
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

import sentinel_s3 as ss
import test_sentinel_s3 as legacy
from datos_sentinel import ASSETS
from utilidades import proyectar


class S3(legacy.S3Memoria):
    def get_object(self, Bucket, Key):
        if Key not in self.objetos:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objetos[Key][0])}


class CalidadTest(unittest.TestCase):
    def setUp(self):
        self.base = legacy.SentinelS3Test()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.raiz
        self.corte = self.base.corte
        self.cfg = self.base.cfg_y_plan()
        self.cfg["sentinel_s3"].update(fecha_inicial="2020-11-01", fecha_final="2020-11-30",
            seleccion_temporal={"modo": "mejor_por_bloque", "fraccion_valida_minima": .95,
                "fraccion_clara_minima": .8, "usar_cache_s3": False})

    def item(self, dia, scl, *, sufijo="A", bbox=None, datatake=None):
        it = copy.deepcopy(self.base.item)
        bbox = bbox or self.corte["bbox_original"]
        it["id"] = f"S2A_T13QG{sufijo}_202011{dia:02d}T171700_L2A"
        it["properties"].update(datetime=f"2020-11-{dia:02d}T17:17:00Z", platform="sentinel-2a",
            **{"s2:datatake_id": datatake or f"GS2A_202011{dia:02d}T171700_012345_N05.11"})
        it["geometry"] = mapping(proyectar(box(*bbox), 32613, 4326))
        for banda, clave in ASSETS.items():
            res = ss.RESOLUCIONES[banda]
            w, h = int((bbox[2] - bbox[0]) / res), int((bbox[3] - bbox[1]) / res)
            ruta = self.root / f"{dia}_{sufijo}_{banda}.tif"
            datos = np.full((h, w), 2000, dtype="uint16") if banda != "SCL" else np.broadcast_to(
                np.asarray(scl, dtype="uint8"), (h, w)).copy()
            with rasterio.open(ruta, "w", driver="GTiff", height=h, width=w, count=1,
                    dtype=datos.dtype, nodata=0, crs="EPSG:32613",
                    transform=from_origin(bbox[0], bbox[3], res, res)) as dst:
                dst.write(datos, 1)
            it["assets"][clave]["href"] = str(ruta)
        return it

    def plan(self, items, s3=None):
        with patch.object(ss, "buscar_escenas", return_value=items):
            return ss.planificar(self.cfg, s3=s3)

    def casos(self, plan):
        return [c for parte in plan["partes"] for c in json.loads(
            (self.root / parte["archivo"]).read_text())["casos"]]

    def test_solo_se_abre_scl_y_se_mide_sobre_todo_el_recuadro(self):
        it = self.item(2, [[4, 9], [4, 0]])
        with patch.object(ss.rasterio, "open", wraps=rasterio.open) as abrir:
            r = ss.medir_calidad([it], self.corte)
        self.assertEqual([c.args[0] for c in abrir.call_args_list], [it["assets"]["scl"]["href"]])
        self.assertEqual(r["fraccion_clara"], .5)
        self.assertEqual(r["fraccion_valida"], .75)

    def test_mejor_local_aunque_la_nubosidad_global_sea_alta(self):
        a, b = self.item(2, [[4, 9], [4, 4]]), self.item(6, 4)
        a["properties"]["eo:cloud_cover"] = 0
        b["properties"]["eo:cloud_cover"] = 95
        p = self.plan([a, b])
        self.assertEqual(len(self.casos(p)), 1)
        self.assertEqual(self.casos(p)[0]["fecha_utc"][:10], "2020-11-06")
        self.assertEqual(p["resumen_temporal"]["faltantes"], 3)

    def test_una_adquisicion_por_bloque_y_no_rellena_otro_bloque(self):
        p = self.plan([self.item(2, 4), self.item(6, 4), self.item(17, 4), self.item(28, 4)])
        casos = self.casos(p)
        self.assertEqual(len(casos), 3)
        self.assertEqual([c["bloque"] for c in casos], [1, 3, 4])
        self.assertTrue(all(c["fecha_utc"][:10] >= c["intervalo"][0] and c["fecha_utc"][:10] <= c["intervalo"][1] for c in casos))

    def test_no_elige_esquina_despejada_con_cobertura_insuficiente(self):
        p = self.plan([self.item(2, [[4, 0], [4, 0]])])
        self.assertEqual(p["conjuntos_periodo_recuadro"], 0)
        estado = json.loads((self.root / "informe_seleccion_temporal.json").read_text())
        self.assertEqual(estado["seleccionados"], 0)
        self.assertEqual(estado["faltantes"], 4)

    def test_teselas_de_misma_toma_completan_un_conjunto(self):
        x0, y0, x1, y1 = self.corte["bbox_original"]
        toma = "GS2A_20201102T171700_012345_N05.11"
        a = self.item(2, 4, sufijo="A", bbox=[x0,y0,x0+20,y1], datatake=toma)
        b = self.item(2, 4, sufijo="B", bbox=[x0+20,y0,x1,y1], datatake=toma)
        p = self.plan([a, b])
        self.assertEqual(p["conjuntos_periodo_recuadro"], 1)
        c = self.casos(p)[0]
        self.assertEqual(len(c["items"]), 2)
        self.assertEqual(c["metricas"]["fraccion_clara"], 1)
        self.assertEqual(p["archivos_banda_previstos"], 22)
        s3 = S3()
        r = ss.transferir(self.cfg, s3=s3)
        self.assertTrue(r["terminado"])
        self.assertEqual(r["casos_completados_esta_ejecucion"], 1)
        self.assertEqual(s3.subidas, 22)
        with patch.object(ss, "construir_tiff", side_effect=AssertionError("No se repiten datos")):
            r2 = ss.transferir(self.cfg, s3=s3)
        self.assertEqual(r2["casos_reutilizados"], 1)

    def test_teselas_de_dias_distintos_no_se_combinan(self):
        x0,y0,x1,y1=self.corte["bbox_original"]
        a=self.item(2,4,bbox=[x0,y0,x0+20,y1])
        b=self.item(3,4,sufijo="B",bbox=[x0+20,y0,x1,y1])
        p=self.plan([a,b])
        self.assertEqual(p["conjuntos_periodo_recuadro"],0)

    def test_evaluacion_fallida_bloquea_transferencia_y_se_reintenta(self):
        a,b=self.item(2,4),self.item(6,4)
        original=ss.medir_calidad
        def medir(items,*args,**kwargs):
            if items[0]["id"]==b["id"]:raise IOError("lectura fallida")
            return original(items,*args,**kwargs)
        with patch.object(ss,"buscar_escenas",return_value=[a,b]),patch.object(ss,"medir_calidad",side_effect=medir):
            with self.assertRaisesRegex(ValueError,"evaluaciones SCL"):
                ss.planificar(self.cfg)
        with self.assertRaisesRegex(ValueError,"no está listo"):
            ss.transferir(self.cfg,s3=S3())
        with patch.object(ss,"medir_calidad",wraps=original) as medir2:
            p=self.plan([a,b])
        self.assertEqual(medir2.call_count,1)
        self.assertTrue(p["listo_para_transferir"])

    def test_cache_s3_recupera_plan_y_calidad_sin_releer_imagenes(self):
        s3=S3();self.cfg["sentinel_s3"]["seleccion_temporal"]["usar_cache_s3"]=True
        p=self.plan([self.item(2,4)],s3)
        for carpeta in ("catalogos","calidad","seleccionados"):
            shutil.rmtree(self.root/carpeta)
        with patch.object(ss,"buscar_escenas",side_effect=AssertionError("No repetir catálogo")),patch.object(ss,"medir_calidad",side_effect=AssertionError("No repetir SCL")):
            p2=ss.planificar(self.cfg,s3=s3)
        self.assertEqual(p["plan_id"],p2["plan_id"])

    def test_cambio_de_calidad_requiere_replanificar(self):
        self.plan([self.item(2,4)])
        self.cfg["sentinel_s3"]["seleccion_temporal"]["fraccion_clara_minima"]=.9
        with self.assertRaisesRegex(ValueError,"cambió la calidad"):
            ss.transferir(self.cfg,s3=S3())

    def test_reutiliza_tiff_de_una_descarga_anterior_sin_seleccion_temporal(self):
        item = self.item(2, 4)
        p = copy.deepcopy(self.cfg["sentinel_s3"]["seleccion_temporal"])
        self.cfg["sentinel_s3"]["seleccion_temporal"]["modo"] = "todas"
        with patch.object(ss, "buscar_escenas", return_value=[item]):
            ss.planificar_todas(self.cfg)
        s3 = S3()
        ss.transferir(self.cfg, s3=s3)
        self.assertEqual(s3.subidas, 11)
        self.cfg["sentinel_s3"]["seleccion_temporal"] = p
        self.plan([item])
        with patch.object(ss, "construir_tiff", side_effect=AssertionError("No debe repetir TIFF")):
            r = ss.transferir(self.cfg, s3=s3)
        self.assertEqual(r["bytes_tiff_nuevos"], 0)
        self.assertEqual(s3.subidas, 11)
        self.assertTrue(any("/selecciones/" in k and k.endswith("/_completo.json") for k in s3.objetos))

    def test_interrupcion_entre_teselas_no_publica_un_conjunto_incompleto(self):
        x0,y0,x1,y1 = self.corte["bbox_original"]
        toma = "GS2A_20201102T171700_012345_N05.11"
        a = self.item(2,4,sufijo="A",bbox=[x0,y0,x0+20,y1],datatake=toma)
        b = self.item(2,4,sufijo="B",bbox=[x0+20,y0,x1,y1],datatake=toma)
        self.plan([a,b])
        s3 = S3()
        original = s3.upload_fileobj
        def subir(Fileobj, Bucket, Key, ExtraArgs, Config):
            if b["id"] in Key and Key.endswith("B04.tif"):
                raise IOError("Interrupción en segunda tesela")
            return original(Fileobj, Bucket, Key, ExtraArgs, Config)
        with patch.object(s3,"upload_fileobj",side_effect=subir):
            r = ss.transferir(self.cfg,s3=s3)
        self.assertFalse(r["terminado"])
        self.assertFalse(any("/selecciones/" in k and k.endswith("/_completo.json") for k in s3.objetos))
        r2 = ss.transferir(self.cfg,s3=s3)
        self.assertTrue(r2["terminado"])
        self.assertEqual(s3.subidas,22)

    def test_plan_interrumpido_no_habilita_descarga_completa_sin_filtrar(self):
        with patch.object(ss,"buscar_escenas",return_value=[self.item(2,4)]):
            ss.planificar_todas(self.cfg)
        with self.assertRaisesRegex(ValueError,"no está listo"):
            ss.transferir(self.cfg,s3=S3())


class CalendarioTest(unittest.TestCase):
    def test_48_bloques_por_anio_sin_huecos_ni_solapes(self):
        for anio in (2020,2021,2024,2025):
            dias=[];n=0
            for a,b in ss.meses(dt.date(anio,1,1),dt.date(anio,12,31)):
                for _,ini,fin in ss.bloques_mes(a,b):
                    n+=1
                    dias.extend(ini+dt.timedelta(days=i) for i in range((fin-ini).days+1))
            self.assertEqual(n,48)
            self.assertEqual(len(dias),len(set(dias)))
            self.assertEqual(len(dias),366 if anio in (2020,2024) else 365)

    def test_anios_separados_y_configuracion_original_intacta(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config.json"
            p.write_text(json.dumps({"sentinel_s3":{"bucket":"mi-bucket","region":"us-east-2"},"territorio":{}}))
            original=p.read_bytes()
            a,b=ss.configuracion(p,2020),ss.configuracion(p,2025)
            self.assertNotEqual(a["_trabajo"],b["_trabajo"])
            self.assertEqual(a["sentinel_s3"]["fecha_inicial"],"2020-01-01")
            self.assertEqual(b["sentinel_s3"]["fecha_final"],"2025-12-31")
            self.assertEqual(b["sentinel_s3"]["bucket"],"mi-bucket")
            self.assertEqual(p.read_bytes(),original)
            for invalido in (True,2019,2026,"2020",2020.5):
                with self.assertRaises(ValueError):ss.configuracion(p,invalido)


if __name__=="__main__":unittest.main()
