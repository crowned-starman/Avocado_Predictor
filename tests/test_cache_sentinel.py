"""Regresión de la caché: un cambio de formato no vuelve a bajar el mosaico."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from cache_sentinel import _script_anterior, descriptor, firma_recuadro, resolver_cache
from datos_sentinel import ALL_DOWNLOAD_BANDS, make_evalscript
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as transformar_ventana
from utilidades import guardar_json, huella


class CacheSentinelTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.tr = from_origin(100000, 2100000, 10, 10)
        self.peticion = {
            "version_cobertura": 2,
            "bbox": [100000, 2099980, 100040, 2100000],
            "crs": "EPSG:32613",
            "size": [4, 2],
            "timestamp": "2026-03-26T17:28:52.431000+00:00",
            "interval": ["2026-03-26T17:18:52.431000Z", "2026-03-26T17:38:52.431000Z"],
            "bands": ALL_DOWNLOAD_BANDS,
            "evalscript": make_evalscript(),
            "max_cloud": 0.4,
            "resampling": "NEAREST",
            "harmonizeValues": True,
            "ids_productos_catalogo": ["escena_A"],
            "max_px": 2,
            "huellas": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [100000, 2099980],
                        [100040, 2099980],
                        [100040, 2100000],
                        [100000, 2100000],
                        [100000, 2099980],
                    ]
                ],
            },
        }
        self.plan = [
            {
                "corte": i,
                "fila_inicio": 0,
                "columna_inicio": i * 2,
                "alto_px": 2,
                "ancho_px": 2,
                "intersecta_escenas": i < 2,
            }
            for i in range(3)
        ]

    def crear(self, peticion, cortes=(0, 1), *, metadatos=False):
        firma = huella(peticion)
        carpeta = self.raiz / ("escenas_completas_" + firma)
        carpeta.mkdir(exist_ok=True)
        for i in cortes:
            p = self.plan[i]
            with rasterio.open(
                carpeta / f"recuadro_{i:04d}.tif",
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=12,
                dtype="float32",
                crs=peticion["crs"],
                transform=transformar_ventana(Window(i * 2, 0, 2, 2), self.tr),
            ) as dst:
                dst.write(np.zeros((12, 2, 2), dtype=np.float32))
                dst.descriptions = tuple(ALL_DOWNLOAD_BANDS)
                dst.update_tags(
                    huella_peticion=firma_recuadro(firma, p), fecha_utc="2026-03-26"
                )
        if metadatos:
            guardar_json(
                self.raiz / "metadatos.json",
                {
                    **peticion,
                    "carpeta_recuadros": "C:\\anterior\\sentinel2\\" + carpeta.name,
                },
            )
        return carpeta

    def resolver(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return resolver_cache(
                self.raiz, self.peticion, self.plan, self.tr, **kwargs
            )

    def test_formato_original_reutiliza_todos_sin_cambiar_tiff(self):
        anterior = {**self.peticion, "evalscript": _script_anterior()}
        self.assertNotEqual(huella(anterior), huella(self.peticion))
        carpeta = self.crear(anterior)
        antes = {
            f: (f.read_bytes(), f.stat().st_mtime_ns) for f in carpeta.glob("*.tif")
        }
        resultado = self.resolver()
        self.assertEqual(resultado["carpeta"], carpeta)
        self.assertEqual(resultado["informe"]["reutilizables"], 2)
        self.assertEqual(resultado["informe"]["faltantes"], [])
        self.assertEqual(
            antes, {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in antes}
        )
        self.assertEqual(len(list(self.raiz.glob("escenas_completas_*"))), 1)

    def test_metadatos_reconocen_orden_geometrico_anterior(self):
        anterior = json.loads(json.dumps(self.peticion))
        anterior["huellas"]["coordinates"][0].reverse()
        anterior["evalscript"] = _script_anterior()
        carpeta = self.crear(anterior, metadatos=True)
        self.assertEqual(descriptor(anterior), descriptor(self.peticion))
        self.assertEqual(self.resolver()["carpeta"], carpeta)
        # Aunque otro paso reescriba metadatos, la carpeta conserva su procedencia.
        (self.raiz / "metadatos.json").write_text("{}")
        self.assertEqual(self.resolver()["carpeta"], carpeta)

    def test_parciales_compatibles_se_complementan(self):
        antigua = self.crear({**self.peticion, "evalscript": _script_anterior()}, (0,))
        nueva = self.crear(self.peticion, (1,))
        resultado = self.resolver()
        self.assertEqual(resultado["archivos"][0].parent, antigua)
        self.assertEqual(resultado["archivos"][1].parent, nueva)
        self.assertEqual(resultado["informe"]["faltantes"], [])

    def test_faltantes_se_detienen_por_defecto_y_no_crean_tiff(self):
        self.crear(self.peticion, (0,))
        with self.assertRaisesRegex(ValueError, "No se descargó ningún TIFF"):
            self.resolver()
        informe = json.loads((self.raiz / "diagnostico_cache.json").read_text())
        self.assertEqual(
            informe["esperados"], 2
        )  # El hueco sin escenas no es faltante.
        self.assertEqual(informe["faltantes"], [1])
        self.assertFalse(informe["descargas_habilitadas"])
        self.assertEqual(len(list(self.raiz.rglob("*.tif"))), 1)

    def test_descarga_explicita_solo_habilita_faltante(self):
        carpeta = self.crear(self.peticion, (0,))
        resultado = self.resolver(permitir_descargas=True)
        self.assertEqual(resultado["informe"]["faltantes"], [1])
        self.assertEqual(resultado["carpeta"], carpeta)
        self.assertEqual(set(resultado["archivos"]), {0})

    def test_incompatibles_se_conservan_aun_con_descarga_habilitada(self):
        for defecto in ["fecha", "crs", "malla", "bandas", "firma", "corrupto"]:
            with self.subTest(defecto=defecto):
                carpeta = self.crear(self.peticion)
                archivo = carpeta / "recuadro_0001.tif"
                if defecto == "corrupto":
                    archivo.write_bytes(b"interrumpido")
                else:
                    with rasterio.open(archivo, "r+") as dst:
                        if defecto == "fecha":
                            dst.update_tags(fecha_utc="2025-03-26")
                        if defecto == "crs":
                            dst.crs = "EPSG:32614"
                        if defecto == "malla":
                            dst.transform = from_origin(0, 10, 20, 20)
                        if defecto == "bandas":
                            dst.set_band_description(1, "B04")
                        if defecto == "firma":
                            dst.update_tags(huella_peticion="incorrecta")
                antes = archivo.read_bytes()
                with self.assertRaisesRegex(ValueError, "incompatible conservado"):
                    self.resolver(permitir_descargas=True)
                self.assertEqual(archivo.read_bytes(), antes)

    def test_no_reutiliza_otra_adquisicion_o_algoritmo(self):
        for cambio in [
            {"timestamp": "2026-03-24T17:28:52.431000+00:00"},
            {"evalscript": make_evalscript().replace("s.B02", "s.B02*2")},
            {"ids_productos_catalogo": ["otra_escena"]},
        ]:
            with self.subTest(cambio=cambio):
                self.crear({**self.peticion, **cambio}, metadatos=True)
                with self.assertRaisesRegex(ValueError, "No se descargó ningún TIFF"):
                    self.resolver()
                informe = json.loads((self.raiz / "diagnostico_cache.json").read_text())
                self.assertEqual(informe["reutilizables"], 0)

    def test_carpeta_nueva_se_reconoce_despues_de_interrumpir(self):
        resultado = self.resolver(permitir_descargas=True)
        carpeta, firma = resultado["carpeta"], resultado["firma"]
        self.assertTrue((carpeta / "peticion_cache.json").is_file())
        p = self.plan[0]
        with rasterio.open(
            carpeta / "recuadro_0000.tif",
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=12,
            dtype="float32",
            crs=self.peticion["crs"],
            transform=self.tr,
        ) as dst:
            dst.write(np.zeros((12, 2, 2), dtype=np.float32))
            dst.descriptions = tuple(ALL_DOWNLOAD_BANDS)
            dst.update_tags(
                huella_peticion=firma_recuadro(firma, p), fecha_utc="2026-03-26"
            )
        siguiente = self.resolver(permitir_descargas=True)
        self.assertEqual(siguiente["carpeta"], carpeta)
        self.assertEqual(siguiente["informe"]["reutilizables"], 1)


if __name__ == "__main__":
    unittest.main()
