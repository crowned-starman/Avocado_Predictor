"""Preparación y prueba de entrada real a TAMRF y Chronos-2. Python 3.12."""

from __future__ import annotations

from utilidades import cargar_configuracion, guardar_json

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "accion",
        choices=["preparar", "descargar-modelos", "probar", "todo"],
        nargs="?",
        default="todo",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("configuracion.json")
    )
    args = parser.parse_args()
    cfg = cargar_configuracion(args.config)
    salida = Path(cfg["carpeta_salida"])
    salida.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(salida / "ejecucion.log", encoding="utf-8"),
        ],
    )
    registro = {
        "inicio_utc": datetime.now(timezone.utc).isoformat(),
        "accion": args.accion,
        "tercer_modelo": "se ejecuta en la etapa piloto; esta prueba local solo carga TAMRF y Chronos-2",
        "etapas": {},
    }
    fallos = []

    def etapa(nombre, funcion):
        try:
            logging.info("Iniciando %s", nombre)
            resultado = funcion()
            registro["etapas"][nombre] = {"estado": "correcto", **resultado}
            logging.info("OK: %s", nombre)
            return True
        except (
            Exception
        ) as error:  # Registrar el fallo de un modelo y continuar con el otro.
            registro["etapas"][nombre] = {
                "estado": "error",
                "tipo": type(error).__name__,
                "detalle": str(error),
                "traza": traceback.format_exc(),
            }
            fallos.append(nombre)
            logging.error("%s: %s", nombre, error)
            logging.debug(traceback.format_exc())
            return False
        finally:
            guardar_json(salida / "informe_ingesta.json", registro)

    if args.accion in ("preparar", "todo"):
        from preparacion import preparar_sentinel, preparar_era5

        etapa("preparar_sentinel2", lambda: preparar_sentinel(cfg))
        etapa("preparar_era5_land", lambda: preparar_era5(cfg))

    if args.accion in ("descargar-modelos", "todo"):
        from recursos_modelos import obtener_tamrf, obtener_chronos

        for nombre, preparacion, fn in [
            ("descargar_tamrf", "preparar_sentinel2", obtener_tamrf),
            ("descargar_chronos2", "preparar_era5_land", obtener_chronos),
        ]:
            if preparacion in fallos:
                registro["etapas"][nombre] = {
                    "estado": "omitido",
                    "detalle": "Primero hay que preparar datos válidos para ese modelo.",
                }
                continue
            etapa(nombre, lambda fn=fn: fn(cfg))

    if args.accion in ("probar", "todo"):
        from modelos import probar_tamrf, probar_chronos

        # Una preparación fallida no debe hacer pasar por actuales los archivos de una ejecución anterior.
        for nombre, preparacion, descarga, fn in [
            ("ingesta_tamrf", "preparar_sentinel2", "descargar_tamrf", probar_tamrf),
            (
                "ingesta_chronos2",
                "preparar_era5_land",
                "descargar_chronos2",
                probar_chronos,
            ),
        ]:
            if preparacion in fallos or descarga in fallos:
                registro["etapas"][nombre] = {
                    "estado": "omitido",
                    "detalle": "La etapa previa falló en esta ejecución.",
                }
                continue
            etapa(nombre, lambda fn=fn: fn(cfg))

    registro["fin_utc"] = datetime.now(timezone.utc).isoformat()
    registro["resultado"] = "error" if fallos else "correcto"
    registro["alcance"] = (
        "Prueba de formato e inferencia; no entrenamiento ni validación de precisión."
    )
    guardar_json(salida / "informe_ingesta.json", registro)
    if fallos:
        logging.error(
            "Quedaron etapas con error: %s. Consulta informe_ingesta.json.",
            ", ".join(fallos),
        )
        return 1
    logging.info("Prueba terminada. Informe: %s", salida / "informe_ingesta.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
