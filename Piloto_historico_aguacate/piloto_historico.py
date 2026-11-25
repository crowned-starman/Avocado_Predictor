"""Prioridades 2 y 3: historial coincidente, dos modelos y regresor tabular piloto."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from utilidades import guardar_json

from .comun_piloto import cargar_config, ubicaciones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "accion",
        nargs="?",
        default="todo",
        choices=[
            "planificar",
            "descargar",
            "descargar-modelos",
            "dos-modelos",
            "tabla",
            "tercero",
            "todo",
        ],
    )
    parser.add_argument(
        "--config-ingesta",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configuracion.json",
    )
    parser.add_argument(
        "--config-piloto", type=Path, default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    cfg, p = cargar_config(args.config_ingesta, args.config_piloto)
    from .plan_piloto import crear_plan

    plan = crear_plan(cfg, p)
    datos, salida = ubicaciones(cfg, plan)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(salida / "ejecucion.log", encoding="utf-8"),
        ],
    )
    for nombre in [
        "sentinelhub",
        "urllib3",
        "httpx",
        "httpcore",
        "botocore",
        "rasterio.session",
    ]:
        logging.getLogger(nombre).setLevel(logging.WARNING)
    registro = dict(
        inicio_utc=datetime.now(timezone.utc).isoformat(),
        accion=args.accion,
        firma_plan=plan["firma_plan"],
        etapas={"planificar": {"estado": "correcto", "casos": len(plan["casos"])}},
        alcance="Prueba de ingesta conectada. Sin métricas de precisión ni ajuste de pesos.",
        datos_2026_utilizados=False,
    )
    from .descargas_piloto import descargar
    from .modelos_piloto import descargar_modelos, ejecutar_dos_modelos, probar_tercero
    from .tabla_piloto import construir_tabla

    funciones = {
        "descargar": descargar,
        "descargar-modelos": lambda c, p: descargar_modelos(c),
        "dos-modelos": ejecutar_dos_modelos,
        "tabla": construir_tabla,
        "tercero": probar_tercero,
    }
    secuencia = (
        list(funciones)
        if args.accion == "todo"
        else ([] if args.accion == "planificar" else [args.accion])
    )
    guardar_json(salida / "informe_piloto.json", registro)
    if args.accion in ["todo", "descargar", "dos-modelos", "tabla", "tercero"]:
        for nombre in ["predicciones_rendimiento_2025.csv", "informe_tabiclv2.json"]:
            (salida / nombre).unlink(missing_ok=True)
    logging.info(
        "Plan: %d municipios × %d años = %d casos. Datos: %s",
        len(plan["lugares"]),
        len(set(c["anio"] for c in plan["casos"])),
        len(plan["casos"]),
        datos,
    )
    for etapa in secuencia:
        logging.info("Iniciando %s", etapa)
        try:
            resultado = funciones[etapa](cfg, plan)
            registro["etapas"][etapa] = {"estado": "correcto", **resultado}
        except Exception as e:
            registro["etapas"][etapa] = {
                "estado": "error",
                "tipo": type(e).__name__,
                "detalle": str(e),
            }
            registro["resultado"] = "error"
            registro["fin_utc"] = datetime.now(timezone.utc).isoformat()
            guardar_json(salida / "informe_piloto.json", registro)
            # No se incluyen cabeceras HTTP ni valores de credenciales en los informes.
            logging.error("%s: %s: %s", etapa, type(e).__name__, e)
            logging.error(
                "Se detuvo la cadena; las etapas posteriores no usan salidas antiguas. Informe: %s",
                salida / "informe_piloto.json",
            )
            return 1
        guardar_json(salida / "informe_piloto.json", registro)
    registro["resultado"] = "correcto"
    registro["fin_utc"] = datetime.now(timezone.utc).isoformat()
    guardar_json(salida / "informe_piloto.json", registro)
    logging.info("Terminado: %s", salida)
    if args.accion in ["todo", "tercero"]:
        logging.info(
            "Predicciones de la prueba: %s",
            salida / "predicciones_rendimiento_2025.csv",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
