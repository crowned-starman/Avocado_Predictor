"""Entrada principal del proyecto. La guía completa está en LEEME.md."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def ejecutar_modulo(modulo, argumentos):
    return subprocess.run(
        [sys.executable, "-m", modulo, *map(str, argumentos)], cwd=RAIZ
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="etapa", required=True)
    etapas = {}
    for nombre, ayuda in [
        ("territorio", "Generar etiquetas, mapas y cruce con tu malla."),
        ("mapa", "Abrir el mapa de selección municipal."),
        (
            "actualizar-fuentes",
            "Actualizar únicamente las fuentes públicas agrícolas y municipales.",
        ),
        ("ver-resultados", "Visualizar resultados guardados de la prueba de ingesta."),
    ]:
        etapas[nombre] = sub.add_parser(nombre, help=ayuda)
    etapas["territorio"].add_argument("--datos", type=Path)
    etapas["territorio"].add_argument("--salida", type=Path)
    etapas["ingesta"] = sub.add_parser(
        "ingesta", help="Prueba local de TAMRF y Chronos-2."
    )
    etapas["ingesta"].add_argument(
        "accion",
        nargs="?",
        default="todo",
        choices=["preparar", "descargar-modelos", "probar", "todo"],
    )
    etapas["piloto"] = sub.add_parser(
        "piloto", help="Historial coincidente y conexión de los tres modelos."
    )
    etapas["piloto"].add_argument(
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
    etapas["revisar-datos"] = sub.add_parser(
        "revisar-datos", help="Ejecutar la revisión del notebook desde la terminal."
    )
    etapas["revisar-datos"].add_argument(
        "--fuente", choices=["sentinel", "lluvia", "agricola", "todas"], default="todas"
    )
    etapas["revisar-datos"].add_argument(
        "--descargar-faltantes-sentinel",
        action="store_true",
        help="Descargar únicamente los TIFF Sentinel que falten después de validar la caché.",
    )
    etapas["aplicar-seleccion"] = sub.add_parser(
        "aplicar-seleccion", help="Importar la selección descargada del mapa."
    )
    etapas["aplicar-seleccion"].add_argument("--archivo", type=Path, required=True)
    etapas["sentinel-s3"] = sub.add_parser(
        "sentinel-s3", help="Filtrar recuadros, planificar el histórico y transferir a S3 desde la nube."
    )
    etapas["sentinel-s3"].add_argument("accion", choices=["preparar", "planificar", "transferir"])
    etapas["sentinel-s3"].add_argument("--max-casos", type=int, default=0)
    etapas["sentinel-s3"].add_argument("--max-gib", type=float, default=0)
    actualizar = sub.add_parser(
        "actualizar",
        help="Aplicar esta versión a tu proyecto anterior conservando datos y ajustes.",
    )
    actualizar.add_argument("--destino", type=Path, required=True)
    for p in etapas.values():
        p.add_argument("--config", type=Path, default=RAIZ / "configuracion.json")
    args = parser.parse_args(argv)
    if args.etapa == "actualizar":
        from mantenimiento import actualizar_proyecto

        actualizar_proyecto(RAIZ, args.destino)
        return 0
    from utilidades import guardar_json, leer_json

    config = args.config.resolve()
    if args.etapa == "sentinel-s3":
        return ejecutar_modulo("sentinel_s3", [args.accion, "--config", config,
            "--max-casos", args.max_casos, "--max-gib", args.max_gib])
    if args.etapa == "ingesta":
        return ejecutar_modulo("prueba_ingesta", [args.accion, "--config", config])
    if args.etapa == "piloto":
        return ejecutar_modulo(
            "Piloto_historico_aguacate.piloto_historico",
            [args.accion, "--config-ingesta", config],
        )
    if args.etapa == "territorio":
        opciones = ["--config", config, "--config-ingesta", config]
        if args.datos:
            opciones.extend(["--datos", args.datos.resolve()])
        if args.salida:
            opciones.extend(["--salida", args.salida.resolve()])
        return ejecutar_modulo(
            "Delimitacion_rendimiento_aguacate.preparar_territorio", opciones
        )
    if args.etapa == "actualizar-fuentes":
        return ejecutar_modulo(
            "Delimitacion_rendimiento_aguacate.actualizar_fuentes", []
        )
    if args.etapa == "ver-resultados":
        return ejecutar_modulo("ver_resultados", ["--config", config])
    if args.etapa == "revisar-datos":
        from revision_datos import ejecutar_revision

        ejecutar_revision(
            config,
            args.fuente,
            descargar_faltantes_sentinel=args.descargar_faltantes_sentinel,
        )
        return 0
    if args.etapa == "mapa":
        ruta = (
            RAIZ
            / "Delimitacion_rendimiento_aguacate/resultados_territorio/mapa_seleccion.html"
        )
        if not ruta.exists():
            raise FileNotFoundError(
                "Genera primero el mapa con: python ejecutar.py territorio"
            )
        if os.name == "nt":
            os.startfile(str(ruta))
        else:
            webbrowser.open(ruta.as_uri())
        return 0
    if args.etapa == "aplicar-seleccion":
        nuevo = leer_json(args.archivo)
        claves = nuevo.get("cvegeo_seleccionados")
        if (
            not isinstance(claves, list)
            or not claves
            or len(set(claves)) != len(claves)
            or any(
                not isinstance(c, str) or len(c) != 5 or not c.isdigit() for c in claves
            )
        ):
            raise ValueError(
                "La selección debe contener claves municipales únicas de cinco dígitos."
            )
        catalogo = leer_json(
            RAIZ
            / "Delimitacion_rendimiento_aguacate/fuentes/michoacan_inegi_original.json"
        )
        conocidas = {f["properties"]["cvegeo"] for f in catalogo["features"]}
        if not set(claves) <= conocidas:
            raise ValueError(
                "Hay claves que no pertenecen al catálogo municipal de Michoacán."
            )
        cfg = leer_json(config)
        cfg.setdefault("territorio", {})["cvegeo_seleccionados"] = sorted(claves)
        guardar_json(config, cfg)
        print(
            f"Selección guardada: {len(claves)} municipios. Ejecuta ahora: python ejecutar.py territorio"
        )
        return 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
