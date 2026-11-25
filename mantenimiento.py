"""Actualización local de código: respalda cambios y conserva datos, pesos y ajustes."""

from datetime import datetime
from pathlib import Path
import shutil

from utilidades import leer_json, guardar_json

OBSOLETOS = [
    "Delimitacion_rendimiento_aguacate/requirements_territorio.txt",
    "Piloto_historico_aguacate/requirements_piloto.txt",
    "Delimitacion_rendimiento_aguacate/LEEME.md",
    "Piloto_historico_aguacate/LEEME.md",
    "Piloto_historico_aguacate/VERIFICACION.md",
    "VERIFICACION.md",
    "Delimitacion_rendimiento_aguacate/configuracion_territorio.json",
    "Piloto_historico_aguacate/configuracion_piloto.json",
    "Delimitacion_rendimiento_aguacate/datos_agricolas.py",
    "ver_resultados .py",
]


def actualizar_proyecto(origen, destino):
    origen, destino = Path(origen).resolve(), Path(destino).resolve()
    if origen == destino or origen in destino.parents or destino in origen.parents:
        raise ValueError(
            "Extrae la versión limpia en una carpeta separada del proyecto que vas a actualizar."
        )
    if (
        not (destino / "configuracion.json").is_file()
        or not (destino / "prueba_ingesta.py").is_file()
    ):
        raise ValueError(
            "El destino debe ser la carpeta del proyecto anterior, con configuracion.json y prueba_ingesta.py."
        )
    cfg = leer_json(origen / "configuracion.json")
    anterior = leer_json(destino / "configuracion.json")
    # Respetar rutas y ajustes reales del equipo al actualizar desde el paquete.
    for k, v in anterior.items():
        if k in ["territorio", "piloto", "revision"] and isinstance(v, dict):
            if k == "territorio" and not v.get("cvegeo_seleccionados"):
                v = {a: b for a, b in v.items() if a != "cvegeo_seleccionados"}
            cfg.setdefault(k, {}).update(v)
        else:
            cfg[k] = v
    for ruta, seccion in [
        (
            "Delimitacion_rendimiento_aguacate/configuracion_territorio.json",
            "territorio",
        ),
        ("Piloto_historico_aguacate/configuracion_piloto.json", "piloto"),
    ]:
        p = destino / ruta
        if p.exists():
            valores = leer_json(p)
            if seccion == "territorio" and not valores.get("cvegeo_seleccionados"):
                valores = {
                    k: v for k, v in valores.items() if k != "cvegeo_seleccionados"
                }
            cfg[seccion].update(valores)
    respaldo = destino.parent / (
        destino.name + "_respaldo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    respaldo.mkdir()
    archivos = []
    for p in origen.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(origen)
        if rel.as_posix() == "credenciales.json" and (destino / rel).exists():
            continue
        if "resultados_territorio" in rel.parts:
            continue
        if "fuentes" in rel.parts and (destino / rel).exists():
            continue
        archivos.append(rel)
    # Respaldar únicamente archivos que van a ser cambiados, fuera del proyecto.
    for rel in sorted(set(archivos) | {Path(p) for p in OBSOLETOS}):
        p = destino / rel
        if p.is_file():
            copia = respaldo / rel
            copia.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, copia)
    for rel in archivos:
        p = destino / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if str(rel) == "configuracion.json":
            guardar_json(p, cfg)
        else:
            shutil.copy2(origen / rel, p)
    for nombre in OBSOLETOS:
        (destino / nombre).unlink(missing_ok=True)
    print("Proyecto actualizado:", destino)
    print("Respaldo de código y ajustes anteriores:", respaldo)
    print("Se conservaron los datos, pesos, resultados y entorno virtual del destino.")
    print(
        "Desde el destino: instala requirements.txt y ejecuta python ejecutar.py territorio para actualizar el mapa."
    )
