"""Ejecuta el mismo notebook de revisión sin duplicar su procesamiento."""

import json
from pathlib import Path

from utilidades import cargar_configuracion


def ejecutar_revision(
    configuracion, fuente="todas", *, descargar_faltantes_sentinel=False
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cargar_configuracion(configuracion)
    notebook = Path(__file__).with_name("Revision_fuentes_aguacate_VSCode.ipynb")
    documento = json.loads(notebook.read_text(encoding="utf-8"))
    entorno = {
        "__name__": "revision_fuentes",
        "CONFIG_PROYECTO": cfg,
        "REVISION_OVERRIDES": {
            k: fuente == "todas" or fuente == f
            for k, f in [
                ("EJECUTAR_SENTINEL", "sentinel"),
                ("EJECUTAR_ERA5", "lluvia"),
                ("EJECUTAR_AGRICOLA", "agricola"),
            ]
        },
    }
    entorno["REVISION_OVERRIDES"]["DESCARGAR_FALTANTES_SENTINEL"] = (
        descargar_faltantes_sentinel
    )
    original = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    try:
        for numero, celda in enumerate(documento["cells"]):
            if celda["cell_type"] != "code":
                continue
            codigo = "".join(celda.get("source", []))
            if codigo.strip():
                print(f"Revisión: celda {numero + 1}")
                exec(
                    compile(codigo, f"{notebook.name}:celda_{numero + 1}", "exec"),
                    entorno,
                )
    finally:
        plt.show = original
    print("Revisión terminada. Resultados:", cfg["carpeta_datos"])
