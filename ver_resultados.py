"""Visualiza resultados de ingesta guardados. Ejecutar mediante ejecutar.py ver-resultados."""

from __future__ import annotations

from utilidades import leer_json

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def leer_json_si_existe(ruta):
    return leer_json(ruta) if ruta.is_file() else {}


def leer_salida_tamrf(ruta, campo):
    with np.load(ruta, allow_pickle=False) as z:
        datos = np.array(z[campo], dtype=np.float32)
        bandas = z["bandas"].astype(str).tolist()
        fechas = z["fecha"].astype(str).tolist() if "fecha" in z.files else []
    if datos.ndim != 5 or datos.shape[0] != 1 or datos.shape[2] != len(bandas):
        raise ValueError(f"Forma inesperada en {ruta.name}: {datos.shape}")
    if not np.isfinite(datos).all():
        raise ValueError(
            f"{ruta.name} contiene NaN o infinitos; revisa el informe del modelo."
        )
    return datos[0], bandas, fechas


def escala_rgb(referencia, bandas):
    indices = [bandas.index(b) for b in ("B04", "B03", "B02")]
    rgb = referencia[indices].transpose(1, 2, 0)
    limites = np.percentile(rgb, [2, 98], axis=(0, 1))
    return indices, limites


def rgb(datos, indices, limites):
    x = datos[indices].transpose(1, 2, 0)
    return np.clip((x - limites[0]) / np.maximum(limites[1] - limites[0], 1e-5), 0, 1)


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Creado:", ruta.name)


def salida_utilizable(informe, etapa, salida, entrada, avisos):
    if not salida.is_file():
        return False
    estado = informe.get("etapas", {}).get(etapa, {}).get("estado")
    if estado in ("error", "omitido"):
        avisos.append(
            f"{etapa}: el último informe indica {estado}; no se muestra una posible salida anterior."
        )
        return False
    if entrada.is_file() and salida.stat().st_mtime < entrada.stat().st_mtime:
        avisos.append(
            f"{etapa}: la entrada preparada es más reciente que la salida. Ejecuta probar para actualizarla."
        )
        return False
    return True


def visualizar_sentinel(base, destino, informe, avisos):
    entrada = base / "preparados" / "sentinel2_entrada.npz"
    metadatos = leer_json_si_existe(base / "preparados" / "sentinel2_metadatos.json")
    predicho = base / "tamrf" / "salida_tamrf.npz"
    observado, bandas, indices, limites = None, None, None, None
    if entrada.is_file():
        observado, bandas, _ = leer_salida_tamrf(entrada, "reflectancia")
        indices, limites = escala_rgb(observado[-1], bandas)
        fecha = (metadatos.get("fechas_utc") or ["fecha no disponible"])[-1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb(observado[-1], indices, limites))
        ax.set_title(
            f"Entrada real preparada · {fecha}\n{observado.shape[-1]} × {observado.shape[-2]} píxeles"
        )
        ax.axis("off")
        guardar(fig, destino / "01_entrada_sentinel2.png")
        avisos.append(
            f"Sentinel-2 preparado: [1, {observado.shape[0]}, {observado.shape[1]}, {observado.shape[2]}, {observado.shape[3]}]."
        )
    if not salida_utilizable(informe, "ingesta_tamrf", predicho, entrada, avisos):
        avisos.append(
            "TAMRF: todavía no hay una salida actual disponible para visualizar."
        )
        return
    estimado, bandas_pred, fechas_pred = leer_salida_tamrf(
        predicho, "reflectancia_estimada"
    )
    if bandas is not None and bandas != bandas_pred:
        raise ValueError("El orden de las bandas de entrada y salida no coincide.")
    bandas = bandas_pred
    if indices is None:
        indices, limites = escala_rgb(estimado[0], bandas)
    for t in range(estimado.shape[0]):
        fecha = (
            fechas_pred[t]
            if t < len(fechas_pred)
            else str(metadatos.get("consulta_utc", "sin fecha"))
        )
        paneles = 2 if observado is not None else 1
        fig, axes = plt.subplots(1, paneles, figsize=(6 * paneles, 6), squeeze=False)
        if observado is not None:
            axes[0, 0].imshow(rgb(observado[-1], indices, limites))
            fecha_real = (metadatos.get("fechas_utc") or ["sin fecha"])[-1]
            axes[0, 0].set_title(f"Última entrada observada\n{fecha_real}")
        axes[0, -1].imshow(rgb(estimado[t], indices, limites))
        axes[0, -1].set_title(f"Estimación TAMRF\n{fecha}")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(
            "Misma escala RGB · No es una comparación contra una observación real futura",
            fontsize=11,
        )
        guardar(fig, destino / f"02_tamrf_comparacion_{t + 1:02d}.png")

        fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
        registros = []
        for i, ax in enumerate(axes.flat):
            valores = estimado[t, i]
            referencia = observado[-1, i] if observado is not None else valores
            lo, hi = np.percentile(referencia, [2, 98])
            if hi <= lo:
                hi = lo + 0.01
            im = ax.imshow(valores, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(bandas[i])
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.7, label="Reflectancia estimada")
            fila = {
                "fecha": fecha,
                "banda": bandas[i],
                "minimo_estimado": float(valores.min()),
                "media_estimada": float(valores.mean()),
                "maximo_estimado": float(valores.max()),
            }
            if observado is not None and observado[-1, i].shape == valores.shape:
                fila["media_ultima_observacion"] = float(observado[-1, i].mean())
                fila["cambio_medio_estimado_no_error"] = float(
                    (valores - observado[-1, i]).mean()
                )
            registros.append(fila)
        fig.suptitle(f"Diez bandas estimadas · {fecha} · Escala individual por banda")
        guardar(fig, destino / f"03_tamrf_bandas_{t + 1:02d}.png")
        pd.DataFrame(registros).to_csv(
            destino / f"estadisticas_tamrf_{t + 1:02d}.csv", index=False
        )
    avisos.append(
        f"TAMRF: {estimado.shape[0]} fecha(s) estimada(s), diez bandas y valores finitos."
    )
    avisos.append(
        "Los cambios respecto a la última entrada NO son errores: corresponden a fechas distintas. No se calculó precisión."
    )


def visualizar_lluvia(base, destino, informe, avisos):
    entrada = base / "preparados" / "chronos2_entrada.csv"
    salida = base / "chronos2" / "salida_chronos2.csv"
    if not entrada.is_file():
        avisos.append(
            "ERA5-Land: falta la serie preparada; todavía no hay gráfica de lluvia."
        )
        return
    historial = pd.read_csv(entrada, dtype={"item_id": str}, parse_dates=["timestamp"])
    requeridas = {"item_id", "timestamp", "target"}
    if not requeridas.issubset(historial.columns):
        raise ValueError("El CSV de entrada de Chronos-2 tiene columnas inesperadas.")
    hay_pred = salida_utilizable(informe, "ingesta_chronos2", salida, entrada, avisos)
    pronostico = (
        pd.read_csv(salida, dtype={"item_id": str}, parse_dates=["timestamp"])
        if hay_pred
        else None
    )
    if pronostico is not None and not {
        "item_id",
        "timestamp",
        "predictions",
        "0.1",
        "0.5",
        "0.9",
    }.issubset(pronostico.columns):
        raise ValueError("El CSV de salida de Chronos-2 tiene columnas inesperadas.")
    ids = historial.item_id.unique().tolist()
    # Una o varias series por celda, divididas en páginas de cuatro.
    for inicio in range(0, len(ids), 4):
        pagina = ids[inicio : inicio + 4]
        fig, axes = plt.subplots(
            len(pagina), 1, figsize=(10, 4 * len(pagina)), squeeze=False
        )
        for ax, nombre in zip(axes.flat, pagina):
            h = historial.loc[historial.item_id == nombre].sort_values("timestamp")
            ax.plot(
                h.timestamp,
                h.target,
                "o-",
                color="#176b87",
                label="Historial observado",
            )
            if pronostico is not None:
                p = pronostico.loc[pronostico.item_id == nombre].sort_values(
                    "timestamp"
                )
                if len(p):
                    ax.plot(
                        pd.concat([h.timestamp.tail(1), p.timestamp]),
                        pd.concat([h.target.tail(1), p["0.5"]]),
                        "--",
                        color="#d46b08",
                        alpha=0.6,
                    )
                    ax.plot(
                        p.timestamp,
                        p["0.5"],
                        "o",
                        color="#d46b08",
                        label="Estimación central: cuantil 0.5",
                    )
                    ax.errorbar(
                        p.timestamp,
                        p["0.5"],
                        yerr=np.vstack(
                            [
                                (p["0.5"] - p["0.1"]).clip(lower=0),
                                (p["0.9"] - p["0.5"]).clip(lower=0),
                            ]
                        ),
                        fmt="none",
                        ecolor="#d46b08",
                        capsize=6,
                        label="Cuantiles 0.1–0.9 del modelo",
                    )
                    for fecha, central in zip(p.timestamp, p["0.5"]):
                        ax.annotate(
                            f"{central:.2f} mm",
                            (fecha, central),
                            xytext=(5, 9),
                            textcoords="offset points",
                        )
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set_title(
                f"{nombre} · Fechas de los datos, no necesariamente la fecha actual"
            )
            ax.set_ylabel("Precipitación (mm/día)")
            ax.set_xlabel("Fecha UTC")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        guardar(fig, destino / f"04_lluvia_{inicio // 4 + 1:02d}.png")
    if pronostico is not None:
        pronostico.to_csv(destino / "lluvia_predicha_para_revisar.csv", index=False)
        avisos.append(
            f"Chronos-2: {len(pronostico)} fila(s) de pronóstico; conserva salida cruda y cuantiles."
        )
        if (pronostico[["predictions", "0.1", "0.5", "0.9"]] < 0).any().any():
            avisos.append(
                "Hay salidas o cuantiles negativos: el modelo no impone el límite físico de lluvia. La columna prediccion_mm_no_negativa limita solo la central a cero."
            )
    else:
        avisos.append(
            "Chronos-2: se grafica solo el historial; falta ejecutar la prueba del modelo o actualizar su salida."
        )
    avisos.append(
        "Los cuantiles 0.1–0.9 describen un intervalo nominal del 80 % del modelo, cuya cobertura no se ha validado localmente."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("configuracion.json")
    )
    parser.add_argument(
        "--no-abrir",
        action="store_true",
        help="No abrir la carpeta de PNG al terminar en Windows",
    )
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = leer_json_si_existe(config)
    if not cfg:
        raise FileNotFoundError(
            "Coloca ver_resultados.py junto a configuracion.json, o usa --config RUTA."
        )
    salida = Path(cfg.get("carpeta_salida", "resultados")).expanduser()
    base = salida if salida.is_absolute() else config.parent / salida
    if not base.is_dir():
        raise FileNotFoundError(
            f"No existe {base}. Ejecuta primero prueba_ingesta.py preparar."
        )
    destino = base / "visualizaciones" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destino.mkdir(parents=True, exist_ok=True)
    informe = leer_json_si_existe(base / "informe_ingesta.json")
    avisos = [
        "VISOR DE ARCHIVOS GUARDADOS. No se ejecutaron modelos ni consultas de datos.",
        "La prueba de ingesta comprueba recepción y salida; no acredita precisión.",
        f"Última acción registrada: {informe.get('accion', 'sin informe')}",
    ]
    for nombre, datos in informe.get("etapas", {}).items():
        avisos.append(f"{nombre}: {datos.get('estado', 'sin estado')}")
    errores = []
    for nombre, funcion in [
        ("Sentinel-2 / TAMRF", visualizar_sentinel),
        ("ERA5-Land / Chronos-2", visualizar_lluvia),
    ]:
        try:
            funcion(base, destino, informe, avisos)
        except Exception as error:
            mensaje = f"ERROR al visualizar {nombre}: {error}"
            avisos.append(mensaje)
            errores.append(mensaje)
    avisos.append(
        "Estos PNG pertenecen a esta ejecución del visor. Los datos y los pesos originales no se modifican."
    )
    (destino / "COMO_INTERPRETAR.txt").write_text("\n".join(avisos), encoding="utf-8")
    print("\n" + "\n".join(avisos))
    print("\nAbre los PNG de:", destino.resolve())
    if os.name == "nt" and not args.no_abrir:
        try:
            os.startfile(str(destino.resolve()))
        except OSError:
            pass
    return 1 if errores else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print("ERROR:", error)
        raise SystemExit(1)
