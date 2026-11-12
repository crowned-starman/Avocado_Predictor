from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


@lru_cache(maxsize=16)
def transformer(src, dst):
    return Transformer.from_crs(src, dst, always_xy=True)


def projected(geometry, src, dst):
    return transform(geometry, transformer(src, dst).transform, interleaved=False)


def make_tiles(scenes, epsg, size=224, resolution=30, limit=8):
    """Cubre la unión completa de las escenas, sin recortar por el AOI de búsqueda."""
    if not scenes:
        raise ValueError("El catálogo no devolvió escenas. Revisa las fechas, nubes y credenciales.")
    union = unary_union([projected(shape(s["geometry"]), 4326, epsg) for s in scenes])
    side = size * resolution
    x0, y0, x1, y1 = union.bounds
    tiles = []
    for y in range(math.floor(y0 / side), math.ceil(y1 / side)):
        for x in range(math.floor(x0 / side), math.ceil(x1 / side)):
            cell = box(x * side, y * side, (x + 1) * side, (y + 1) * side)
            overlap = union.intersection(cell).area / cell.area
            if overlap <= 0:
                continue
            wgs = projected(cell, epsg, 4326)
            center = projected(cell.centroid, epsg, 4326)
            tiles.append({"id": f"epsg{epsg}_x{x}_y{y}", "bbox": list(cell.bounds),
                          "geometry": mapping(wgs), "lat": center.y, "lon": center.x,
                          "footprint_fraction": overlap, "selected": False})
    if limit is None or limit >= len(tiles):
        chosen = list(range(len(tiles)))
    else:
        # Muestreo espacial reproducible y distribuido, priorizando celdas bien cubiertas.
        pool = [i for i, t in enumerate(tiles) if t["footprint_fraction"] >= .99]
        if len(pool) < limit:
            pool = list(range(len(tiles)))
        centers = np.array([(tiles[i]["bbox"][0], tiles[i]["bbox"][1]) for i in pool])
        nearest_center = int(np.argmin(((centers - centers.mean(0)) ** 2).sum(1)))
        chosen_local = [nearest_center]
        distance = np.full(len(pool), np.inf)
        while len(chosen_local) < limit:
            distance = np.minimum(distance, ((centers - centers[chosen_local[-1]]) ** 2).sum(1))
            distance[chosen_local] = -1
            chosen_local.append(int(np.argmax(distance)))
        chosen = [pool[i] for i in chosen_local]
    for i in chosen:
        tiles[i]["selected"] = True
    return tiles


def plot_plan(plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(10, 7))
    for tile in plan["tiles"]:
        x0, y0, x1, y1 = tile["bbox"]
        selected = tile["selected"]
        ax.add_patch(Rectangle((x0 / 1000, y0 / 1000), (x1-x0)/1000, (y1-y0)/1000,
                              facecolor="#2474c7" if selected else "#edf0f4",
                              edgecolor="#a4adb5", linewidth=.25, alpha=.85))
    aoi = projected(shape(plan["aoi"]), 4326, plan["epsg"])
    ax.add_patch(Polygon(np.array(aoi.exterior.coords)/1000, fill=False, edgecolor="#188449", linewidth=2))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel="Este UTM (km)", ylabel="Norte UTM (km)",
           title="Cobertura de escenas: todas las celdas\nAzul: descarga seleccionada · Verde: AOI de búsqueda")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
