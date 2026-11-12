from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from dotenv import load_dotenv
from sentinelhub import (BBox, CRS, DataCollection, MimeType, MosaickingOrder,
                         ResamplingType, SHConfig, SentinelHubCatalog, SentinelHubRequest)
from .common import BANDS, ROOT


def connect():
    load_dotenv(ROOT / ".env", override=False)
    conf = SHConfig()
    conf.sh_client_id = os.environ.get("CDSE_CLIENT_ID", "")
    conf.sh_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")
    if not conf.sh_client_id or not conf.sh_client_secret:
        raise ValueError("Completa CDSE_CLIENT_ID y CDSE_CLIENT_SECRET en .env (OAuth de Data Space).")
    conf.sh_base_url = "https://sh.dataspace.copernicus.eu"
    conf.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    collection = DataCollection.SENTINEL2_L2A.define_from("S2_CDSE_PRITHVI", service_url=conf.sh_base_url)
    return conf, collection


def search(c):
    from shapely.geometry import shape
    conf, collection = connect()
    query = SentinelHubCatalog(config=conf).search(
        collection, bbox=BBox(shape(c["aoi"]).bounds, CRS.WGS84),
        time=(c["data"]["start"], c["data"]["end"]),
        filter=f'eo:cloud_cover < {100*c["data"]["max_cloud"]:g}',
        filter_lang="cql2-text", limit=100)
    # Iteración completa: no se limita el año a la primera página de cien escenas.
    unique = {}
    for s in query:
        unique[s["id"]] = {"id": s["id"], "datetime": s["properties"]["datetime"],
                            "cloud": s["properties"].get("eo:cloud_cover", 100),
                            "geometry": s["geometry"]}
    return sorted(unique.values(), key=lambda s: (s["datetime"], s["id"]))


def evalscript(quality=False):
    bands = ["SCL", "dataMask"] if quality else BANDS
    units = "DN" if quality else "REFLECTANCE"
    sample_type = "UINT8" if quality else "FLOAT32"
    import json
    return ('//VERSION=3\nfunction setup(){return {input:[{bands:' + json.dumps(bands) +
            ',units:"' + units + '"}],output:{bands:' + str(len(bands)) +
            ',sampleType:"' + sample_type + '"}};}\nfunction evaluatePixel(s){return [' +
            ','.join('s.' + b for b in bands) + '];}')


class ImageClient:
    def __init__(self):
        self.config, self.collection = connect()

    def request(self, tile, scene, c, quality=False):
        dt = datetime.fromisoformat(scene["datetime"].replace("Z", "+00:00"))
        interval = ((dt - timedelta(minutes=10)).isoformat(), (dt + timedelta(minutes=10)).isoformat())
        sampling = ResamplingType.NEAREST if quality else ResamplingType.BILINEAR
        req = SentinelHubRequest(
            evalscript=evalscript(quality),
            input_data=[SentinelHubRequest.input_data(
                data_collection=self.collection, time_interval=interval,
                maxcc=c["data"]["max_cloud"], mosaicking_order=MosaickingOrder.LEAST_CC,
                upsampling=sampling, downsampling=sampling,
                other_args={"processing": {"harmonizeValues": True}})],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=BBox(tile["bbox"], CRS(c["grid_epsg"])),
            size=(c["data"]["size"], c["data"]["size"]), config=self.config)
        # Se pide el recuadro completo. No se pasa geometry=AOI.
        for attempt in range(3):
            try:
                result = req.get_data()[0]
                expected = (c["data"]["size"], c["data"]["size"], 2 if quality else 6)
                if result.shape != expected:
                    raise ValueError(f"Respuesta inesperada: {result.shape}, se esperaba {expected}.")
                return result
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def valid_mask(quality):
    # Vegetación, suelo y agua. Excluye nubes, sombras, nieve y píxeles sin dato.
    return (quality[..., 1] > 0) & np.isin(quality[..., 0], [4, 5, 6])
