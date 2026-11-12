from datetime import date, timedelta
import numpy as np
import torch

from .common import load_plan, work, write_json
from .data import cached_record, record_path
from .forecast_data import (build_feature_metadata, context_tensors, extract_spatial_features,
                            feature_path, forecast_work, make_sample, save_features, valid_feature)
from .forecast_train import load_saved_head
from .forecast_io import export_product, reflectance
from .model import WeeklyDataset, build_model
from .model import device_for, move


def prediction_samples(c, forecast_week=None, tile_id=None):
    p=load_plan(c)
    # Clave de lunes aunque la primera/última semana del año estén recortadas.
    monday=lambda value:date.fromisoformat(value)-timedelta(days=date.fromisoformat(value).weekday())
    lookup={monday(w["start"]):w for w in p["weeks"]}
    start=date.fromisoformat(forecast_week) if forecast_week else max(lookup)+timedelta(days=7)
    if start.weekday()!=0:raise ValueError("--forecast-week debe ser un lunes (AAAA-MM-DD).")
    target=lookup.get(start,{"start":start.isoformat(),"end":(start+timedelta(days=6)).isoformat()})
    selected=[t for t in p["tiles"] if t["selected"] and (tile_id is None or t["id"]==tile_id)]
    if not selected:raise ValueError("El tile-id no está entre los recuadros seleccionados.")
    samples,skipped=[],[]
    for tile in selected:
        context=[]
        for back in range(4,0,-1):
            week=lookup.get(start-timedelta(days=7*back))
            if week is None:break
            path=record_path(c,tile["id"],week)
            metadata=cached_record(path,p["fingerprint"])
            if metadata is None:break
            context.append((path.relative_to(work(c)).as_posix(),metadata))
        if len(context)!=4:
            skipped.append({"tile":tile["id"],"reason":"missing_one_of_previous_four_calendar_weeks"})
            continue
        samples.append(make_sample(c,p,tile,context,target))
    if not samples:
        raise ValueError(f"No hay cuatro semanas consecutivas disponibles antes de {start}. Revisa los huecos o elige --forecast-week.")
    return samples,skipped


def predict_forecast(c,forecast_week=None,tile_id=None,allow_cpu=False):
    # No se requiere ni se abre un archivo de imagen objetivo.
    samples,skipped=prediction_samples(c,forecast_week,tile_id)
    head,meta,best=load_saved_head(c)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head.to(device).eval();bundle=None;encoder=None
    products=[]
    with torch.inference_mode():
        for sample in samples:
            cache=feature_path(c,meta,sample)
            if valid_feature(cache,meta["encoder_signature"]):
                with np.load(cache,allow_pickle=False) as z:
                    features=torch.from_numpy(z["features"].astype(np.float32)).unsqueeze(0)
            else:
                if bundle is None:
                    encoder_device=device_for(allow_cpu)
                    bundle=build_model(c)
                    actual=build_feature_metadata(c,bundle,{"fingerprint":meta["dataset_fingerprint"]})
                    if actual["encoder_signature"]!=meta["encoder_signature"]:
                        raise ValueError("El codificador no coincide con el utilizado para entrenar el pronóstico.")
                    encoder=bundle.model.encoder.requires_grad_(False).to(encoder_device).eval()
                raw=WeeklyDataset([sample],work(c),bundle.mean,bundle.std)[0]
                batch=move({k:v.unsqueeze(0) for k,v in raw.items() if torch.is_tensor(v)},encoder_device)
                # Mismo redondeo float16 que el caché de entrenamiento.
                features=extract_spatial_features(encoder,batch).half().float().cpu()
                save_features(cache,features[0].numpy().astype(np.float16),meta["encoder_signature"])
            batch={k:v.unsqueeze(0) for k,v in context_tensors(c,sample,meta).items()}
            batch["features"]=features;batch=move(batch,device)
            pred=head(batch)
            out=forecast_work(c)/"predictions"/sample["forecast_week"]["start"]/sample["tile"]
            export_product(out,reflectance(pred,meta)[0].cpu().numpy(),
                           reflectance(batch["baseline"],meta)[0].cpu().numpy(),
                           batch["support"][0].cpu().numpy(),sample,
                           {"signature":best["signature"],"best_epoch":best["epoch"],"model":c["model"],
                            "encoder_frozen":True,"uses_future_weather":False})
            products.append(str(out))
            print(f"ESTIMACIÓN {sample['forecast_week']['start']} a {sample['forecast_week']['end']}: {out}",flush=True)
    write_json(forecast_work(c)/"last_prediction.json",{"products":products,"skipped":skipped,
               "forecast_week":samples[0]["forecast_week"],"best_epoch":best["epoch"]})
    if skipped:print(f"Se omitieron {len(skipped)} ventanas con semanas de contexto faltantes.")
    if best["epoch"]==0:print("La validación eligió persistencia: estas estimaciones mantienen la última observación válida.")
    return products
