from __future__ import annotations
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .common import digest, read_json, write_json
from .forecast_data import (ForecastDataset, feature_path, forecast_work, load_feature_metadata,
                            load_forecast_manifest, settings, valid_feature)
from .forecast_net import forecast_loss, make_head
from .forecast_io import ForecastMetrics, export_product, reflectance
from .model import seed_all
from .model import autocast, move, save_checkpoint


def signature(c,manifest,meta):
    f={k:v for k,v in settings(c).items() if k!="epochs"}
    return digest({"schema":1,"dataset":manifest["fingerprint"],"features":meta["fingerprint"],
                   "forecast":f,"seed":c["seed"]})


def data_loaders(c,manifest,meta):
    result={}
    for name,rows in manifest["splits"].items():
        if not rows:
            raise ValueError(f"No hay parejas de pronóstico para {name}.")
        for s in rows:
            if not valid_feature(feature_path(c,meta,s),meta["encoder_signature"]):
                raise ValueError("Faltan características. Ejecuta pronostico.py features completo.")
        result[name]=DataLoader(ForecastDataset(c,rows,meta),batch_size=c["forecast"]["batch_size"],
                                 shuffle=False,num_workers=0)
    return result


def evaluate_head(head,loader,device,meta,amp=False,example_path=None,signature_value=None):
    head.eval();metrics=ForecastMetrics();num_total=den_total=0.0
    with torch.no_grad():
        for i,raw in enumerate(loader):
            batch=move(raw,device)
            with autocast(device,amp):
                pred=head(batch)
            num,den=forecast_loss(pred,batch)
            num_total+=num.item();den_total+=den.item()
            metrics.update(pred,batch["baseline"],batch["target"],batch["valid"],meta)
            if i==0 and example_path is not None:
                s=loader.dataset.samples[int(raw["index"][0])]
                export_product(example_path,reflectance(pred,meta)[0].cpu().numpy(),
                    reflectance(batch["baseline"],meta)[0].cpu().numpy(),batch["support"][0].cpu().numpy(),s,
                    {"signature":signature_value,"purpose":"held_out_example_with_observed_target"},
                    reflectance(batch["target"],meta)[0].cpu().numpy(),batch["valid"][0].cpu().numpy())
    result=metrics.result();result["masked_mse_normalized"]=num_total/den_total
    return result


def plot_curves(history,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r["epoch"] for r in history],[r["val_mse"] for r in history],"o-",label="Validación temporal")
    if len(history)>1:
        ax.plot([r["epoch"] for r in history[1:]],[r["train_mse"] for r in history[1:]],"o-",label="Entrenamiento")
    ax.axhline(history[0]["val_mse"],ls="--",color="gray",label="Persistencia (época cero)")
    ax.set(xlabel="Época",ylabel="MSE normalizado sobre píxeles válidos");ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def train_forecast(c,resume=False):
    f=settings(c);manifest=load_forecast_manifest(c);meta=load_feature_metadata(c,manifest)
    loaders=data_loaders(c,manifest,meta)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando solo ConvLSTM y salida espacial en {device}; Prithvi usa características congeladas.",flush=True)
    seed_all(c["seed"])
    head=make_head(c,meta).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=f["lr"],weight_decay=f["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and f["amp"])
    directory=forecast_work(c)/"training";directory.mkdir(exist_ok=True)
    sig=signature(c,manifest,meta);last=directory/"last.pt"
    if last.exists() and not resume:
        raise ValueError("Ya existe entrenamiento de pronóstico. Usa --resume o conserva aparte forecast/training.")
    if resume:
        cp=torch.load(last,map_location="cpu",weights_only=True)
        if cp["signature"]!=sig:
            raise ValueError("Checkpoint de otra configuración o conjunto. Solo epochs puede ampliarse.")
        head.load_state_dict(cp["head"]);optimizer.load_state_dict(cp["optimizer"]);scaler.load_state_dict(cp["scaler"])
        torch.set_rng_state(cp["torch_rng"])
        if device.type=="cuda" and cp["cuda_rng"]:torch.cuda.set_rng_state_all(cp["cuda_rng"])
        history=cp["history"];best_score=cp["best_score"];bad=cp["bad_epochs"];start=cp["epoch"]+1
    else:
        base=evaluate_head(head,loaders["val"],device,meta,f["amp"])
        best_score=base["masked_mse_normalized"];bad=0;start=1
        history=[{"epoch":0,"train_mse":None,"val_mse":best_score}]
        save_checkpoint(directory/"best.pt",{"signature":sig,"epoch":0,"head":head.state_dict(),"val":base,
                        "feature_metadata":meta,"forecast_settings":f})
        write_json(directory/"baseline_validation.json",base)
    write_json(directory/"run.json",{"signature":sig,"model":c["model"],"forecast":f,
               "parameters_trained":sum(p.numel() for p in head.parameters()),
               "prithvi_frozen":True,"feature_metadata":meta,"device":str(device),"torch":str(torch.__version__)})
    if start>f["epochs"]:
        print("Las épocas configuradas ya están completas.");return
    for epoch in range(start,f["epochs"]+1):
        gen=torch.Generator().manual_seed(c["seed"]+epoch)
        loader=DataLoader(loaders["train"].dataset,batch_size=f["batch_size"],shuffle=True,generator=gen,num_workers=0)
        head.train();optimizer.zero_grad(set_to_none=True);num_total=den_total=0.0
        for step,raw in enumerate(loader):
            batch=move(raw,device)
            with autocast(device,f["amp"]):
                pred=head(batch);num,den=forecast_loss(pred,batch)
                group_start=(step//f["accumulate"])*f["accumulate"]
                group=min(f["accumulate"],len(loader)-group_start)
                loss=num/den/group
            if not torch.isfinite(loss):raise ValueError("Pérdida de pronóstico no finita.")
            scaler.scale(loss).backward();num_total+=num.detach().item();den_total+=den.item()
            if (step+1)%f["accumulate"]==0 or step+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(head.parameters(),f["clip_grad"])
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if (step+1)%20==0 or step+1==len(loader):
                print(f"Pronóstico época {epoch}/{f['epochs']} · lote {step+1}/{len(loader)}",flush=True)
        val=evaluate_head(head,loaders["val"],device,meta,f["amp"])
        score=val["masked_mse_normalized"]
        history.append({"epoch":epoch,"train_mse":num_total/den_total,"val_mse":score})
        if score<best_score:
            best_score=score;bad=0
            save_checkpoint(directory/"best.pt",{"signature":sig,"epoch":epoch,"head":head.state_dict(),"val":val,
                            "feature_metadata":meta,"forecast_settings":f})
        else:bad+=1
        save_checkpoint(last,{"signature":sig,"epoch":epoch,"head":head.state_dict(),"optimizer":optimizer.state_dict(),
             "scaler":scaler.state_dict(),"best_score":best_score,"bad_epochs":bad,"history":history,
             "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else []})
        write_json(directory/"history.json",history)
        with (directory/"history.csv").open("w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=["epoch","train_mse","val_mse"]);writer.writeheader();writer.writerows(history)
        plot_curves(history,directory/"curvas.png")
        print(f"Validación: {score:.6f}; mejor: {best_score:.6f}",flush=True)
        if bad>=f["patience"]:
            print("Parada temprana: no mejoró la validación.");break
    best=torch.load(directory/"best.pt",map_location="cpu",weights_only=True)
    head.load_state_dict(best["head"])
    result=evaluate_head(head,loaders["val"],device,meta,f["amp"],directory/"ejemplo_validacion",sig)
    write_json(directory/"best_validation.json",{"best_epoch":best["epoch"],**result})
    print(f"Mejor época {best['epoch']}; 0 significa que la persistencia ganó. Ejecuta pronostico.py evaluate.")


def load_saved_head(c,manifest=None):
    directory=forecast_work(c)/"training"
    best=torch.load(directory/"best.pt",map_location="cpu",weights_only=True)
    meta=best["feature_metadata"]
    if meta["model"]!=c["model"]:
        raise ValueError("La configuración apunta a otro Prithvi.")
    if manifest is not None and best["signature"]!=signature(c,manifest,load_feature_metadata(c,manifest)):
        raise ValueError("El checkpoint no corresponde a las parejas/configuración actuales.")
    if {k:v for k,v in best["forecast_settings"].items() if k!="epochs"} != {k:v for k,v in settings(c).items() if k!="epochs"}:
        raise ValueError("Los parámetros actuales no coinciden con el modelo guardado.")
    head=make_head(c,meta);head.load_state_dict(best["head"])
    return head,meta,best


def evaluate_forecast(c):
    manifest=load_forecast_manifest(c)
    head,meta,best=load_saved_head(c,manifest)
    path=forecast_work(c)/"training/test_metrics.json"
    if path.exists():
        old=read_json(path)
        if old["signature"]==best["signature"] and old["best_epoch"]==best["epoch"]:
            print(old);return old
        raise ValueError("El modelo cambió después de evaluar el conjunto reservado; conserva aparte test_metrics.json.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");head.to(device)
    loader=data_loaders(c,manifest,meta)["test"]
    result=evaluate_head(head,loader,device,meta,c["forecast"]["amp"],
                         forecast_work(c)/"training/ejemplo_prueba",best["signature"])
    result.update({"signature":best["signature"],"best_epoch":best["epoch"],
                   "meaning":"Next-week reflectance forecast; no avocado yield or causal effect is evaluated."})
    write_json(path,result);print(result);return result
