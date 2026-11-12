"""Métricas físicas y productos de pronóstico; no usa objetivos para crear predicciones."""
from pathlib import Path
import numpy as np
import torch
from .common import BANDS, write_json


def reflectance(x, meta):
    shape = (1,6,1,1) if x.ndim == 4 else (6,1,1)
    mean = torch.tensor(meta["mean"],device=x.device,dtype=torch.float32).reshape(shape)
    std = torch.tensor(meta["std"],device=x.device,dtype=torch.float32).reshape(shape)
    return (x.float()*std+mean)/10000


def indices_numpy(image, valid):
    def ratio(a,b):
        den = image[a]+image[b]
        ok = valid & np.isfinite(den) & (den>1e-6)
        value = np.full(valid.shape,np.nan,np.float32)
        np.divide(image[a]-image[b],den,out=value,where=ok)
        return value
    return {"ndvi_b8a":ratio(3,2),"ndmi":ratio(3,4)}


class ForecastMetrics:
    def __init__(self):
        self.count = 0
        self.abs = {k:np.zeros(6,np.float64) for k in ["forecast","persistence"]}
        self.sq = {k:np.zeros(6,np.float64) for k in self.abs}
        self.idx = {i:{k:[0.0,0.0,0] for k in self.abs} for i in ["ndvi_b8a","ndmi"]}

    def update(self,pred,baseline,target,valid,meta):
        arrays = {"forecast":reflectance(pred,meta).detach().cpu().numpy(),
                  "persistence":reflectance(baseline,meta).detach().cpu().numpy()}
        y = reflectance(target,meta).detach().cpu().numpy()
        mask = valid.detach().cpu().numpy().astype(bool)
        self.count += int(mask.sum())
        for k,x in arrays.items():
            error = x-y
            self.abs[k] += (np.abs(error)*mask[:,None]).sum(axis=(0,2,3),dtype=np.float64)
            self.sq[k] += (error**2*mask[:,None]).sum(axis=(0,2,3),dtype=np.float64)
        for n in range(len(y)):
            truth = indices_numpy(y[n],mask[n])
            derived = {k:indices_numpy(x[n],mask[n]) for k,x in arrays.items()}
            for index in self.idx:
                # Misma máscara para comparar el modelo y la referencia.
                common = np.isfinite(truth[index])
                for k in arrays:
                    common &= np.isfinite(derived[k][index])
                for k in arrays:
                    error = derived[k][index][common]-truth[index][common]
                    totals = self.idx[index][k]
                    totals[0] += float(np.abs(error).sum(dtype=np.float64))
                    totals[1] += float((error**2).sum(dtype=np.float64))
                    totals[2] += int(common.sum())

    def result(self):
        if not self.count:
            raise ValueError("No hay píxeles válidos para evaluar el pronóstico.")
        result = {"bands":BANDS,"valid_pixels":self.count,"baseline":"last_valid_context_observation_per_pixel"}
        for k in self.abs:
            result[k] = {"mae_reflectance_per_band":(self.abs[k]/self.count).tolist(),
                         "rmse_reflectance_per_band":np.sqrt(self.sq[k]/self.count).tolist(),
                         "mean_mae_reflectance":float(self.abs[k].sum()/(self.count*6))}
            for index in self.idx:
                a,s,n = self.idx[index][k]
                result[k][index] = {"mae":a/n if n else None,"rmse":float(np.sqrt(s/n)) if n else None,"valid_pixels":n}
        base = result["persistence"]["mean_mae_reflectance"]
        result["skill_vs_persistence_mae"] = 1-result["forecast"]["mean_mae_reflectance"]/base if base>0 else None
        return result


def geotiff(path,images,names,sample,valid):
    import rasterio
    from rasterio.transform import from_bounds
    h,w = images.shape[-2:]
    profile = {"driver":"GTiff","height":h,"width":w,"count":len(names),"dtype":"float32",
               "crs":f"EPSG:{sample['epsg']}","transform":from_bounds(*sample["bbox"],w,h),
               "nodata":np.nan,"compress":"deflate"}
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(images.astype(np.float32))
        dst.write_mask(valid.astype(np.uint8)*255)
        for i,name in enumerate(names,1):
            dst.set_band_description(i,name)
        dst.update_tags(product="MODEL_FORECAST_NOT_SATELLITE_OBSERVATION",
                        issue_time=sample["issue_time"],forecast_start=sample["forecast_week"]["start"],
                        forecast_end=sample["forecast_week"]["end"])


def comparison_plot(path,pred,baseline,support,sample,target=None,target_valid=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = [("Última observación válida",baseline,support),
                ("Pronóstico estimado",pred,support)]
    if target is not None:
        datasets.append(("Observación posterior real",target,target_valid))
    else:
        datasets.append(("Cambio estimado",None,support))
    derived = [indices_numpy(x,m) if x is not None else None for _,x,m in datasets]
    fig,axes = plt.subplots(3,3,figsize=(11,9))
    for col,(title,image,valid) in enumerate(datasets):
        axes[0,col].set_title(title)
        if image is not None:
            rgb = np.clip(image[[2,1,0]].transpose(1,2,0)/.3,0,1)
            rgb[~valid]=.5
            axes[0,col].imshow(rgb)
            for row,key in [(1,"ndvi_b8a"),(2,"ndmi")]:
                plot=axes[row,col].imshow(derived[col][key],vmin=-1,vmax=1,cmap="RdYlGn")
                fig.colorbar(plot,ax=axes[row,col],fraction=.046,pad=.03)
        else:
            delta=np.mean(np.abs(pred-baseline),axis=0)
            delta=np.where(support,delta,np.nan)
            plot=axes[0,col].imshow(delta,vmin=0,vmax=.1,cmap="magma")
            fig.colorbar(plot,ax=axes[0,col],fraction=.046,pad=.03)
            for row,key in [(1,"ndvi_b8a"),(2,"ndmi")]:
                plot=axes[row,col].imshow(derived[1][key]-derived[0][key],vmin=-.3,vmax=.3,cmap="RdBu")
                fig.colorbar(plot,ax=axes[row,col],fraction=.046,pad=.03)
        for row in range(3):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    for row,label in enumerate(["RGB","NDVI con B8A","NDMI"]):
        axes[row,0].set_ylabel(label)
    fig.suptitle(f"Estimación: {sample['forecast_week']['start']} a {sample['forecast_week']['end']}\n"
                 "Gris / sin color: sin soporte válido; no es una observación satelital futura")
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def export_product(directory,pred,baseline,support,sample,metadata,target=None,target_valid=None):
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pred = pred.astype(np.float32).copy()
    baseline = baseline.astype(np.float32).copy()
    support = support.astype(bool)
    pred[:,~support]=np.nan
    baseline[:,~support]=np.nan
    indices = indices_numpy(pred,support)
    np.savez_compressed(directory/"estimacion.npz",reflectance=pred,support_from_history=support,
                        ndvi_b8a=indices["ndvi_b8a"],ndmi=indices["ndmi"])
    geotiff(directory/"bandas_estimadas.tif",pred,BANDS,sample,support)
    geotiff(directory/"indices_estimados.tif",np.stack(list(indices.values())),list(indices),sample,support)
    comparison_plot(directory/"comparacion.png",pred,baseline,support,sample,target,target_valid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(10,6))
    for i,ax in enumerate(axes.flat):
        plot=ax.imshow(pred[i],vmin=0,vmax=.5,cmap="gray")
        ax.set_title(BANDS[i]+" · reflectancia estimada");ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(plot,ax=ax,fraction=.046,pad=.03)
    fig.tight_layout();fig.savefig(directory/"seis_bandas.png",dpi=130);plt.close(fig)
    write_json(directory/"metadatos.json", {"product":"FORECAST_NOT_OBSERVATION","sample":sample,
               "bands":BANDS,"units":"surface_reflectance_estimate","index_convention":"NDVI=(B8A-B04)/(B8A+B04), NDMI=(B8A-B11)/(B8A+B11)",
               "support_meaning":"at_least_one_valid_observation_in_previous_four_weeks_not_confidence",
               "uncertainty":"not_calibrated_by_this_pilot",**metadata})
