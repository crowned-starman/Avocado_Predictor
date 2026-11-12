from __future__ import annotations

import argparse
import os
import platform
from .common import ROOT, load_config


def doctor(c):
    import torch
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    print("Credenciales OAuth configuradas:", bool(os.getenv("CDSE_CLIENT_ID") and os.getenv("CDSE_CLIENT_SECRET")))
    print("Modelo:", c["model"]["repo"], "| commit:", c["model"]["revision"])
    print("Datos:", c["data"]["start"], "a", c["data"]["end"], "| límite espacial:", c["data"]["max_spatial_tiles"])
    print("Salida:", c["_work"])
    print("No se descargaron imágenes ni pesos; doctor solo inspecciona el entorno.")


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2: pronóstico semanal con Prithvi 600M-TL y ConvLSTM.")
    parser.add_argument("command", choices=["doctor","inspect-model","plan","download","prepare","features","train","evaluate","predict"])
    parser.add_argument("--config", default=str(ROOT/"configs/piloto_2025.json"))
    parser.add_argument("--max-records",type=int,help="download/features: máximo de registros nuevos en esta ejecución.")
    parser.add_argument("--resume",action="store_true",help="train: continuar last.pt desde la última época completa.")
    parser.add_argument("--allow-cpu",action="store_true",help="Permite ejecutar Prithvi en CPU para depuración; requiere RAM y puede ser lento.")
    parser.add_argument("--forecast-week",help="predict: lunes de la semana a estimar, AAAA-MM-DD.")
    parser.add_argument("--tile-id",help="predict: identificador de una ventana seleccionada.")
    args=parser.parse_args()
    try:
        if args.max_records is not None and (args.command not in {"download","features"} or args.max_records<1):
            raise ValueError("--max-records requiere download o features y un valor positivo.")
        if args.resume and args.command!="train":
            raise ValueError("--resume solo corresponde a train.")
        if (args.forecast_week or args.tile_id) and args.command!="predict":
            raise ValueError("--forecast-week y --tile-id solo corresponden a predict.")
        c=load_config(args.config)
        if args.command=="doctor":
            doctor(c)
        elif args.command=="inspect-model":
            from .model import inspect_model
            inspect_model(c)
        elif args.command=="plan":
            from .data import create_plan
            create_plan(c)
        elif args.command=="download":
            from .data import download
            download(c,args.max_records)
        elif args.command=="prepare":
            from .forecast_data import prepare_forecast
            prepare_forecast(c)
        elif args.command=="features":
            from .forecast_data import cache_features
            cache_features(c,args.allow_cpu,args.max_records)
        elif args.command=="train":
            from .forecast_train import train_forecast
            train_forecast(c,args.resume)
        elif args.command=="evaluate":
            from .forecast_train import evaluate_forecast
            evaluate_forecast(c)
        else args.command=="predict":
            from .forecast_predict import predict_forecast
            predict_forecast(c,args.forecast_week,args.tile_id,args.allow_cpu)
    except (ValueError,FileNotFoundError) as error:
        parser.exit(1,f"\nRevisa: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130,"\nInterrumpido. Download reutiliza lo terminado; train --resume recupera la última época completa.\n")
