# predict.py (Versión Final)
import os
import torch
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import argparse
from collections import OrderedDict
import sys

# Importar las funciones clave de tu proyecto
from config import get_config
from models import build_model
from NpyDataset import NpyDataset
from torch.utils.data import DataLoader

def predict_on_new_data(config, model, data_loader):
    """
    Realiza predicciones sobre un conjunto de datos y devuelve los resultados.
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        # Asumimos que los .npy ya están pre-procesados, así que el Dataloader solo devuelve imágenes
        for images in tqdm(data_loader, desc="Realizando predicciones"):
            images = images.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                output = model(images)
            
            all_outputs.append(output.cpu())
            
    return torch.cat(all_outputs, dim=0)

def main():
    parser = argparse.ArgumentParser(description='Script de inferencia para estimar parámetros de nuevos discos.')
    parser.add_argument('--cfg', required=True, type=str, help='Ruta al archivo de configuración .yaml del modelo.')
    parser.add_argument('--input_dir', required=True, type=str, help='Directorio que contiene los nuevos archivos .npy pre-procesados.')
    parser.add_argument('--image_stats_path', required=True, type=str, help='Ruta al archivo .npy con las estadísticas de las IMÁGENES (mean/std).')
    parser.add_argument('--param_stats_path', required=True, type=str, help='Ruta al archivo .npy con las estadísticas de los PARÁMETROS (min/max/range).')
    parser.add_argument('--checkpoint', required=True, type=str, help='Ruta al checkpoint del modelo entrenado (.pth).')
    parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Ruta al archivo CSV de salida.')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del lote para la inferencia.')
    args = parser.parse_args()

    # --- 1. Cargar configuración y estadísticas ---
    config = get_config(args)
    
    try:
        image_stats = np.load(args.image_stats_path, allow_pickle=True).item()
        param_stats = np.load(args.param_stats_path, allow_pickle=True).item()
        param_names = list(param_stats.keys())
    except Exception as e:
        print(f"Error fatal al cargar los archivos de estadísticas: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Configurar y cargar el modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    config.defrost()
    config.MODEL.NUM_CLASSES = len(param_names)
    config.freeze()

    model = build_model(config)
    model.to(device)

    print(f"Cargando checkpoint desde: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # --- 3. Preparar DataLoader para los datos ya pre-procesados ---
    # Creamos una subclase simple de Dataset para leer solo la imagen del .npy
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, npy_dir):
            self.npy_files = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')])
        def __len__(self):
            return len(self.npy_files)
        def __getitem__(self, idx):
            # Carga el .npy y devuelve solo el tensor de la imagen
            data = np.load(self.npy_files[idx], allow_pickle=True).item()
            return torch.from_numpy(data['image']).float()

    new_dataset = PredictionDataset(args.input_dir)
    new_data_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.DATA.NUM_WORKERS)
    print(f"Se encontraron {len(new_dataset)} imágenes pre-procesadas para predecir.")

    # --- 4. Realizar Predicciones ---
    predictions_normalized = predict_on_new_data(config, model, new_data_loader)
    predictions_normalized = predictions_normalized.numpy()

    # --- 5. Des-normalizar y Guardar ---
    print("Des-normalizando las predicciones a valores físicos...")
    results_list = []
    file_names = [os.path.basename(f) for f in new_dataset.npy_files]

    for i, file_name in enumerate(file_names):
        result_row = {'file_name': file_name}
        for j, name in enumerate(param_names):
            stats_p = param_stats[name]
            norm_val = predictions_normalized[i, j]
            physical_val = norm_val * stats_p['range'] + stats_p['min']
            result_row[f'pred_{name}'] = physical_val
        results_list.append(result_row)

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n¡Predicciones completadas! Resultados guardados en '{args.output_csv}'")
        print("Primeras 5 predicciones:")
        print(results_df.head())
    else:
        print("\nNo se pudo procesar ningún disco.")

if __name__ == '__main__':
    main()
