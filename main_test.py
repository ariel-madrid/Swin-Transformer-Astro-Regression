import torch
import numpy as np
import argparse
from astropy.io import fits
from torchvision import transforms
from collections import OrderedDict

# Importar las funciones clave de tu proyecto
from config import get_config
from models import build_model

# =============================================================================
# --- NUEVA FUNCIÓN DE PRE-PROCESAMIENTO ---
# =============================================================================
def preprocess_fits_to_tensor(fits_path_740, fits_path_2067, image_stats_path, device):
    """
    Carga un par de archivos FITS, los normaliza usando las estadísticas del
    entrenamiento y los convierte en un tensor listo para el modelo.

    Args:
        fits_path_740 (str): Ruta al archivo FITS de la banda de 740um.
        fits_path_2067 (str): Ruta al archivo FITS de la banda de 2067um.
        image_stats_path (str): Ruta al archivo .npy que contiene la media y std del train set.
        device: El dispositivo de PyTorch (ej. "cuda" o "cpu") al que se moverá el tensor.

    Returns:
        torch.Tensor: Un tensor de forma [1, 2, H, W] listo para la inferencia.
    """
    print("--- Iniciando pre-procesamiento de archivos FITS ---")
    try:
        # 1. Cargar las estadísticas de normalización del entrenamiento
        image_stats = np.load(image_stats_path, allow_pickle=True).item()
        mean_per_channel = image_stats['mean']
        std_per_channel = image_stats['std']
        print(f"Estadísticas de normalización cargadas: Mean={mean_per_channel}, Std={std_per_channel}")

        # 2. Cargar los datos de las imágenes FITS
        print(f"Cargando FITS 740um desde: {fits_path_740}")
        with fits.open(fits_path_740, memmap=False) as hdul:
            img_b_740 = np.squeeze(hdul[0].data).astype(np.float32)
        
        print(f"Cargando FITS 2067um desde: {fits_path_2067}")
        with fits.open(fits_path_2067, memmap=False) as hdul:
            img_b_2067 = np.squeeze(hdul[0].data).astype(np.float32)

        # 3. Aplicar la misma normalización (Z-score) que en el entrenamiento
        print("Aplicando normalización estándar (Z-score)...")
        epsilon = 1e-9
        img_b_740_norm = (img_b_740 - mean_per_channel[0]) / (std_per_channel[0] + epsilon)
        img_b_2067_norm = (img_b_2067 - mean_per_channel[1]) / (std_per_channel[1] + epsilon)

        # 4. Apilar los dos canales para formar la imagen de 2 canales
        two_channels_img = np.stack([img_b_740_norm, img_b_2067_norm], axis=0)
        
        # 5. Convertir a tensor de PyTorch y preparar para el modelo
        input_tensor = torch.from_numpy(two_channels_img).float()
        input_tensor = input_tensor.unsqueeze(0)  # Añadir dimensión de lote: [C, H, W] -> [B, C, H, W]
        input_tensor = input_tensor.to(device)    # Mover al dispositivo correcto

        print(f"Pre-procesamiento completado. Forma del tensor final: {input_tensor.shape}")
        return input_tensor

    except FileNotFoundError as e:
        print(f"ERROR: No se pudo encontrar un archivo necesario: {e}")
        return None
    except Exception as e:
        print(f"ERROR durante el pre-procesamiento de FITS: {e}")
        return None

# =============================================================================
# --- FUNCIÓN DE PREDICCIÓN ---
# =============================================================================
def predict(config, fits_path_740, fits_path_2067, image_stats_path, checkpoint_path):
    """
    Función principal para hacer una predicción a partir de archivos FITS.
    """
    # --- 1. Configurar el dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- 2. Construir el modelo ---
    print(f"Construyendo el modelo '{config.MODEL.NAME}'...")
    model = build_model(config)
    model.to(device)

    # --- 3. Cargar los pesos del checkpoint ---
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("¡Pesos del modelo cargados exitosamente!")
    del checkpoint, state_dict, new_state_dict
    torch.cuda.empty_cache()
    model.eval()

    # --- 4. Pre-procesar las imágenes FITS para obtener el tensor de entrada ---
    input_tensor = preprocess_fits_to_tensor(fits_path_740, fits_path_2067, image_stats_path, device)
    if input_tensor is None:
        return # Salir si el pre-procesamiento falló

    # --- 5. Realizar la predicción ---
    print("\nRealizando la predicción...")
    with torch.no_grad():
        output = model(input_tensor)
    predicted_params_normalized = output.cpu().numpy()[0]

    # --- 6. Des-normalizar los resultados para obtener los valores físicos ---
    print("Des-normalizando las predicciones a valores físicos...")
    param_stats_path = os.path.join(os.path.dirname(image_stats_path), "param_minmax_stats_train_3params.npy")
    try:
        param_stats = np.load(param_stats_path, allow_pickle=True).item()
        
        param_names = ["mDisk", "gamma", "psi"]
        predicted_params_physical = {}
        for i, name in enumerate(param_names):
            stats = param_stats[name]
            norm_val = predicted_params_normalized[i]
            # Invertir la normalización: valor_fisico = valor_norm * rango + min
            physical_val = norm_val * stats['range'] + stats['min']
            predicted_params_physical[name] = physical_val
            
    except FileNotFoundError:
        print(f"\nADVERTENCIA: No se encontró el archivo de estadísticas de parámetros '{param_stats_path}'.")
        print("Se mostrarán solo los resultados normalizados (rango 0-1).")
        predicted_params_physical = None
    
    # --- 7. Mostrar los resultados ---
    print("\n--- Resultados de la Predicción ---")
    print("  Valores Normalizados (0-1):")
    for i, name in enumerate(param_names):
        print(f"    {name:<10}: {predicted_params_normalized[i]:.6f}")

    if predicted_params_physical:
        print("\n  Valores Físicos (Estimados):")
        for name, value in predicted_params_physical.items():
            print(f"    {name:<10}: {value:.6f}")
    
    return predicted_params_physical if predicted_params_physical else predicted_params_normalized


"""
Metodo de llamado al script (Ejemplo):

STATS_DIR="/home/aargomedo/TESIS/Preprocesar/img_stdnorm_3params_norm"

python main_test.py \
  --cfg configs/swin_tiny.yaml \
  --fits740 /home/aargomedo/TESIS/Data/fits/1234-740.fits \
  --fits2067 /home/aargomedo/TESIS/Data/fits/1234-2067.fits \
  --stats ${STATS_DIR}/image_stats_train.npy \
  --checkpoint /home/aargomedo/TESIS/Modelos/Two-Channel/swin_tiny/default/ckpt_epoch_47.pth
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script de inferencia para el modelo Swin Transformer a partir de archivos FITS.')
    parser.add_argument('--cfg', required=True, type=str, help='Ruta al archivo de configuración .yaml del modelo.')
    parser.add_argument('--fits740', required=True, type=str, help='Ruta al archivo FITS de entrada (banda 740um).')
    parser.add_argument('--fits2067', required=True, type=str, help='Ruta al archivo FITS de entrada (banda 2067um).')
    parser.add_argument('--stats', required=True, type=str, help='Ruta al archivo "image_stats_train.npy" generado durante el pre-procesamiento.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Ruta al checkpoint del modelo entrenado (.pth).')
    args = parser.parse_args()

    config = get_config(args)
    predict(config, args.fits740, args.fits2067, args.stats, args.checkpoint)