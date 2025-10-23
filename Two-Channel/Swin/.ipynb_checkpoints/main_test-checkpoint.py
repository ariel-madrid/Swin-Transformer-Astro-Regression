import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# Importar las funciones clave de tu proyecto
from config import get_config
from models import build_model

def predict(config, image_path, checkpoint_path):
    """
    Función principal para hacer una predicción en una sola imagen.
    Esta función está diseñada para ser autónoma y no depender de un entorno de entrenamiento.

    Args:
        config: Objeto de configuración (de get_config).
        image_path (str): Ruta a la imagen que se va a predecir (formato .npy).
        checkpoint_path (str): Ruta al archivo .pth del mejor checkpoint.
    """
    # --- 1. Configurar el dispositivo (usar GPU si está disponible) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- 2. Construir el modelo ---
    print(f"Construyendo el modelo '{config.MODEL.NAME}'...")
    model = build_model(config)
    model.to(device)

    # --- 3. Cargar los pesos del checkpoint ---
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    
    # Cargar el checkpoint en la CPU para evitar picos de memoria en GPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extraer el state_dict del modelo del checkpoint
    # Es la práctica recomendada guardar el state_dict del modelo base
    state_dict = checkpoint['model']

    # Manejar el prefijo 'module.' que añade DistributedDataParallel (DDP)
    # Esto hace que la carga sea robusta, sin importar si el checkpoint se guardó
    # desde un modelo envuelto en DDP o no.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Cargar los pesos en el modelo. Usamos strict=True para asegurar que todas las
    # claves coincidan, lo que confirma que estamos usando el modelo correcto.
    model.load_state_dict(new_state_dict, strict=True)
    
    print("¡Pesos del modelo cargados exitosamente!")
    
    # Liberar memoria
    del checkpoint, state_dict, new_state_dict
    torch.cuda.empty_cache()

    # --- 4. Poner el modelo en modo de evaluación ---
    # ¡CRUCIAL! Desactiva Dropout y ajusta el comportamiento de capas como BatchNorm.
    model.eval()

    # --- 5. Preparar la imagen de entrada ---
    print(f"Procesando la imagen: {image_path}")
    
    try:
        input_array = np.load(image_path) 
    except Exception as e:
        print(f"Error al cargar la imagen .npy: {e}")
        return

    # Convertir el array de numpy a un tensor de PyTorch
    # El modelo espera un lote (batch), así que añadimos una dimensión extra con unsqueeze(0)
    input_tensor = torch.from_numpy(input_array).float()
    input_tensor = input_tensor.unsqueeze(0) # [C, H, W] -> [B, C, H, W]
    
    # Mover el tensor de entrada al mismo dispositivo que el modelo
    input_tensor = input_tensor.to(device)
    print(f"Forma del tensor de entrada: {input_tensor.shape}")

    # --- 6. Realizar la predicción ---
    print("Realizando la predicción...")
    with torch.no_grad(): # Desactiva el cálculo de gradientes para eficiencia
        output = model(input_tensor)

    # Mover el resultado a la CPU y convertirlo a NumPy
    predicted_params = output.cpu().numpy()[0] # [0] para quitar la dimensión de lote

    # --- 7. Mostrar los resultados ---
    param_names = ["M_Disk", "gamma", "psi"]
    print("\n--- Resultados de la Predicción ---")
    for name, value in zip(param_names, predicted_params):
        print(f"  {name:<10}: {value:.6f}")
    
    return predicted_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script de inferencia para el modelo Swin Transformer.')
    parser.add_argument('--cfg', required=True, type=str, help='Ruta al archivo de configuración .yaml del modelo.')
    parser.add_argument('--image', required=True, type=str, help='Ruta a la imagen de entrada (archivo .npy).')
    parser.add_argument('--checkpoint', required=True, type=str, help='Ruta al checkpoint del modelo entrenado (.pth).')
    args = parser.parse_args()

    # Cargar la configuración
    config = get_config(args)

    # Lanzar la predicción
    predict(config, args.image, args.checkpoint)