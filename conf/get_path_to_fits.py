import os

# Ruta base donde se encuentran las carpetas de datos
base_dir = "/home/aargomedo/TESIS/Simulate/files/FITS_Images_Gap"
output_file = "rutas_fits.txt"

# Abrimos el archivo para escribir las rutas
with open(output_file, "w") as f:
    # Iterar sobre todas las carpetas en la ruta base
    if os.path.isdir(base_dir):  # Verificar que la carpeta exista
        for file in os.listdir(base_dir):  # Listar archivos en la subcarpeta
            if file.endswith(".fits"):  # Filtrar archivos con extensión .fits
                full_path = os.path.join(base_dir, file)
                f.write(full_path + "\n")  # Guardar la ruta en el archivo

print(f"Rutas de archivos .fits guardadas en {output_file}.")

