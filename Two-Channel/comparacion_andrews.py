# create_comparison_plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_multipanel_comparison(predictions_df, literature_df, param_map):
    """
    Crea una figura multi-panel con histogramas de las predicciones
    y superpone los valores de la literatura como líneas verticales.

    Args:
        predictions_df (pd.DataFrame): DataFrame con las predicciones del modelo.
        literature_df (pd.DataFrame): DataFrame con los valores de la literatura.
        param_map (dict): Un diccionario que mapea los parámetros a sus unidades y colores.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4) # Aumentar escala de fuente
    # Creamos una figura de 3x2. `constrained_layout` ayuda a evitar solapamientos.
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 22), constrained_layout=True)
    fig.suptitle('Comparación de Distribuciones de Parámetros Predichos vs. Literatura', fontsize=24, fontweight='bold')
    
    # Aplanamos el array de ejes para iterar fácilmente
    axes = axes.flatten()

    plot_idx = 0
    for pred_col, (lit_col, unit, color) in param_map.items():
        if pred_col not in predictions_df.columns or lit_col not in literature_df.columns:
            print(f"Advertencia: Saltando {pred_col}, columna no encontrada.")
            continue

        ax = axes[plot_idx]
        predicted_values = predictions_df[pred_col].dropna()
        literature_values = literature_df[lit_col].dropna()
        literature_names = literature_df['Name'].dropna()

        # Ajustar el número de bins y los límites del eje X dinámicamente
        if lit_col == 'mDisk':
            # Para M_disk, centramos el gráfico en nuestras predicciones
            bins = 25
            # Calculamos los límites basados en el 99% de nuestras predicciones para evitar outliers
            vmin, vmax = np.quantile(predicted_values, [0.005, 0.995])
            ax.set_xlim(vmin - 0.01, vmax + 0.01) # Añadir un pequeño margen
        else:
            # Para otros parámetros, usamos un número de bins estándar
            bins = 20

        # 1. Graficar el Histograma de tus predicciones (usando DENSIDAD)
        sns.histplot(predicted_values, bins=bins, color=color, alpha=0.6, 
                     edgecolor='black', ax=ax, stat='density', kde=False, label=f'Predicciones del Modelo (N={len(predicted_values)})')
        
        # 2. Superponer una curva de ajuste Gaussiana
        mu, std = norm.fit(predicted_values)
        xmin, xmax = ax.get_xlim()
        x_range = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x_range, mu, std)
        ax.plot(x_range, p, 'k-', linewidth=2.5, label=f'Ajuste Gaussiano (μ={mu:.3f}, σ={std:.3f})')

        # 3. Superponer los valores de la literatura como líneas verticales
        line_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
        for i, (name, value) in enumerate(zip(literature_names, literature_values)):
            # Solo mostrar la línea si cae dentro de los límites del gráfico
            if xmin <= value <= xmax:
                style = line_styles[i % len(line_styles)]
                ax.axvline(x=value, linestyle=style, linewidth=2, label=f'{name} = {value:.3f}')

        # 4. Títulos y Etiquetas
        param_label = lit_col.replace('mDisk', 'M_{disk}').replace('gamma', '\\gamma').replace('psi', '\\psi')
        ax.set_title(f'Histograma de ${param_label}$', fontsize=18, fontweight='bold')
        
        axis_label = f'${param_label}$'
        if unit:
            axis_label += f' [{unit}]'
        ax.set_xlabel(axis_label, fontsize=16)
        ax.set_ylabel('Densidad de Probabilidad', fontsize=16)
        ax.legend(fontsize='small', loc='upper right') # Ubicación de la leyenda
        ax.tick_params(axis='both', which='major', labelsize=12)

        plot_idx += 1

    # Apagar el último subplot si no se usa
    if plot_idx < len(axes):
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')

    # Guardar la figura
    plt.savefig('multipanel_comparison_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Gráfico multi-panel corregido guardado como 'multipanel_comparison_corrected.png'")

# =============================================================================
# --- EJECUCIÓN PRINCIPAL ---
# =============================================================================
if __name__ == '__main__':
    # --- 1. Crear DataFrame con los datos de Andrews et al. (2009), Tablas 4 y 5 ---
    andrews_lit_data = {
        'Name': ['AS 205', 'GSS 39', 'AS 209', 'DoAr 25', 'WaOph 6', 'VSSG 1', 'SR 21', 'WSB 60', 'DoAr 44'],
        'mDisk': [0.029, 0.143, 0.028, 0.136, 0.077, 0.029, 0.005, 0.021, 0.017],
        'gamma': [0.9, 0.7, 0.4, 0.9, 1.0, 0.8, 0.9, 0.8, 1.0],
        'Rc': [46, 198, 126, 80, 153, 33, 17, 31, 80],
        'H100': [19.6, 7.3, 13.3, 6.7, 4.4, 9.7, 7.7, 11.0, 3.5],
        'psi': [0.11, 0.08, 0.10, 0.15, 0.06, 0.08, 0.26, 0.13, 0.04],
        'incl': [25, 60, 38, 59, 39, 53, 22, 25, 45]
    }
    andrews_df = pd.DataFrame(andrews_lit_data)

    # --- 2. Cargar tus predicciones ---
    try:
        my_predictions = pd.read_csv('resultados_finales.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'resultados_finales.csv'.")
        exit()

    # --- 3. Calcular H100 a partir de tus predicciones de H30 ---
    if 'pred_H30' in my_predictions.columns and 'pred_psi' in my_predictions.columns:
        print("Calculando H100 a partir de las predicciones de H30 y psi...")
        my_predictions['pred_H100'] = my_predictions['pred_H30'] * ((100 / 30) ** my_predictions['pred_psi'])
        print("Columna 'pred_H100' creada con éxito.")
    else:
        print("Advertencia: No se encontraron las columnas 'pred_H30' o 'pred_psi' en tu archivo de predicciones.")

    # --- 4. Definir el mapeo de parámetros para los gráficos ---
    # (pred_col, lit_col, unidad, color)
    param_map = {
        'pred_Rc': ('Rc', 'UA', 'seagreen'),
        'pred_mDisk': ('mDisk', 'M$_{\\odot}$', 'mediumorchid'),
        'pred_gamma': ('gamma', '', 'gray'),
        'pred_H100': ('H100', 'UA', 'darkcyan'),
        'pred_psi': ('psi', '', 'indianred'),
        # 'pred_incl': ('incl', '$^{\\circ}$', 'darkblue') # Descomenta si quieres los 6
    }

    # --- 5. Generar la figura ---
    plot_multipanel_comparison(my_predictions, andrews_df, param_map)
