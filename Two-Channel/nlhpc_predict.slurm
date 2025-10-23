#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J 2chan-six
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=gn005
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH --mail-user=ariel.argomedo@usach.cl
#SBATCH --mail-type=ALL
#SBATCH -t 2-0:0:0
#SBATCH -o logs/gn004_swin_%j.log
#SBATCH -e logs/gn004_swin_%j.log

# ------------------- Módulos -------------------
ml purge
ml Miniconda3
ml aocc/4.2.0
ml glibc/2.34-zen4-ed5siie

# ------------------- Conda ---------------------
eval "$(conda shell.bash hook)"
conda activate /home/aargomedo/miniconda3/envs/torch_rocm

# ------------------- Entorno PyTorch/ROCm -------
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=12345
export WORLD_SIZE=1
export NODE_RANK=$SLURM_NODEID

export OMP_NUM_THREADS=4
export OMP_PLACES=cores

# ROCm
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_MODE=1
export MIOPEN_DISABLE_FIND_DB=1
export MIOPEN_LOG_LEVEL=1
export HSA_ENABLE_SDMA=1

# PyTorch ROCm memory config
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# NCCL ROCm backend (rccl)
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

STATS_DIR_IMAGENES="/home/aargomedo/TESIS/Preprocesar/img_stdnorm_3params_norm_six"
STATS_DIR_PARAMETROS="/home/aargomedo/TESIS/Preprocesar/img_stdnorm_3params_norm_six" # Asumiendo que están en el mismo dir
MODEL_DIR="/home/aargomedo/TESIS/Modelos/Two-Channel/swin_tiny/default"
INPUT_DIR="/home/aargomedo/TESIS/Preprocesar/preprocessed_for_prediction" # Directorio con los .npy pre-procesados
CONFIG_FILE="/home/aargomedo/TESIS/Modelos/Two-Channel/Swin/configs/swin/swin_base_patch4_window7_224.yaml"

PYTHONWARNINGS="ignore" \
python3 /home/aargomedo/TESIS/Modelos/Two-Six/Swin/predict.py \
    --cfg "${CONFIG_FILE}" \
    --input_dir "${INPUT_DIR}" \
    --image_stats_path "${STATS_DIR_IMAGENES}/image_stats_train.npy" \
    --param_stats_path "${STATS_DIR_PARAMETROS}/param_minmax_stats_train_3params.npy" \
    --checkpoint "${MODEL_DIR}/ckpt_epoch_180.pth" \
    --output_csv resultados_finales.csv

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Predicción completada con éxito."
else
    echo "ERROR: El script de predicción falló con el código de salida $EXIT_CODE."
fi
