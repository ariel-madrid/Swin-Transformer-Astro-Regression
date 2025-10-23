#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J 2chan
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=gn004
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=64GB
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

# ------------------- Ejecución --------------------
PYTHONWARNINGS="ignore" \


# Simplemente ejecuta el script con python
python3 /home/aargomedo/TESIS/Modelos/Optuna-Two-Channel/Swin/main.py \
  --cfg /ruta/a/tu/config.yaml \
  --data_path_train /home/aargomedo/TESIS/Preprocesar/img_optuna/train \
  --data_path_val /home/aargomedo/TESIS/Preprocesar/img_optuna/val \
  --n_trials 50 \
  --epochs_per_trial 40 \
  --output output_optuna
