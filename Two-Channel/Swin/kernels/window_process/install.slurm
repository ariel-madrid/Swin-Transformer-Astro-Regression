#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J install
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=ariel.argomedo@usach.cl
#SBATCH --mail-type=ALL
#SBATCH -o install_%j.err.out
#SBATCH -e install_%j.err.out


ml Miniconda3
ml purge
ml aocc/4.2.0
eval "$(conda shell.bash hook)"
conda activate /home/aargomedo/miniconda3/envs/swin_slurm

# ----------------Modulos----------------------------
# ----------------Comando--------------------------
python3 setup.py install --user

