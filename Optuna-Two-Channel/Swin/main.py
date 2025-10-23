# optimize_hyperparams.py
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from config import get_config
from logger import create_logger
from NpyDataset import NpyDataset
from utils import NativeScalerWithGradNormCount
import argparse
from BuildModel import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import optuna
import sys

# --- Lista de Parámetros Global ---
param_names = ['Rc','H30','incl','mDisk', 'gamma', 'psi'] 

# =============================================================================
# --- FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN ---
# =============================================================================

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, logger, trial_num):
    model.train()
    loss_meter = AverageMeter()
    loss_scaler = NativeScalerWithGradNormCount()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = torch.nn.functional.mse_loss(outputs, targets)
        
        loss_for_scaler = loss / config.TRAIN.ACCUMULATION_STEPS
        loss_scaler(loss_for_scaler, optimizer, clip_grad=config.TRAIN.CLIP_GRAD, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        lr_scheduler.step_update((epoch * len(data_loader) + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_meter.update(loss.item(), targets.size(0))
    
    logger.info(f"T{trial_num} E{epoch} | Train Loss: {loss_meter.avg:.6f}")
    return loss_meter.avg

@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    all_outputs, all_targets = [], []
    for images, target in data_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            output = model(images)
        all_outputs.append(output)
        all_targets.append(target)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    return torch.mean((all_outputs - all_targets)**2).item()

# =============================================================================
# --- FUNCIÓN OBJECTIVE PARA OPTUNA ---
# =============================================================================
def objective(trial, args, config_base):
    config = config_base.clone()
    config.defrost()
    
    # --- Espacio de Búsqueda de Hiperparámetros Sugerido ---
    config.TRAIN.BASE_LR = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    config.TRAIN.WEIGHT_DECAY = trial.suggest_float("weight_decay", 0.01, 0.2)
    config.MODEL.DROP_RATE = trial.suggest_float("drop_rate", 0.1, 0.5)
    config.MODEL.DROP_PATH_RATE = trial.suggest_float("drop_path_rate", 0.1, 0.3)
    config.TRAIN.CLIP_GRAD = trial.suggest_float("clip_grad", 0.5, 5.0)
    config.TRAIN.WARMUP_EPOCHS = trial.suggest_int("warmup_epochs", 5, 15)
    
    config.MODEL.NUM_CLASSES = len(param_names)
    config.TRAIN.EPOCHS = args.epochs_per_trial 
    config.OUTPUT = os.path.join(args.output, f"trial_{trial.number}")
    os.makedirs(config.OUTPUT, exist_ok=True)
    config.freeze()

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"trial_{trial.number}")
    logger.info(f"--- Iniciando Trial {trial.number} ---")
    logger.info(f"Hiperparámetros: {trial.params}")

    train_dataset = NpyDataset(args.data_path_train, transform=None) 
    val_dataset = NpyDataset(args.data_path_val, transform=None)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, pin_memory=True)

    model = build_model(config).cuda()
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    
    min_val_loss = float('inf')
    for epoch in range(config.TRAIN.EPOCHS):
        train_one_epoch(config, model, train_loader, optimizer, epoch, lr_scheduler, logger, trial.number)
        val_loss = validate(val_loader, model)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        
        logger.info(f"T{trial.number} E{epoch} | Val Loss: {val_loss:.6f} (Mejor: {min_val_loss:.6f})")
        
        trial.report(min_val_loss, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} podado por Optuna.")
            raise optuna.exceptions.TrialPruned()

    return min_val_loss

# =============================================================================
# --- SCRIPT PRINCIPAL ---
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Optuna Hyperparameter Search Script - Single GPU')
    parser.add_argument('--cfg', required=True, type=str, help='Path to base config file.')
    parser.add_argument('--data_path_train', type=str, default="/home/aargomedo/TESIS/Preprocesar/img_optuna/train", help='Path to a subset of training data.')
    parser.add_argument('--data_path_val', type=str, default="/home/aargomedo/TESIS/Preprocesar/img_optuna/val", help='Path to a subset of validation data.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials to run.')
    parser.add_argument('--epochs_per_trial', type=int, default=40, help='Number of epochs to train for each trial.')
    parser.add_argument('--output', default='optuna_output_single_gpu', type=str, help='Output directory for Optuna results.')
    
    args, _ = parser.parse_known_args()

    cudnn.benchmark = True
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config_base = get_config(args)
    
    os.makedirs(args.output, exist_ok=True)
    study_storage = f"sqlite:///{os.path.join(args.output, 'optuna_study.db')}"
    study_name = "swin_hpo_6params_2ch" 
    
    print(f"Iniciando/Reanudando estudio Optuna: {study_name}")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15) 
    study = optuna.create_study(direction='minimize', pruner=pruner, storage=study_storage, study_name=study_name, load_if_exists=True)
    
    study.optimize(lambda trial: objective(trial, args, config_base), n_trials=args.n_trials)

    print("\n" + "="*50)
    print("--- BÚSQUEDA DE HIPERPARÁMETROS COMPLETADA ---")
    best_trial = study.best_trial
    print(f"\nMejor Trial #{best_trial.number} con MSE: {best_trial.value:.6f}")
    print("Mejores Hiperparámetros:", best_trial.params)

    # --- SECCIÓN DE VISUALIZACIÓN ---
    try:
        import kaleido
        import plotly
        print("\nGenerando gráficos de diagnóstico de Optuna...")
        plot_dir = os.path.join(args.output, study_name, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        for plot_func_name in ['plot_optimization_history', 'plot_param_importances', 'plot_parallel_coordinate', 'plot_slice']:
            plot_func = getattr(optuna.visualization, plot_func_name)
            params_to_plot = list(study.best_params.keys()) if study.best_params else None
            if ("coordinate" in plot_func_name or "slice" in plot_func_name) and params_to_plot:
                fig = plot_func(study, params=params_to_plot)
            else:
                fig = plot_func(study)
            fig.write_image(os.path.join(plot_dir, f"{plot_func_name}.png"))
        print(f"Gráficos guardados en: {plot_dir}")
    except Exception as e:
        print(f"\nNo se pudieron generar los gráficos. Error: {e}")
        
    print("="*50)
