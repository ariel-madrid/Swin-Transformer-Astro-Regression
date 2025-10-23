import os
import time
import random
import datetime
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Importaciones de Optuna
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances

# Importaciones de tus módulos de proyecto
from timm.utils import AverageMeter
from config import get_config
from logger import create_logger
from utils import NativeScalerWithGradNormCount
from NpyDataset import NpyDataset
from PrepareDataloaders import prepare_dataloaders
from BuildModel import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

# --- Funciones de Entrenamiento y Validación (Adaptadas para DDP) ---

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, logger, rank):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    criterion_mse = torch.nn.MSELoss()

    # Sincronizar el dataloader en modo DDP
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # El bucle de entrenamiento no necesita cambiar
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            mse_loss = criterion_mse(outputs, targets)
        
        loss_meter.update(mse_loss.item(), targets.size(0))
        loss = mse_loss / config.TRAIN.ACCUMULATION_STEPS
        
        loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD, parameters=model.parameters(), 
                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        
        lr_scheduler.step()
            
    return loss_meter.avg
@torch.no_grad()
def validate(config, data_loader, model):
    model.eval()
    rmse_meter = AverageMeter()
    criterion = torch.nn.MSELoss()
    for idx, (images, target) in enumerate(data_loader):
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        mse = criterion(output, target)
        rmse = torch.sqrt(mse)
        rmse_meter.update(rmse.item(), images.size(0))
    return rmse_meter.avg

# --- Función OBJECTIVE que Optuna optimizará ---
def objective(trial, base_config, train_loader, val_loader, logger, rank):
    # PASO 1: Rank 0 sugiere hiperparámetros y los prepara para el broadcast
    if rank == 0:
        params_dict = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4),
            "depth": float(trial.suggest_int("depth", 4, 12, step=2)),
            "dim": float(trial.suggest_categorical("dim", [64, 128, 256])),
        }
        params_tensor = torch.tensor(list(params_dict.values()), dtype=torch.float32).cuda(rank)
        logger.info(f"\n--- INICIANDO TRIAL {trial.number} ---\n  Parámetros: {params_dict}")
    else:
        params_tensor = torch.empty(5, dtype=torch.float32).cuda(rank)

    # PASO 2: Broadcast de hiperparámetros desde Rank 0 a todos los demás
    dist.broadcast(params_tensor, src=0)
    
    params_list = params_tensor.cpu().tolist()
    hyperparams = {
        "lr": params_list[0], "weight_decay": params_list[1], "dropout_rate": params_list[2],
        "depth": int(params_list[3]), "dim": int(params_list[4])
    }

    # PASO 3: Todos los ranks actualizan su configuración localmente
    config = base_config.clone()
    config.defrost()
    config.TRAIN.BASE_LR = hyperparams["lr"]
    config.TRAIN.WEIGHT_DECAY = hyperparams["weight_decay"]
    config.MODEL.DROP_RATE = hyperparams["dropout_rate"]
    config.MODEL.CONV_VIT.DEPTH = hyperparams["depth"]
    config.MODEL.CONV_VIT.DIM = hyperparams["dim"]
    
    head_options = {64: 2, 128: 4, 256: 8}
    config.MODEL.CONV_VIT.HEADS = head_options[hyperparams["dim"]]
    config.MODEL.CONV_VIT.DIM_HEAD = hyperparams["dim"] // head_options[hyperparams["dim"]]
    config.freeze()

    # PASO 4: Construcción y entrenamiento del modelo (idéntico en todos los ranks)
    model = build_model(config).cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    loss_scaler = NativeScalerWithGradNormCount()

    N_EPOCHS_OPTUNA = 30 # Número de épocas por trial

    for epoch in range(N_EPOCHS_OPTUNA):
        train_one_epoch(config, model, train_loader, optimizer, epoch, lr_scheduler, loss_scaler, logger, rank)
        
        # Validar en todos los procesos, pero solo reportar y podar desde el rank 0
        val_rmse = validate(config, val_loader, model.module) # .module para acceder al modelo original
        
        if rank == 0:
            logger.info(f"TRIAL {trial.number} - ÉPOCA {epoch+1}/{N_EPOCHS_OPTUNA} - Val RMSE: {val_rmse:.6f}")
            trial.report(val_rmse, epoch)
            prune_flag = 1.0 if trial.should_prune() else 0.0
            prune_tensor = torch.tensor([prune_flag], device=rank)
        else:
            prune_tensor = torch.empty(1, device=rank)

        dist.broadcast(prune_tensor, src=0)
        
        if prune_tensor.item() == 1.0:
            if rank == 0: logger.info(f"TRIAL {trial.number} podado en la época {epoch+1}.")
            raise optuna.exceptions.TrialPruned()
            
    return val_rmse

# --- Función Principal para ejecutar el estudio DDP ---
def main_optuna_ddp():
    parser = argparse.ArgumentParser(description='Optuna DDP Hyperparameter Optimization')
    parser.add_argument('--cfg', type=str, required=True, help='Path to base config file (.yaml)')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials to run')
    parser.add_argument('--tag', type=str, default='optuna_ddp_study', help='A tag for the output folder')
    args, _ = parser.parse_known_args()
    
    config = get_config(args)
    config.defrost()
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, args.tag)
    config.freeze()

    # Inicializar DDP
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    if rank == 0: os.makedirs(config.OUTPUT, exist_ok=True)
    dist.barrier()

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank, name=f"optuna_log")
    
    # Todos los procesos cargan los datos para tener los dataloaders listos
    train_loader, val_loader, _ = prepare_dataloaders(config, logger, NpyDataset)
    
    # La gestión del estudio sigue siendo solo para el rank 0
    if rank == 0:
        study_name = "disk_param_optimization_ddp"
        storage_path = os.path.join(config.OUTPUT, f"{study_name}.db")
        storage_name = f"sqlite:///{storage_path}"
        study = optuna.create_study(
            study_name=study_name, storage=storage_name, direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True
        )
        logger.info(f"Iniciando/Reanudando estudio de Optuna para {args.n_trials} trials.")
    else:
        study = None

    # Bucle de trials sincronizado
    for i in range(args.n_trials):
        # El rank 0 es el único que interactúa con el objeto 'study'
        if rank == 0:
            trial = study.ask()
        else:
            trial = None
            
        try:
            # Todos los procesos ejecutan la función objective
            final_rmse = objective(trial, config, train_loader, val_loader, logger, rank)
            
            # Solo el rank 0 le dice a Optuna que el trial terminó con éxito
            if rank == 0:
                study.tell(trial, final_rmse)
        except optuna.exceptions.TrialPruned:
            # Si se podó, el rank 0 se lo informa a Optuna
            if rank == 0:
                study.tell(trial, state=TrialState.PRUNED)
        except Exception as e:
            # Si hay otro error, el rank 0 lo informa
            if rank == 0:
                logger.error(f"TRIAL {trial.number} falló con error: {e}", exc_info=True)
                study.tell(trial, state=TrialState.FAIL)

    dist.barrier()

    if rank == 0:
        print("\n" + "="*50 + "\nOPTIMIZACIÓN COMPLETADA\n" + "="*50)
        # Recargar el estudio para asegurar la consistencia final
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print(f"Resultados del estudio: {len(study.trials)} trials en total.")
        print(f"  - {len(complete_trials)} trials completados.")
        print(f"  - {len(pruned_trials)} trials podados.")

        print("\n--- MEJOR TRIAL ---")
        try:
            best_trial = study.best_trial
            print(f"  Valor (mejor Val RMSE): {best_trial.value:.6f}")
            print("  Mejores Hiperparámetros:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

            with open(os.path.join(config.OUTPUT, "best_params.txt"), "w") as f:
                f.write(f"Best trial value (RMSE): {best_trial.value}\n")
                for key, value in best_trial.params.items():
                    f.write(f"{key}: {value}\n")
        except ValueError:
            print("No se encontraron trials completados exitosamente.")

        # Guardar gráficos
        try:
            fig_history = plot_optimization_history(study)
            fig_history.write_image(os.path.join(config.OUTPUT, "optuna_history.png"))

            fig_importance = plot_param_importances(study)
            fig_importance.write_image(os.path.join(config.OUTPUT, "optuna_importances.png"))
            
            print(f"\nGráficos de resultados de Optuna guardados en: {config.OUTPUT}")
        except Exception as e:
            print(f"\nADVERTENCIA: No se pudieron generar los gráficos. Error: {e}")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    cudnn.benchmark = True
    main_optuna_ddp()
