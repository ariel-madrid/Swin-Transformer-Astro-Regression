import os
import time
import random
import datetime
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.backends.cudnn as cudnn

from timm.utils import AverageMeter
from config import get_config
from logger import create_logger 
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from NpyDataset import NpyDataset
import argparse
from torchvision import transforms
import optuna

PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, current_logger):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time, loss_meter, norm_meter, scaler_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    start_time_epoch = time.time()
    criterion_mse = torch.nn.MSELoss()

    for idx, (samples, targets) in enumerate(data_loader):
        start_batch_time = time.time()
        samples = samples.cuda(non_blocking=True) if torch.cuda.is_available() else samples
        targets = targets.cuda(non_blocking=True) if torch.cuda.is_available() else targets

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE if hasattr(config, 'AMP_ENABLE') else False):
            outputs = model(samples)
            loss = criterion_mse(outputs, targets)
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        
        loss_scale_value = loss_scaler.state_dict()["scale"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None: norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - start_batch_time)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            mem_used_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0
            etas = batch_time.avg * (num_steps - idx)
            current_logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f} wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {mem_used_mb:.0f}MB')
    
    current_logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(time.time() - start_time_epoch))}")
    return loss_meter.avg

@torch.no_grad()
def validate(config, data_loader, model, current_logger):
    criterion = torch.nn.MSELoss()
    model.eval()
    loss_meter = AverageMeter()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
        target = target.float().cuda(non_blocking=True) if torch.cuda.is_available() else target.float()
        
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE if hasattr(config, 'AMP_ENABLE') else False):
            output = model(images)
        loss = criterion(output, target)
        loss_meter.update(loss.item(), target.size(0))

    current_logger.info(f' * Validation: Average Loss {loss_meter.avg:.4f}') # , Average MAE {mae_meter.avg:.3f}')
    return loss_meter.avg


def prepare_dataloaders_no_ddp(config, logger, NpyDataset_class, train_dir, val_dir, test_dir=None):

    # Definir los percentiles de clipping para las imágenes
    data_norm = np.load("/home/aargomedo/TESIS/Swin-Transformer-Six-Parameters/data.npy")
    mean = [data_norm[0], data_norm[1]]
    std = [data_norm[2],data_norm[3]]

    transform = transforms.Compose([
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),  # Redimensionar la imagen
        transforms.Normalize(mean=mean, std=std)  #
    ])
    train_dataset = NpyDataset_class(train_dir, transform)
    val_dataset = NpyDataset_class(val_dir, transform)
    
    logger.info(f"Largo de train_dataset: {len(train_dataset)}")
    logger.info(f"Largo de val_dataset: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True #
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=False
    )
    
    test_loader_out = None
    if test_dir:
        test_dataset = NpyDataset_class(test_dir, clip_percentiles=clip_cfg)
        logger.info(f"Largo de test_dataset: {len(test_dataset)}")
        test_loader_out = DataLoader(
            test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
            num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=False
        )
    return train_loader, val_loader, test_loader_out


def initialize_model_optimizer_scheduler_no_ddp(config, train_loader_len, logger):
    from models import build_model as build_model_func
    from optimizer import build_optimizer
    from lr_scheduler import build_scheduler

    model = build_model_func(config) 
    logger.info(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if hasattr(model, 'flops'): logger.info(f"number of GFLOPs: {model.flops() / 1e9}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = build_optimizer(config, model)
    
    num_updates_for_scheduler = train_loader_len 
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        num_updates_for_scheduler = train_loader_len // config.TRAIN.ACCUMULATION_STEPS
        
    lr_scheduler = build_scheduler(config, optimizer, num_updates_for_scheduler)
    
    min_val_metric = float('inf') 
    
    if config.MODEL.RESUME:
        min_val_metric = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
    elif config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)

    return model, optimizer, lr_scheduler, min_val_metric


def scale_learning_rates(config, current_world_size=1): 
    scale_factor = config.DATA.BATCH_SIZE * current_world_size / 512.0
    if scale_factor > 0:
        config.TRAIN.BASE_LR *= scale_factor
        config.TRAIN.WARMUP_LR *= scale_factor
        config.TRAIN.MIN_LR *= scale_factor
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.TRAIN.BASE_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.ACCUMULATION_STEPS
    return config


# --- Función Objetivo para Optuna ---
def objective(trial, base_config_obj, fixed_args_obj):
    current_logger = fixed_args_obj.main_optuna_logger # Usar el logger principal de Optuna para el trial
    current_logger.info(f"--- Starting Optuna Trial {trial.number} ---")

    config_trial = base_config_obj.clone()
    config_trial.defrost()

    # Sugerir Hiperparámetros
    config_trial.TRAIN.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    
    base_lr_raw = trial.suggest_float("base_lr_raw", 1e-5, 1e-2, log=True)
    warmup_factor = trial.suggest_float("warmup_factor", 0.01, 0.5, log=True) 
    min_lr_factor_of_warmup = trial.suggest_float("min_lr_factor_of_warmup", 0.01, 0.5, log=True) 

    config_trial.TRAIN.BASE_LR = base_lr_raw
    config_trial.TRAIN.WARMUP_LR = base_lr_raw * warmup_factor
    config_trial.TRAIN.MIN_LR = config_trial.TRAIN.WARMUP_LR * min_lr_factor_of_warmup
    
    # Asegurar límites y consistencia
    config_trial.TRAIN.WARMUP_LR = max(1e-8, min(config_trial.TRAIN.WARMUP_LR, config_trial.TRAIN.BASE_LR * 0.9))
    config_trial.TRAIN.MIN_LR = max(1e-9, min(config_trial.TRAIN.MIN_LR, config_trial.TRAIN.WARMUP_LR * 0.9))


    trial_output_dir = os.path.join(fixed_args_obj.base_output_dir_optuna, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    config_trial.OUTPUT = trial_output_dir
    config_trial.SAVE_FREQ = fixed_args_obj.optuna_epochs + 1 # No guardar checkpoints

    config_trial = scale_learning_rates(config_trial, current_world_size=1)
    config_trial.freeze()

    current_logger.info("Hyperparameters for this trial (before scaling by scale_learning_rates):")
    current_logger.info(f"  WEIGHT_DECAY: {trial.params['weight_decay']:.3e}")
    current_logger.info(f"  BASE_LR (raw): {trial.params['base_lr_raw']:.3e}")
    current_logger.info(f"  WARMUP_FACTOR: {trial.params['warmup_factor']:.3f} -> WARMUP_LR (raw): {base_lr_raw * warmup_factor:.3e}")
    current_logger.info(f"  MIN_LR_FACTOR_OF_WARMUP: {trial.params['min_lr_factor_of_warmup']:.3f} -> MIN_LR (raw): {(base_lr_raw * warmup_factor) * min_lr_factor_of_warmup:.3e}")
    current_logger.info("Effective Hyperparameters for this trial (after scaling and adjustments):")
    current_logger.info(f"  Effective WEIGHT_DECAY: {config_trial.TRAIN.WEIGHT_DECAY:.3e}")
    current_logger.info(f"  Effective BASE_LR: {config_trial.TRAIN.BASE_LR:.3e}")
    current_logger.info(f"  Effective WARMUP_LR: {config_trial.TRAIN.WARMUP_LR:.3e}")
    current_logger.info(f"  Effective MIN_LR: {config_trial.TRAIN.MIN_LR:.3e}")

    # --- Preparar DataLoaders ---
    train_loader, val_loader, _ = prepare_dataloaders_no_ddp(
        config_trial, current_logger, NpyDataset,
        fixed_args_obj.optuna_train_data_dir,
        fixed_args_obj.optuna_val_data_dir
    )

    # --- Inicializar Modelo y Optimizador ---
    model_trial, optimizer_trial, lr_scheduler_trial, _ = \
        initialize_model_optimizer_scheduler_no_ddp(config_trial, len(train_loader), current_logger)

    # --- Bucle de Entrenamiento para Optuna ---
    min_val_loss_for_this_trial = float('inf')
    loss_scaler = NativeScalerWithGradNormCount()

    for epoch in range(fixed_args_obj.optuna_epochs):
        # No necesitamos train_loader.sampler.set_epoch() sin DDP
        train_loss_avg = train_one_epoch(config_trial, model_trial, train_loader, optimizer_trial,
                                         epoch, lr_scheduler_trial, loss_scaler, current_logger)
        
        current_val_loss = validate(config_trial, val_loader, model_trial, current_logger)
        current_logger.info(f"TRIAL {trial.number} - EPOCH {epoch}/{fixed_args_obj.optuna_epochs-1} - Train_Loss: {train_loss_avg:.4f} - Val_Loss: {current_val_loss:.4f}")

        min_val_loss_for_this_trial = min(min_val_loss_for_this_trial, current_val_loss)
        
        trial.report(current_val_loss, epoch)
        if trial.should_prune():
            current_logger.info(f"Trial {trial.number} pruned at epoch {epoch} with val_loss {current_val_loss:.4f}.")
            raise optuna.TrialPruned()

    current_logger.info(f"--- Trial {trial.number} finished. Min Val Loss for this trial: {min_val_loss_for_this_trial:.4f} ---")
    return min_val_loss_for_this_trial


def parse_option_optuna(): # Renombrado para Optuna
    parser = argparse.ArgumentParser('Swin Transformer Optuna HPO script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to base config file for Swin model (e.g., Swin-B C_MODEL...)')
    parser.add_argument('--optuna-trials', type=int, default=30, help='Number of Optuna trials to run')
    parser.add_argument('--optuna-epochs', type=int, default=20, help='Number of epochs to train each Optuna trial')
    parser.add_argument('--optuna-study-name', type=str, default="swin_regression_hpo_study", help='Name for the Optuna study')
    parser.add_argument('--optuna-train-data-dir', type=str, required=True, help='Path to NPY train data for Optuna (can be a small subset)')
    parser.add_argument('--optuna-val-data-dir', type=str, required=True, help='Path to NPY val data for Optuna (can be a small subset)')
    
    # Argumentos para sobreescribir la config base (ejemplos)
    parser.add_argument('--batch-size', type=int, help="Batch size for Optuna trials (overwrites config.DATA.BATCH_SIZE)")
    parser.add_argument('--num-workers', type=int, help="Number of data loading workers (overwrites config.DATA.NUM_WORKERS)")
    parser.add_argument('--output', default='output_optuna_study', type=str, metavar='PATH', help='Root output folder for Optuna study logs and trial subfolders')
    parser.add_argument("--opts",help="Modify base config options: 'KEY1 VALUE1 KEY2 VALUE2'.",default=None,nargs='+',)
    
    args = parser.parse_args() 
    
    config_loader_args = argparse.Namespace(cfg=args.cfg, opts=args.opts if args.opts else []) 
    base_swin_config = get_config(config_loader_args)
    
    base_swin_config.defrost()
    if args.batch_size is not None:
        base_swin_config.DATA.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        base_swin_config.DATA.NUM_WORKERS = args.num_workers
    base_swin_config.freeze()

    return args, base_swin_config


# --- Script Principal para Ejecutar Optuna ---
if __name__ == '__main__':
    optuna_cli_args, base_config = parse_option_optuna() 

    torch.manual_seed(base_config.SEED)
    torch.cuda.manual_seed_all(base_config.SEED) 
    np.random.seed(base_config.SEED)
    random.seed(base_config.SEED)
    cudnn.benchmark = True 
    main_optuna_logger = create_logger(output_dir=optuna_cli_args.output, dist_rank=0, name="OptunaStudyRunner")

    # Argumentos fijos para pasar a la función `objective`
    fixed_objective_args = argparse.Namespace()
    fixed_objective_args.base_output_dir_optuna = optuna_cli_args.output
    fixed_objective_args.optuna_epochs = optuna_cli_args.optuna_epochs
    fixed_objective_args.optuna_train_data_dir = optuna_cli_args.optuna_train_data_dir
    fixed_objective_args.optuna_val_data_dir = optuna_cli_args.optuna_val_data_dir
    fixed_objective_args.main_optuna_logger = main_optuna_logger # Pasar el logger

    main_optuna_logger.info(f"--- Starting Optuna Study: {optuna_cli_args.optuna_study_name} ---")
    main_optuna_logger.info(f"Number of trials: {optuna_cli_args.optuna_trials}")
    main_optuna_logger.info(f"Epochs per trial: {optuna_cli_args.optuna_epochs}")
    main_optuna_logger.info(f"Train data for Optuna: {optuna_cli_args.optuna_train_data_dir}")
    main_optuna_logger.info(f"Validation data for Optuna: {optuna_cli_args.optuna_val_data_dir}")
    main_optuna_logger.info(f"Optuna trial logs will be in subdirectories of: {optuna_cli_args.output}")
    main_optuna_logger.info(f"Base Swin Config BATCH_SIZE: {base_config.DATA.BATCH_SIZE}")


    study = optuna.create_study(
        study_name=optuna_cli_args.optuna_study_name,
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=fixed_objective_args.optuna_epochs // 3, interval_steps=1)
    )

    try:
        study.optimize(
            lambda trial_obj: objective(trial_obj, base_config.clone(), fixed_objective_args), 
            n_trials=optuna_cli_args.optuna_trials
        )
    except KeyboardInterrupt:
        main_optuna_logger.info("Optuna study interrupted by user.")
    finally:
        main_optuna_logger.info("--- Optuna Study Finished (or Interrupted) ---")
        main_optuna_logger.info(f"Number of finished trials: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            best_trial_overall = study.best_trial 
            main_optuna_logger.info(f"Best trial overall: Trial {best_trial_overall.number}")
            main_optuna_logger.info(f"  Value (Min Val MSE): {best_trial_overall.value:.6f}")
            main_optuna_logger.info("  Best hyperparameters found:")
            for key, value in best_trial_overall.params.items():
                if isinstance(value, float):
                    main_optuna_logger.info(f"    {key}: {value:.3e}")
                else:
                    main_optuna_logger.info(f"    {key}: {value}")
        else:
            main_optuna_logger.info("No trials were completed successfully.")
