import warnings
warnings.filterwarnings("ignore")

import os
import time
import random
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader, DistributedSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from timm.utils import AverageMeter
from config import get_config
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from timm.data import Mixup
from NpyDataset import NpyDataset
import argparse
from PrepareDataloaders import *
from BuildModel import *


PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+',)
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--fits-path', type=str, help='path to train ataset')
    parser.add_argument('--pretrained',help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument('--fused_window_process', action='store_true',help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str,help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config 

def main(config):

    writer = None
    if dist.get_rank() == 0:
        log_dir = os.path.join(config.OUTPUT, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)


    train_loader, val_loader, test_loader = prepare_dataloaders(config, logger, NpyDataset)

    model, model_without_ddp, optimizer, lr_scheduler, max_rmse = initialize_model_and_optimizer(config, train_loader, val_loader, logger)

    mixup_fn = None
    if config.AUG.MIXUP_ENABLE:
        logger.info("Mixup activado.")
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    start_time = time.time()
    loss_val_per_epoch = []
    loss_train_per_epoch = []

    loss_scaler = NativeScalerWithGradNormCount()

    patience = 15
    epochs_no_improve = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        loss_train = train_one_epoch(config, model, train_loader, optimizer, epoch, lr_scheduler, loss_scaler, mixup_fn)

        loss_val= validate(logger,config, val_loader, model)

        if dist.get_rank() == 0:
            logger.info(f"Validation Global MSE: {loss_val:.4f}")
            if np.sqrt(loss_val) < max_rmse:
                max_rmse = np.sqrt(loss_val)
                epochs_no_improve = 0
                logger.info(f"New best RMSE: {max_rmse:.4f}")

                save_checkpoint(config, epoch, model_without_ddp, max_rmse, optimizer, lr_scheduler, logger)
            else:
                epochs_no_improve += 1

            if writer:
                writer.add_scalar('Loss/train', loss_train, epoch)
                writer.add_scalar('Loss/validation', loss_val, epoch)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_Rate', current_lr, epoch)

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

            loss_val_per_epoch.append(loss_val)
            loss_train_per_epoch.append(loss_train)


    if writer and dist.get_rank() == 0:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Training time {}'.format(total_time_str))
    logger.info(f'El modelo no mejoro durante {epochs_no_improve} epocas')
    if dist.get_rank() == 0:
        np.save("loss_train.npy", loss_train_per_epoch)
        np.save("loss_val.npy", loss_val_per_epoch)
    logger.info("\n--- Iniciando Fase de Test Final ---\n")
    del model, model_without_ddp, optimizer, lr_scheduler
    torch.cuda.empty_cache()
    
    #Construir nueva instancia del modelo
    model = build_model(config)
    model.cuda()
    model_without_ddp = model
    
    #Cargar el mejor modelo
    best_checkpoint_path = auto_resume_helper(config.OUTPUT)
    if best_checkpoint_path:
        logger.info(f"Cargando el mejor checkpoint para el test desde: {best_checkpoint_path}")

        config.defrost()
        config.MODEL.RESUME = best_checkpoint_path
        config.freeze()

        dummy_optimizer = build_optimizer(config, model_without_ddp)
        dummy_scheduler = build_scheduler(config, dummy_optimizer, 1)
        load_checkpoint(config, model_without_ddp, dummy_optimizer, dummy_scheduler, logger)
    else:
        logger.warning("No se encontró ningún checkpoint para cargar para el test. Usando los pesos de la última época (puede no ser óptimo).")

    model_for_test = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )

    loss_test_avg, rmse_test_avg = test(config, logger, test_loader, model_for_test)

    if dist.get_rank() == 0:
        logger.info(f"Resultados Finales del Test - MSE: {loss_test_avg:.6f}, RMSE: {rmse_test_avg:.4f}")
    return 0

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()

    criterion_mse = torch.nn.MSELoss()

    update_counter = 0
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = criterion_mse(outputs, targets)

        rmse = torch.sqrt(loss)

        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(), create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)


        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        loss_meter.update(loss.item() * config.TRAIN.ACCUMULATION_STEPS, targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)

        batch_time.update(time.time() - start)

        if idx % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            with torch.no_grad():
                rmse_per_param = torch.sqrt(torch.mean((outputs - targets)**2, dim=0))

            param_names = ['Rc','H30','incl','mDisk','gamma','psi']
            rmse_log_str = " | RMSEs: "
            for name, p_rmse in zip(param_names, rmse_per_param):
                rmse_log_str += f"{name}={p_rmse.item():.4f} "

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'MSELoss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'RMSE (Global) {rmse.item():.4f}{rmse_log_str}\t'
                f'Grad Norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB\t'
                f'Scaler {loss_scale_value:.0f}')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg

@torch.no_grad()
def validate(logger, config, data_loader, model):
    model.eval()
    all_outputs = []
    all_targets = []

    for idx, (images, target) in enumerate(data_loader):
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        all_outputs.append(output)
        all_targets.append(target)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        outputs_list = [torch.zeros_like(all_outputs) for _ in range(world_size)]
        targets_list = [torch.zeros_like(all_targets) for _ in range(world_size)]

        dist.all_gather(outputs_list, all_outputs)
        dist.all_gather(targets_list, all_targets)

        if dist.get_rank() == 0:
            all_outputs = torch.cat(outputs_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

    if not dist.is_initialized() or dist.get_rank() == 0:
        mse_per_param = torch.mean((all_outputs - all_targets)**2, dim=0)

        global_mse = torch.mean(mse_per_param)

        rmse_per_param = torch.sqrt(mse_per_param)
        global_rmse = torch.sqrt(global_mse)

        logger.info(f' * Final Validation MSE (Global): {global_mse.item():.6f}') 
        logger.info(f' * (Para referencia) Final Validation RMSE (Global): {global_rmse.item():.4f}')

        param_names = ['Rc','H30','incl','mDisk','gamma','psi']
        log_str_mse = " * Val MSE por Parámetro: "
        log_str_rmse = " * Val RMSE por Parámetro: "
        for name, mse_val, rmse_val in zip(param_names, mse_per_param, rmse_per_param):
            log_str_mse += f"{name}: {mse_val.item():.6f} | "
            log_str_rmse += f"{name}: {rmse_val.item():.4f} | "
        logger.info(log_str_mse)
        logger.info(log_str_rmse)

        return global_mse.item()
    else:
        return float('inf')

@torch.no_grad()
def test(config, logger, data_loader, model):
    model.eval()

    # Métricas para el proceso actual
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()

    local_outputs_cpu = []
    local_targets_cpu = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.float().cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
            if torch.all(torch.isfinite(output)):
                mse_loss = torch.nn.functional.mse_loss(output, target)
                loss_meter.update(mse_loss.item(), target.size(0))
                rmse_meter.update(torch.sqrt(mse_loss).item(), target.size(0))
        
        # Mueve los resultados a la CPU para liberar VRAM
        local_outputs_cpu.append(output.cpu())
        local_targets_cpu.append(target.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'MSELoss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'RMSE {rmse_meter.val:.3f} ({rmse_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    # --- Sincronización entre todas las GPUs ---
    
    # 1. Concatenar tensores en la CPU
    local_outputs_tensor_cpu = torch.cat(local_outputs_cpu, dim=0)
    local_targets_tensor_cpu = torch.cat(local_targets_cpu, dim=0)

    # 2. Sincronizar si el entorno es distribuido
    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        # Mover los tensores locales a la GPU actual ANTES de la comunicación
        current_gpu = torch.device(f"cuda:{dist.get_rank()}")
        local_outputs_tensor_gpu = local_outputs_tensor_cpu.to(current_gpu)
        local_targets_tensor_gpu = local_targets_tensor_cpu.to(current_gpu)
        
        # Crear listas para recibir los tensores (estos también deben estar en la GPU)
        outputs_list_gpu = [torch.zeros_like(local_outputs_tensor_gpu) for _ in range(world_size)]
        targets_list_gpu = [torch.zeros_like(local_targets_tensor_gpu) for _ in range(world_size)]

        # Realizar all_gather con los tensores en la GPU
        dist.all_gather(outputs_list_gpu, local_outputs_tensor_gpu)
        dist.all_gather(targets_list_gpu, local_targets_tensor_gpu)

        if dist.get_rank() == 0:
            # En el rank 0, concatenar los resultados de todas las GPUs
            all_outputs_tensor = torch.cat(outputs_list_gpu, dim=0)
            all_targets_tensor = torch.cat(targets_list_gpu, dim=0)
    else:
        # Si no es distribuido, los tensores de la CPU ya son los completos
        all_outputs_tensor = local_outputs_tensor_cpu
        all_targets_tensor = local_targets_tensor_cpu

    # --- Filtrado, Guardado y Análisis (SOLO en el proceso rank 0) ---
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("--- Iniciando análisis y guardado de resultados de test ---")
        
        all_outputs_np = all_outputs_tensor.cpu().numpy()
        all_targets_np = all_targets_tensor.cpu().numpy()

        # Filtrado de datos inválidos (NaN/inf)
        valid_mask = np.isfinite(all_outputs_np).all(axis=1)
        
        num_total = len(all_outputs_np)
        num_valid = valid_mask.sum()
        
        logger.info(f"Total de muestras de test procesadas: {num_total}")
        logger.info(f"Número de salidas válidas (finitas): {num_valid} ({num_valid/num_total:.2%})")
        
        if num_valid < num_total:
            logger.warning(f"SE DESCARTARON {num_total - num_valid} MUESTRAS por contener salidas NaN/inf.")
        
        clean_outputs = all_outputs_np[valid_mask]
        clean_targets = all_targets_np[valid_mask]

        if num_valid > 0:

            r2_global = r2_score(clean_targets, clean_outputs)
            r2_per_param = []
            for i in range(clean_targets.shape[1]):
                r2 = r2_score(clean_targets[:, i], clean_outputs[:, i])
                r2_per_param.append(r2)

            # Guardado de arrays limpios
            logger.info("Guardando arrays de salidas, etiquetas y RMSEs válidos...")
            np.save(os.path.join(config.OUTPUT, "test_outputs.npy"), clean_outputs)
            np.save(os.path.join(config.OUTPUT, "test_targets.npy"), clean_targets)
            
            # Cálculo y guardado de métricas sobre datos limpios
            sample_wise_rmse = np.sqrt(np.mean((clean_outputs - clean_targets)**2, axis=1))
            np.save(os.path.join(config.OUTPUT, "test_rmse_per_sample.npy"), sample_wise_rmse)
            
            final_mse_per_param = np.mean((clean_outputs - clean_targets)**2, axis=0)
            final_rmse_per_param = np.sqrt(final_mse_per_param)

            param_names = ['Rc','H30','incl','mDisk','gamma','psi']

            logger.info("--- Resumen de Métricas Globales ---")
            logger.info(f' * MSE (promedio de lotes válidos, desde meter): {loss_meter.avg:.6f}')
            logger.info(f' * RMSE (promedio de lotes válidos, desde meter): {rmse_meter.avg:.4f}')
            logger.info(f' * RMSE Global (calculado desde array limpio): {np.mean(sample_wise_rmse):.4f}')
            logger.info(f' * R-cuadrado (Global): {r2_global:.4f}')


            log_str_rmse = " * RMSE por Parámetro: "
            log_str_r2 =   " * R^2 por Parámetro:  "
            for i, name in enumerate(param_names):
                log_str_rmse += f"{name}: {final_rmse_per_param[i]:.4f} | "
                log_str_r2 += f"{name}: {r2_per_param[i]:.4f} | "
            
            logger.info(log_str_rmse)
            logger.info(log_str_r2) 
        
        else:
            logger.error("¡No se encontraron salidas válidas en todo el test set! No se guardarán archivos de resultados.")

    return loss_meter.avg, rmse_meter.avg
def scale_learning_rates(config):
    scale_factor = config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    config.TRAIN.BASE_LR *= scale_factor
    config.TRAIN.WARMUP_LR *= scale_factor
    config.TRAIN.MIN_LR *= scale_factor
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.TRAIN.BASE_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.ACCUMULATION_STEPS
    return config

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    if 'LOCAL_RANK' in os.environ:
        config.defrost()
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        config.freeze()
    else:
        raise ValueError("LOCAL_RANK environment variable not set.")
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if rank == -1 or world_size == -1:
        raise ValueError("RANK and WORLD_SIZE must be set. Please use torch.distributed.launch or torchrun.")
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank {rank} is using GPU {local_rank}")

    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #config.defrost()
    #config = scale_learning_rates(config)
    #config.DATA.FITS = args.fits_path
    #config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    main(config)
