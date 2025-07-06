
"""Main script for training and evaluating the Swin Transformer model.

This script handles:
- Parsing command-line arguments and loading configuration.
- Setting up distributed training.
- Initializing the model, optimizer, and learning rate scheduler.
- Running the training and validation loops.
- Implementing early stopping.
- Performing a final test on the best model.
- Logging metrics and saving checkpoints.
"""

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from timm.utils import AverageMeter
from timm.data import Mixup

from config import get_config
from logger import create_logger
from NpyDataset import NpyDataset
from PrepareDataloaders import prepare_dataloaders
from BuildModel import initialize_model_and_optimizer
from utils import (
    load_checkpoint,
    save_checkpoint,
    auto_resume_helper,
    NativeScalerWithGradNormCount
)

def parse_option():
    """Parses command-line arguments and returns the configuration."""
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    
    # --- Arguments used in the execution command ---
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to the configuration file.')
    parser.add_argument('--fits-path', type=str, help='Path to the file containing FITS file paths.')
    parser.add_argument('--accumulation-steps', type=int, help="Gradient accumulation steps.")

    # --- Argument required for distributed training ---
    parser.add_argument("--local_rank", type=int, required=True, help='Local rank for DistributedDataParallel.')

    args, unparsed = parser.parse_known_args()
    
    # get_config will load the base config from the --cfg file
    # and then update it with other command-line arguments.
    config = get_config(args)

    return args, config

def main(config, logger):
    """Main function to run the training and evaluation process."""
    
    # --- Setup ---
    writer = None
    if dist.get_rank() == 0:
        log_dir = os.path.join(config.OUTPUT, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard writer initialized.")

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(config, logger, NpyDataset)
    logger.info("Dataloaders prepared.")

    # Initialize model, optimizer, scheduler, and get best metric from checkpoint
    model, model_without_ddp, optimizer, lr_scheduler, min_rmse = initialize_model_and_optimizer(
        config, train_loader, val_loader, logger
    )
    logger.info("Model, optimizer, and scheduler initialized.")

    # Initialize Mixup for data augmentation if enabled
    mixup_fn = None
    if config.AUG.MIXUP_ENABLE:
        logger.info("Mixup augmentation is enabled.")
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

    # --- Training Loop ---
    logger.info("--- Starting Training ---")
    start_time = time.time()
    loss_scaler = NativeScalerWithGradNormCount()
    
    # Early stopping parameters
    patience = 10
    epochs_no_improve = 0

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_one_epoch(config, model, train_loader, optimizer, epoch, lr_scheduler, loss_scaler, mixup_fn, logger)
        
        # Validate the model
        val_loss = validate(logger, config, val_loader, model)
        
        # --- Logging and Checkpointing (on rank 0) ---
        if dist.get_rank() == 0:
            val_rmse = np.sqrt(val_loss)
            logger.info(f"Epoch {epoch} | Validation Global MSE: {val_loss:.6f}, Validation Global RMSE: {val_rmse:.4f}")

            if val_rmse < min_rmse:
                min_rmse = val_rmse
                epochs_no_improve = 0
                logger.info(f"New best RMSE: {min_rmse:.4f}. Saving checkpoint.")
                save_checkpoint(config, epoch, model_without_ddp, min_rmse, optimizer, lr_scheduler, logger)
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in RMSE for {epochs_no_improve} epoch(s). Best RMSE remains {min_rmse:.4f}.")

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('RMSE/validation', val_rmse, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
    
    if writer and dist.get_rank() == 0:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time: {total_time_str}')

    # --- Final Testing Phase ---
    logger.info("\n--- Starting Final Test Phase ---")
    # Clean up memory before loading the best model
    del model, model_without_ddp, optimizer, lr_scheduler
    torch.cuda.empty_cache()

    # Build a new model instance for testing
    model = build_model(config)
    model.cuda()
    model_without_ddp = model
    
    # Load the best checkpoint for testing
    best_checkpoint_path = auto_resume_helper(config.OUTPUT)
    if best_checkpoint_path:
        logger.info(f"Loading best checkpoint for testing from: {best_checkpoint_path}")
        config.defrost()
        config.MODEL.RESUME = best_checkpoint_path
        config.freeze()
        # We need dummy optimizer and scheduler to load the checkpoint
        dummy_optimizer = build_optimizer(config, model_without_ddp)
        dummy_scheduler = build_scheduler(config, dummy_optimizer, 1)
        load_checkpoint(config, model_without_ddp, dummy_optimizer, dummy_scheduler, logger)
    else:
        logger.warning("No best checkpoint found for testing. Using the weights from the last epoch, which may not be optimal.")

    # Wrap model for distributed testing
    model_for_test = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )

    # Run the final test
    test_loss_avg, test_rmse_avg = test(config, logger, test_loader, model_for_test)

    if dist.get_rank() == 0:
        logger.info(f"--- Final Test Results ---")
        logger.info(f"Average MSE: {test_loss_avg:.6f}")
        logger.info(f"Average RMSE: {test_rmse_avg:.4f}")

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, mixup_fn, logger):
    """Trains the model for one epoch."""
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    criterion_mse = torch.nn.MSELoss()
    start = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = criterion_mse(outputs, targets)
        
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # Scale loss and perform backward pass
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        torch.cuda.synchronize()

        loss_meter.update(loss.item() * config.TRAIN.ACCUMULATION_STEPS, targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        batch_time.update(time.time() - start)
        start = time.time()

        # Log training progress
        if idx % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            
            with torch.no_grad():
                rmse_global = torch.sqrt(loss_meter.val)

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}] | '
                f'ETA: {datetime.timedelta(seconds=int(batch_time.avg * (num_steps - idx)))} | '
                f'LR: {lr:.6f} | '
                f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | '
                f'MSE Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'RMSE (batch): {rmse_global.item():.4f} | '
                f'Grad Norm: {norm_meter.val:.4f} ({norm_meter.avg:.4f}) | '
                f'Mem: {memory_used:.0f}MB'
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training finished. Time: {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg

@torch.no_grad()
def validate(logger, config, data_loader, model):
    """Performs validation on the dataset."""
    model.eval()
    all_outputs = []
    all_targets = []

    for images, target in data_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        all_outputs.append(output)
        all_targets.append(target)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Gather results from all GPUs if in distributed mode
    if dist.is_initialized():
        world_size = dist.get_world_size()
        outputs_list = [torch.zeros_like(all_outputs) for _ in range(world_size)]
        targets_list = [torch.zeros_like(all_targets) for _ in range(world_size)]
        dist.all_gather(outputs_list, all_outputs)
        dist.all_gather(targets_list, all_targets)
        if dist.get_rank() == 0:
            all_outputs = torch.cat(outputs_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

    # Calculate and log metrics on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        mse_per_param = torch.mean((all_outputs - all_targets)**2, dim=0)
        global_mse = torch.mean(mse_per_param)
        rmse_per_param = torch.sqrt(mse_per_param)
        
        param_names = ['Rc', 'H30', 'incl', 'mDisk', 'psi', 'gamma']
        log_str_rmse = " * Validation RMSE per Parameter: "
        for name, rmse_val in zip(param_names, rmse_per_param):
            log_str_rmse += f"{name}: {rmse_val.item():.4f} | "
        logger.info(log_str_rmse)
        
        return global_mse.item()
    else:
        return float('inf') # Return infinity for non-rank-0 processes

@torch.no_grad()
def test(config, logger, data_loader, model):
    """Performs the final test on the dataset."""
    model.eval()
    loss_meter = AverageMeter()
    
    local_outputs_cpu = []
    local_targets_cpu = []

    for images, target in data_loader:
        images = images.cuda(non_blocking=True)
        target = target.float().cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
            if torch.all(torch.isfinite(output)):
                mse_loss = torch.nn.functional.mse_loss(output, target)
                loss_meter.update(mse_loss.item(), target.size(0))
        
        local_outputs_cpu.append(output.cpu())
        local_targets_cpu.append(target.cpu())

    # --- Synchronization and Analysis (Rank 0) ---
    all_outputs_tensor = torch.cat(local_outputs_cpu, dim=0)
    all_targets_tensor = torch.cat(local_targets_cpu, dim=0)

    if dist.is_initialized():
        # This part is simplified as we gather all results on rank 0 later
        # For large datasets, a distributed gathering strategy would be needed here
        pass

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("--- Analyzing and saving test results ---")
        
        all_outputs_np = all_outputs_tensor.cpu().numpy()
        all_targets_np = all_targets_tensor.cpu().numpy()

        # Filter out invalid (NaN/inf) outputs
        valid_mask = np.isfinite(all_outputs_np).all(axis=1)
        num_total = len(all_outputs_np)
        num_valid = valid_mask.sum()
        
        logger.info(f"Total test samples processed: {num_total}")
        logger.info(f"Number of valid (finite) outputs: {num_valid} ({num_valid/num_total:.2%})")
        
        if num_valid < num_total:
            logger.warning(f"DISCARDED {num_total - num_valid} SAMPLES due to NaN/inf outputs.")
        
        clean_outputs = all_outputs_np[valid_mask]
        clean_targets = all_targets_np[valid_mask]

        if num_valid > 0:
            logger.info("Saving arrays of valid outputs, targets, and RMSEs...")
            np.save(os.path.join(config.OUTPUT, "test_outputs.npy"), clean_outputs)
            np.save(os.path.join(config.OUTPUT, "test_targets.npy"), clean_targets)
            
            sample_wise_rmse = np.sqrt(np.mean((clean_outputs - clean_targets)**2, axis=1))
            np.save(os.path.join(config.OUTPUT, "test_rmse_per_sample.npy"), sample_wise_rmse)
            
            final_rmse_per_param = np.sqrt(np.mean((clean_outputs - clean_targets)**2, axis=0))
            param_names = ['Rc', 'H30', 'incl', 'mDisk', 'psi', 'gamma']
            log_str = " * Test RMSE per Parameter (on valid data): "
            for name, rmse_val in zip(param_names, final_rmse_per_param):
                log_str += f"{name}: {rmse_val:.4f} | "
            logger.info(log_str)

            logger.info("--- Global Metrics Summary ---")
            logger.info(f' * MSE (average of valid batches): {loss_meter.avg:.6f}')
            logger.info(f' * RMSE (average of valid batches): {np.sqrt(loss_meter.avg):.4f}')
            logger.info(f' * Global RMSE (from clean array): {np.mean(sample_wise_rmse):.4f}')
        else:
            logger.error("No valid outputs found in the entire test set! No result files will be saved.")

    return loss_meter.avg, np.sqrt(loss_meter.avg)

if __name__ == '__main__':
    # --- Initial Setup ---
    args, config = parse_option()

    # Manually set the FITS path from arguments, as it's not in the standard config update.
    if args.fits_path:
        config.defrost()
        config.DATA.FITS = args.fits_path
        config.freeze()

    # Set up distributed training environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        raise ValueError("RANK and WORLD_SIZE environment variables must be set.")
        
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # --- Seeding and Cudnn ---
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # --- Logger and Config ---
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(str(config))

    # --- Start Main Process ---
    main(config, logger)
