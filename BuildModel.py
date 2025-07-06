# Import standard libraries
import warnings
import torch
import torch.distributed as dist

# Import local modules
from models import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils import (
    load_checkpoint, 
    load_pretrained, 
    auto_resume_helper
)
from main import validate

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def initialize_model_and_optimizer(config, train_loader, val_loader, logger):
    """
    Initializes the model, optimizer, and learning rate scheduler.
    Handles loading checkpoints or pretrained weights.

    Args:
        config: Configuration object.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        logger: Logger for logging information.

    Returns:
        A tuple containing:
        - model: The initialized and potentially distributed model.
        - model_without_ddp: The model without the DDP wrapper.
        - optimizer: The configured optimizer.
        - lr_scheduler: The configured learning rate scheduler.
        - min_rmse: The minimum RMSE loaded from a checkpoint, or infinity.
    """
    logger.info("Creating model")
    
    # Build the model based on the configuration
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    # Log the number of parameters and GFLOPs
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    # Build the optimizer
    optimizer = build_optimizer(config, model)
    
    # Wrap the model for distributed data parallel processing
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )

    # Build the learning rate scheduler
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # Initialize minimum RMSE for tracking the best model
    min_rmse = float('inf')

    # --- Checkpoint and Pretrained Model Loading ---

    # Handle auto-resuming from the last checkpoint if enabled
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            logger.warning(f"Auto-resuming from: {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
        else:
            logger.info(f"No checkpoint found in {config.OUTPUT}, ignoring auto-resume.")

    # Load a checkpoint if a resume file is specified
    if config.MODEL.RESUME:
        min_rmse = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        loss = validate(logger, config, val_loader, model)
        logger.info(f"Validation on {len(val_loader.dataset)} images - Loss: {loss:.4f}")
        # If in evaluation mode, exit after validation
        if config.EVAL_MODE:
            return model, model_without_ddp, optimizer, lr_scheduler, min_rmse
            
    # Load pretrained weights if specified and not resuming from a checkpoint
    elif config.MODEL.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)
        loss = validate(config, val_loader, model)
        logger.info(f"Validation on {len(val_loader.dataset)} images - Loss: {loss:.4f}")

    return model, model_without_ddp, optimizer, lr_scheduler, min_rmse