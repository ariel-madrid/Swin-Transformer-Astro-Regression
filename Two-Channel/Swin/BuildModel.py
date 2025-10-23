import warnings
warnings.filterwarnings("ignore")

from models import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from main import validate
import torch
#from logger import create_logger
import torch.distributed as dist

def initialize_model_and_optimizer(config, train_loader, val_loader, logger):

    model = build_model(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model
    # Construir optimizador
    optimizer = build_optimizer(config, model)

    # Envolver modelo en DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )

    # Construir programador de tasa de aprendizaje
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # Inicializar el mejor RMSE
    max_rmse = float('inf')
    # Restaurar automáticamente el modelo si está habilitado
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
        else:
            logger.info(f"No checkpoint found in {config.OUTPUT}, ignoring auto resume")
    
    # Restaurar desde un archivo de checkpoint si está configurado
    if config.MODEL.RESUME:
        max_rmse = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        loss = validate(logger,config, val_loader, model)
        logger.info(f"Validation on {len(val_loader.dataset)} test images - "
                    f"Loss: {loss:.4f}")
        if config.EVAL_MODE:
            return model, model_without_ddp, optimizer, lr_scheduler, criterion, max_rmse
    # Cargar pesos preentrenados si están habilitados y no se está reanudando
    """if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        #loss = validate(config, val_loader, model)
        #logger.info(f"Validation on {len(val_loader.dataset)} test images - "
        #            f"Loss: {loss:.4f}")
    """
    return model, model_without_ddp, optimizer, lr_scheduler, max_rmse
