# Import necessary libraries
import os
import yaml
import torch
import warnings
from yacs.config import CfgNode as CN

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Get the major PyTorch version
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

# Create a new configuration node
_C = CN()

# -----------------------------------------------------------------------------
# Base configuration settings
# -----------------------------------------------------------------------------
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 64
_C.DATA.DATA_PATH = ""
_C.DATA.TRAIN_DIR = ''
_C.DATA.FITS = "/home/aargomedo/TESIS/Preprocesar/img_stdnorm_3params_norm_six/train"
_C.DATA.PARAMETERS = ''
_C.DATA.VAL_DIR = ''
_C.DATA.PATH_NORMALIZATION = ''
_C.DATA.IMG_SIZE = 256
_C.DATA.INTERPOLATION = 'bicubic'  # Interpolation method for resizing images
_C.DATA.ZIP_MODE = False  # Use a zipped dataset instead of a folder dataset
_C.DATA.CACHE_MODE = 'part'  # Cache data in memory
_C.DATA.PIN_MEMORY = True  # Pin CPU memory for more efficient data transfer to GPU
_C.DATA.NUM_WORKERS = 4  # Number of data loading threads
_C.DATA.MASK_PATCH_SIZE = 16  # Mask patch size for MaskGenerator (SimMIM)
_C.DATA.MASK_RATIO = 0.6  # Mask ratio for MaskGenerator (SimMIM)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'swin'  # Model type, e.g., 'swin' or 'conv_vit'
_C.MODEL.NAME = 'swin_tiny'  # Model name
_C.MODEL.PRETRAINED = ''  # Path to a pretrained weight file
_C.MODEL.RESUME = ''  # Path to a checkpoint file to resume training
_C.MODEL.NUM_CLASSES = 6  # Number of classes
_C.MODEL.DROP_RATE = 0.100  # Dropout rate
_C.MODEL.DROP_PATH_RATE = 0.243  # Drop path rate
_C.MODEL.LABEL_SMOOTHING = 0.0  # Label smoothing factor

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 8
_C.MODEL.SWIN.IN_CHANS = 2
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 2, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 16
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False  # Absolute position embedding
_C.MODEL.SWIN.PATCH_NORM = True

# CONV_VIT Transformer parameters
_C.MODEL.CONV_VIT = CN()
_C.MODEL.CONV_VIT.PATCH_SIZE = 8
_C.MODEL.CONV_VIT.DIM = 128
_C.MODEL.CONV_VIT.DEPTH = 6
_C.MODEL.CONV_VIT.HEADS = 4
_C.MODEL.CONV_VIT.MLP_DIM = 1024
_C.MODEL.CONV_VIT.DIM_HEAD = 32

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.BASE_LR = 1.93e-4
_C.TRAIN.WARMUP_LR = 1e-5
_C.TRAIN.MIN_LR = 1e-5
_C.TRAIN.CLIP_GRAD = 1.0  # Gradient clipping
_C.TRAIN.AUTO_RESUME = True  # Automatically resume from the last checkpoint
_C.TRAIN.ACCUMULATION_STEPS = 1  # Gradient accumulation steps
_C.TRAIN.USE_CHECKPOINT = False  # Use checkpointing to save memory

# Learning rate scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'  # Scheduler name, e.g., 'cosine', 'step'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 10  # Epoch interval for StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1  # Decay rate for StepLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1  # Gamma for MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []  # Milestones for MultiStepLRScheduler

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'  # Optimizer name, e.g., 'adamw', 'sgd'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25  # Random Erasing probability
_C.AUG.REMODE = 'pixel'  # Random Erasing mode
_C.AUG.RECOUNT = 1  # Random Erasing count
_C.AUG.MIXUP = 0.8
_C.AUG.MIXUP_ENABLE = False
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Miscellaneous settings
# -----------------------------------------------------------------------------
_C.MISC = CN()
_C.ENABLE_AMP = True  # Enable PyTorch automatic mixed precision (AMP)
_C.AMP_ENABLE = True
_C.AMP_OPT_LEVEL = ''  # Deprecated: Apex AMP optimization level
_C.OUTPUT = ''  # Path to the output folder
_C.TAG = 'default'  # Experiment tag
_C.SAVE_FREQ = 1  # Checkpoint saving frequency
_C.PRINT_FREQ = 64  # Logging frequency
_C.SEED = 0  # Random seed
_C.EVAL_MODE = False  # Evaluation mode
_C.THROUGHPUT_MODE = False  # Throughput measurement mode
_C.LOCAL_RANK = 0  # Local rank for DistributedDataParallel
_C.FUSED_WINDOW_PROCESS = False  # Enable fused window process for acceleration
_C.FUSED_LAYERNORM = False  # Enable fused LayerNorm for acceleration

def _update_config_from_file(config, cfg_file):
    """
    Update the configuration from a YAML file.
    """
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f'=> merge config from {cfg_file}')
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    """
    Update the configuration from a YAML file and command-line arguments.
    """
    # First, load settings from the specified YAML file
    _update_config_from_file(config, args.cfg)

    config.defrost()

    # Helper function to check if a command-line argument was provided
    def _check_args(arg_name):
        return hasattr(args, arg_name) and getattr(args, arg_name) is not None

    # Override config with specific command-line arguments if they were provided
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    
    if _check_args('fits_path'):
        config.DATA.FITS = args.fits_path

    # Set local rank for distributed training (provided by torchrun)
    config.LOCAL_RANK = args.local_rank

    # Set the final output folder path
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """
    Get a yacs CfgNode object with default values, updated from a config file
    and command-line arguments.
    """
    config = _C.clone()
    update_config(config, args)
    return config
