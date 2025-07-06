from .swin_transformer import SwinTransformer

from .conv_vit import ConvViT  

import torch.nn as nn 

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'conv_vit':
        model = ConvViT(
            image_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.CONV_VIT.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            dim=config.MODEL.CONV_VIT.DIM,
            depth=config.MODEL.CONV_VIT.DEPTH,
            heads=config.MODEL.CONV_VIT.HEADS,
            mlp_dim=config.MODEL.CONV_VIT.MLP_DIM,
            channels=config.MODEL.SWIN.IN_CHANS,
            dim_head=config.MODEL.CONV_VIT.DIM_HEAD
        )
        
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
