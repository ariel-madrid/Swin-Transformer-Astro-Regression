# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from collections import OrderedDict
import torch.nn.functional as F
try:
    from torch._six import inf
except:
    from torch import inf


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    """
    Carga pesos pre-entrenados de forma robusta. Maneja DDP, cambio de resolución
    y cambio en las cabezas de salida.
    """
    logger.info(f"==============> Cargando pesos {config.MODEL.PRETRAINED} para fine-tuning...")
    
    # Cargar el checkpoint en la CPU para evitar picos de memoria en GPU
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    # Extraer el state_dict del modelo del checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        logger.warning("El checkpoint no contiene la clave 'model', se asume que el archivo es el state_dict directamente.")

    # Manejar el prefijo 'module.' que añade DistributedDataParallel (DDP)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Eliminar buffers que dependen de la resolución y se regeneran
    keys_to_delete = [k for k in state_dict.keys() if "relative_position_index" in k or "attn_mask" in k]
    for k in keys_to_delete:
        if k in state_dict:
            del state_dict[k]
            # logger.info(f"Ignorando buffer {k} de checkpoint por cambio de resolución.")

    # Interpolar tablas de sesgo de posición relativa si el tamaño no coincide
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        try:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Número de cabezas diferente. Saltando la carga de {k}")
                del state_dict[k]
                continue
            if L1 != L2:
                logger.warning(f"Interpolando tabla de sesgo de posición relativa para '{k}' de tamaño {L1} a {L2}")
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = F.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic'
                )
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH1, L2).permute(1, 0)
        except Exception as e:
            logger.error(f"Error al interpolar {k}: {e}. Saltando esta clave.")
            del state_dict[k]

    # Lógica para la cabeza de regresión: si no coinciden, no se cargan
    current_head_keys = {k for k in model.state_dict().keys() if k.startswith('heads.')}
    ckpt_head_keys = {k for k in state_dict.keys() if k.startswith('heads.')}
    
    if current_head_keys != ckpt_head_keys:
        logger.warning("Las cabezas de regresión no coinciden entre el modelo y el checkpoint.")
        logger.warning("Se inicializarán las cabezas con pesos aleatorios y no se cargarán desde el checkpoint.")
        for k in ckpt_head_keys:
            if k in state_dict:
                del state_dict[k]
    
    # Cargar el diccionario de estado con strict=False, ya que hemos eliminado/modificado claves
    msg = model.load_state_dict(state_dict, strict=False)
    
    # Imprimir un resumen claro de lo que se cargó y lo que no
    logger.info(f"Mensaje de carga de Pytorch: {msg}")
    if msg.missing_keys:
        logger.warning(f"Claves faltantes en el checkpoint (se usarán los pesos inicializados del modelo): {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.warning(f"Claves inesperadas en el checkpoint (se ignorarán): {msg.unexpected_keys}")

    logger.info(f"=> Pesos de '{config.MODEL.PRETRAINED}' cargados exitosamente para fine-tuning.")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    #print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
