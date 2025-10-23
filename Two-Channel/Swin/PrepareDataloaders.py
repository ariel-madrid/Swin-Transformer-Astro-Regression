import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
import torch.distributed as dist


def prepare_dataloaders(config, logger, NpyDataset):
    train_transform_list = [
        # Voltea la imagen horizontalmente con un 50% de probabilidad
        transforms.RandomHorizontalFlip(p=0.5), 
        # Voltea la imagen verticalmente con un 50% de probabilidad
        transforms.RandomVerticalFlip(p=0.5),
    ]


    if config.AUG.REPROB > 0:
        logger.info(f"Activando Random Erasing con probabilidad: {config.AUG.REPROB}")
        train_transform_list.append(
            transforms.RandomErasing(
                p=config.AUG.REPROB,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0,
                inplace=False
            )
        )
    # Componer todas las transformaciones de la lista
    train_transforms = transforms.Compose(train_transform_list)

    # Crear el dataset completo
    train_dataset = NpyDataset("/home/aargomedo/TESIS/Preprocesar/img_preprocessed/train",train_transforms)
    val_dataset = NpyDataset("/home/aargomedo/TESIS/Preprocesar/img_preprocessed/val",None)
    test_dataset = NpyDataset("/home/aargomedo/TESIS/Preprocesar/img_preprocessed/test",None)

    logger.info(f"Largo de train_dataset: {len(train_dataset)}")
    logger.info(f"Largo de dataset_val: {len(val_dataset)}")
    logger.info(f"Largo de test_dataset: {len(test_dataset)}")

    # Configurar los samplers distribuidos
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    sampler_test = DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    # Generar DataLoaders
    logger.info("Generando loaders...")
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader
