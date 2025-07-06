# Import necessary libraries
import warnings
warnings.filterwarnings("ignore") # Ignore warnings

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
import torch.distributed as dist
import os

def prepare_dataloaders(config, logger, NpyDataset):
    """
    Prepares the data loaders for training, validation, and testing.

    Args:
        config: A configuration object with data augmentation and dataloader settings.
        logger: A logger object for logging information.
        NpyDataset: The dataset class to be used.

    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    # Define a list of transformations for training data augmentation
    train_transform_list = [
        # Randomly flip the image horizontally with a probability of 0.5
        transforms.RandomHorizontalFlip(p=0.5), 

        # Randomly flip the image vertically with a probability of 0.5
        transforms.RandomVerticalFlip(p=0.5),

        # Randomly rotate the image by an angle between -20 and 20 degrees.
        # transforms.RandomRotation(degrees=20),
    ]

    # If Random Erasing is enabled in the configuration, add it to the list of transformations
    if config.AUG.REPROB > 0:
        logger.info(f"Enabling Random Erasing with probability: {config.AUG.REPROB}")
        train_transform_list.append(
            transforms.RandomErasing(
                p=config.AUG.REPROB, 
                scale=(0.02, 0.2), # Range of the area to erase (2% to 20% of the image)
                ratio=(0.3, 3.3),   # Range of the aspect ratio of the rectangle
                value=0,            # Fill with zeros (or you can use 'random')
                inplace=False
            )
        )
    
    # Compose all the transformations in the list
    train_transforms = transforms.Compose(train_transform_list)

    # Create the full datasets for training, validation, and testing
    train_dataset = NpyDataset(os.path.join(config.DATA.FITS, "train"), train_transforms)
    val_dataset = NpyDataset(os.path.join(config.DATA.FITS, "val"), None)
    test_dataset = NpyDataset(os.path.join(config.DATA.FITS, "test"), None)

    # Configure the distributed samplers for multi-GPU training
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    sampler_test = DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    # Generate the DataLoaders for training, validation, and testing
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

    # Return the created data loaders
    return train_loader, val_loader, test_loader