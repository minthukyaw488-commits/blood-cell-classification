import os
import random
from collections import Counter
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image

from src.config import (
    IMG_SIZE, BATCH_SIZE, MEAN, STD, SEED,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)


class DatasetWithTransform(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform


def build_dataloaders(data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, val_transform = get_transforms()
    raw = datasets.ImageFolder(data_dir)
    total = len(raw)

    test_size  = int(TEST_SPLIT * total)
    val_size   = int(VAL_SPLIT  * total)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(total, generator=generator).tolist()
    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    train_set = DatasetWithTransform([raw.samples[i] for i in train_idx], train_transform)
    val_set   = DatasetWithTransform([raw.samples[i] for i in val_idx],   val_transform)
    test_set  = DatasetWithTransform([raw.samples[i] for i in test_idx],  val_transform)

    train_labels  = [lbl for _, lbl in train_set.samples]
    class_counts  = Counter(train_labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f'Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')
    return train_loader, val_loader, test_loader
