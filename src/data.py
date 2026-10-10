"""CIFAR-10 data loading with train-time augmentation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def get_loaders(data_dir: str = "data", batch_size: int = 128,
                num_workers: int = 2, smoke: bool = False,
                smoke_samples: int = 512):
    """Return (train_loader, test_loader). `smoke` uses a tiny subset for tests."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False,
                               download=True, transform=test_tf)
    if smoke:
        train_ds = Subset(train_ds, range(smoke_samples))
        test_ds = Subset(test_ds, range(smoke_samples // 4))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Revert CIFAR-10 normalisation for visualisation."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return tensor * std + mean
