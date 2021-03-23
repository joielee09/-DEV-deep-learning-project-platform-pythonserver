import os

import torch
from torchvision import datasets, transforms

from .config import get_config

cfg = get_config()

IMG_SIZE = (224, 224)
IMG_MEAN, IMG_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms():
    transforms_train = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )

    return transforms_train, transforms_test


def load_dataset():
    train_path = os.path.join(cfg.DATA_DIR, "train")
    test_path = os.path.join(cfg.DATA_DIR, "test")

    transforms_train, transforms_test = get_transforms()

    train_dataset = datasets.ImageFolder(train_path, transforms_train)
    test_dataset = datasets.ImageFolder(test_path, transforms_test)

    return train_dataset, test_dataset


def load_dataloader():
    train_dataset, test_dataset = load_dataset()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4
    )

    return train_dataloader, test_dataloader
