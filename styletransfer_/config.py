import os
import argparse
from PIL import Image
from dataclasses import dataclass

import torch


@dataclass
class Config:
    DEVICE: str
    STYLE_IMAGE: Image
    CONTENT_IMAGE: Image
    LEARNING_RATE: float


def get_config():
    parser = argparse.ArgumentParser(description="Style Transfer Using Cnn")

    parser.add_argument("--style", default="./cat.jpg", type=str)
    parser.add_argument("--content", default="./gunmo.jpg", type=str)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    style_image = Image.open(args.style)
    content_image = Image.open(args.content)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        DEVICE=device,
        LEARNING_RATE=args.lr,
        STYLE_IMAGE=style_image,
        CONTENT_IMAGE=content_image,
    )

    return config
