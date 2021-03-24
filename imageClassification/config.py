import os
import argparse
from dataclasses import dataclass

import torch


@dataclass
class Config:
    EPOCHS: int
    DEVICE: str
    DATA_DIR: str
    MODEL_PATH: str
    BATCH_SIZE: int
    LEARNING_RATE: float


def get_config():
    parser = argparse.ArgumentParser(description="Image Classification for Cat or Dog")

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--data_dir", default="custom_dataset", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--model_path", default="image_weights", type=str)

    dir_name = os.path.dirname(os.path.abspath(__file__))

    args, unknown = parser.parse_known_args()

    data_dir = os.path.join(dir_name, args.data_dir)
    model_path = os.path.join(dir_name, args.model_path)

    config = Config(
        DATA_DIR=data_dir,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        MODEL_PATH=model_path,
        BATCH_SIZE=args.batch_size,
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    return config
