import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from config import get_config

cfg = get_config()

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]


def image_loader(image, imsize):
    loader = transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cnn_normalization_mean, std=cnn_normalization_std
            ),
        ]
    )
    image = loader(image).unsqueeze(0)
    return image.to(cfg.DEVICE, torch.float)


def compare_image(axes, image1, image2, interact=False):
    if not interact:
        plt.ioff()

    assert axes.shape == (2,)
    plt.ion()

    image1 = image1.reshape(28, 28).cpu().detach().numpy()
    image2 = image2.reshape(28, 28).cpu().detach().numpy()

    axes[0].imshow(image1, cmap="gray")
    plt.axis("off")

    axes[1].imshow(image2, cmap="gray")
    plt.axis("off")

    if not interact:
        plt.draw()
        plt.pause(0.1)
        plt.ion()

    plt.tight_layout()
    plt.show()
