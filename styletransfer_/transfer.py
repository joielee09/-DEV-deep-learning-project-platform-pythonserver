import PIL
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from config import get_config
from prepare import image_loader
from network import get_style_transfer_vgg_model

cfg = get_config()


def transfer():
    style_layers = ["conv_1", "conv_3", "conv_5", "conv_7", "conv_9"]
    content_layers = ["conv_4"]

    style_img = image_loader(cfg.STYLE_IMAGE, (512, 640))
    content_img = image_loader(cfg.CONTENT_IMAGE, (512, 640))

    model, style_losses, content_losses = get_style_transfer_vgg_model(
        style_layers, content_layers, style_img, content_img
    )

    input_img = torch.empty_like(content_img).uniform_(0, 1).to(cfg.DEVICE)
    optimizer = optim.Adam([input_img.requires_grad_()])
    # LBFGS 사용할 때는 학습이 안 됐는데 Adam 사용하니깐 너무 잘된다.
    # 흠.. output값은 그렇게 썩 좋지는 않다.

    run = [0]  # 이해 안감.

    while run[0] < 3000:

        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        content_score = 0
        style_score = 0

        for cl in content_losses:
            content_score += cl.loss

        for sl in style_losses:
            style_score += sl.loss

        style_score *= 1e4
        loss = content_score + style_score
        loss.backward()

        run[0] += 1
        if run[0] % 100 == 0:
            print(
                f"[ Step: {run[0]} / Content loss: {content_score.item()} / Style loss: {style_score.item()}]"
            )

        optimizer.step()

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
    img = transfer()
    save_image(img.cpu().detach()[0], "output.png")
