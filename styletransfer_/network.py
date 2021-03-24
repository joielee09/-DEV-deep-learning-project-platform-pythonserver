import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import get_config

cfg = get_config()


def gram_matrix(features):
    a, b, c, d = features.size()  # batch는 1이라고 가정, 그래야 잘 작동할 거 같음...
    features = features.view(a * b, c * d)

    G = torch.mm(features, features.t())  # 크기는 차원의 수하고 비례한다.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, feature):
        G = gram_matrix(feature)
        self.loss = F.mse_loss(G, self.target)
        return feature


# Loss값을 계산하고 다음 Layer를 위해 feature는 그대로 전달한다.
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # target값은 update를 하지 않는다.

    def forward(self, feature):
        self.loss = F.mse_loss(feature, self.target)
        return feature


def get_style_transfer_vgg_model(
    style_layers: list, content_layers: list, style_img, content_img
):
    cnn = models.vgg19(pretrained=True).features.to(cfg.DEVICE).eval()

    style_losses = []
    content_losses = []

    last_layer = sorted(style_layers + content_layers)[-1]

    model = nn.Sequential()

    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        # 설정한 style layer까지의 결과를 이용해 style loss를 계산
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in content_layers:
            target_feature = model(content_img).detach()
            content_loss = ContentLoss(target_feature)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name == last_layer:
            break
    print(len(style_losses), len(content_losses))
    return model, style_losses, content_losses
