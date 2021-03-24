import torch
import torch.nn as nn
from torchvision import models


class ImageClassiicationModel(nn.Module):
    def __init__(self):
        super(ImageClassiicationModel, self).__init__()
        self.backbone = models.resnet34(pretrained=True, progress=False)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 3)

    def forward(self, images):
        return self.backbone(images)
