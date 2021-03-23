import os
import torch

from .train import main
from .config import get_config
from .prepare import get_transforms
from .network import ImageClassiicationModel

cfg = get_config()

if not os.path.exists(cfg.MODEL_PATH):
    main()

model = ImageClassiicationModel().to(cfg.DEVICE)
model.load_state_dict(torch.load(cfg.MODEL_PATH))


def image_predict(image):
    class_names = ["cat", "dog", "squirrel"]
    _, transform = get_transforms()
    image = transform(image).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        output = model(image)

        # torch.max(output, 1)
        pred = torch.argmax(output, dim=1)

    class_idx = pred.item()
    return class_names[class_idx]
