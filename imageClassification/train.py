import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .config import get_config
from .prepare import load_dataloader
from .network import ImageClassiicationModel

cfg = get_config()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, criterion, optimizer, dataloader):
    model.train()
    epoch_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, criterion, optimizer, dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def run(model, criterion, optimizer, train_dataloader, test_dataloader):
    best_valid_loss = float("inf")

    for epoch in range(cfg.EPOCHS):
        start_time = time.time()

        train_loss = train(model, criterion, optimizer, train_dataloader)
        valid_loss = evaluate(model, criterion, optimizer, test_dataloader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), cfg.MODEL_PATH)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tValidation Loss: {valid_loss:.3f}")


def main():
    model = ImageClassiicationModel().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    train_dataloader, test_dataloader = load_dataloader()

    run(model, criterion, optimizer, train_dataloader, test_dataloader)


if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (cfg.DEVICE))

    main()
