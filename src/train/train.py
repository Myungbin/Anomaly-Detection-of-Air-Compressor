import torch
import torch.nn as nn
import numpy as np

from src.config.config import cfg


def train(train_loader, model, criterion, optimizer):
    for epoch in range(cfg.EPOCHS):
        for data in train_loader:
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{cfg.EPOCHS}], Loss: {loss.item():.4f}')


def evaluation(test_loader, model):
    model.eval()
    pred = []
    with torch.no_grad():
        for data in iter(test_loader):
            prediction = model(data)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cosine = cos(data, prediction).tolist()
            batch_pred = np.where(np.array(cosine) < 0.95, 1, 0).tolist()

            # mse = np.mean(np.power(data.detach().numpy() - prediction.detach().numpy(), 2), axis=1)
            # threshold = np.mean(mse) + 3 * np.std(mse)
            # batch_pred = np.where(np.array(mse) < threshold, 0, 1).tolist()
            pred += batch_pred
    return pred
