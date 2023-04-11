from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.config.config import cfg


def train(train_loader, model, criterion, optimizer):
    model.train()
    model.to(cfg.DEVICE)
    for epoch in range(cfg.EPOCHS):
        for data in train_loader:
            data = data.to(cfg.DEVICE)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{cfg.EPOCHS}], Loss: {loss.item():.7f}')


def evaluation(test_loader, model):
    model.eval()
    model.to("cpu")
    pred = []
    threshold = []
    with torch.no_grad():
        for data in iter(test_loader):
            data = data.to("cpu")
            prediction = model(data)

            cos = nn.CosineSimilarity(dim=1)
            cosine = cos(data, prediction).tolist()
            batch_pred = np.where(np.array(cosine) < 0.999692, 1, 0).tolist()

            # mse = np.mean(np.power(data.detach().numpy() - prediction.detach().numpy(), 2), axis=1)
            # threshold = np.mean(mse) + 3 * np.std(mse)
            # batch_pred = np.where(np.array(mse) <= 0.0013142208, 0, 1).tolist()

            threshold += cosine
            pred += batch_pred
    return pred, threshold


def prediction_to_csv(prediction):
    submission = pd.read_csv(cfg.SUBMISSION_PATH)
    submission["label"] = prediction
    print(submission.label.value_counts())
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission.to_csv(
        f'data/submission/{current_time}submission.csv', index=False)
