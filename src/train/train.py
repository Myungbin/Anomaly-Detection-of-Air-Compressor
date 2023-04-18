from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.config.config import cfg


def train(train_loader, model, criterion, optimizer, scheduler=None):
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
            
        if scheduler is not None:
            scheduler.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{cfg.EPOCHS}], Loss: {loss.item():.7f}')


def evaluation(test_loader, model, ths=0.99):
    model.eval()
    model.to("cpu")
    pred = []
    threshold = []
    with torch.no_grad():
        for data in iter(test_loader):
            data = data.to("cpu")
            prediction = model(data)

            # cos = nn.CosineSimilarity(dim=1)
            # cosine = cos(data, prediction).tolist()
            # batch_pred = np.where(np.array(cosine) <= ths, 1, 0).tolist()

            mse = np.mean(np.power(data.detach().numpy() - prediction.detach().numpy(), 2), axis=1)
            batch_pred = np.where(np.array(mse) <= ths, 0, 1).tolist()
            mse = mse.tolist()
            
            # mae = np.mean(np.abs(data.detach().numpy() - prediction.detach().numpy()), axis=1)
            # batch_pred = np.where(np.array(mae) <= ths, 0, 1).tolist()
            # mae = mae.tolist()
            
            threshold += mse
            pred += batch_pred
    return pred, threshold


def prediction_to_csv(prediction):
    submission = pd.read_csv(cfg.SUBMISSION_PATH)
    submission["label"] = prediction
    print(submission.label.value_counts())
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission.to_csv(
        f'data/submission/{current_time}submission.csv', index=False)

    return submission