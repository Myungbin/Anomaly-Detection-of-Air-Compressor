from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.config.config import cfg


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


def train(train_loader, model, optimizer):
    model.train()
    model.to(cfg.DEVICE)
    for epoch in range(cfg.EPOCHS):
        for data in train_loader:
            data = data.to(cfg.DEVICE)
            optimizer.zero_grad()

            recon_x, mu, logvar = model(data)
            loss = vae_loss(recon_x, data, mu, logvar)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{cfg.EPOCHS}], Loss: {loss.item():.7f}')


def evaluation(test_loader, model, cosine_threshold=0.99):
    model.eval()
    model.to("cpu")
    pred = []
    threshold = []
    with torch.no_grad():
        for data in iter(test_loader):
            data = data.to("cpu")
            prediction, _, _ = model(data)

            cos = nn.CosineSimilarity(dim=1)
            cosine = cos(data, prediction).tolist()
            batch_pred = np.where(np.array(cosine) < cosine_threshold, 1, 0).tolist()

            # mse = np.mean(np.power(data.detach().numpy() - prediction.detach().numpy(), 2), axis=1)
            # batch_pred = np.where(np.array(mse) <= cosine_threshold, 0, 1).tolist()
            # mse = mse.tolist()

            # mae = np.mean(np.absolute(data.detach().numpy() - prediction.detach().numpy()), axis=1)
            # batch_pred = np.where(np.array(mae) <= cosine_threshold, 0, 1).tolist()
            # mae = mae.tolist()

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
