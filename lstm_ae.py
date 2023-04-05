import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.features import build_features
from src.models import predict_model
from src.train.train import train, evaluation
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg
from datetime import datetime

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

#  데이터 전처리
scaler = MinMaxScaler()

# 데이터 전처리
_train = pd.read_csv(r'data\raw\train_data.csv')
_train = build_features.add_air_flow_pressure(_train)
_train = _train.drop("type", axis=1)
scaled_train_data = scaler.fit_transform(_train)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = build_features.add_air_flow_pressure(test_data)
test_data = test_data.drop("type", axis=1)
scaled_test_data = scaler.transform(test_data)

n_features = _train.shape[1]

train_window = []
for i in range(cfg.WINDOW_SIZE, len(scaled_train_data)):
    train_window.append(scaled_train_data[i - cfg.WINDOW_SIZE:i, :])
train_window = np.array(train_window)

test_window = []
for i in range(cfg.WINDOW_SIZE, len(scaled_test_data)):
    test_window.append(scaled_test_data[i - cfg.WINDOW_SIZE:i, :])
test_window = np.array(test_window)


dataloader = DatasetLoader(train_window, test_window)
train_loader, test_loader = dataloader.load


model = predict_model.LSTMAutoencoder(n_features, 64, 4, 0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(train_loader, model, criterion, optimizer)
prediction = evaluation(test_loader, model)

print(prediction)
