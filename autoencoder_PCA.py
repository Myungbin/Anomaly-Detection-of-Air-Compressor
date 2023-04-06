import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

from src.features import build_features
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg
from datetime import datetime

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

# scaler = MinMaxScaler()

# # 데이터 전처리
# data = pd.read_csv(r'data\raw\train_data.csv')
# data = build_features.add_air_flow_pressure(data)
# data = data.drop("type", axis=1)
# scaled_data = scaler.fit_transform(data)
# test_data = pd.read_csv(r'data\raw\test_data.csv')
# test_data = build_features.add_air_flow_pressure(test_data)
# test_data = test_data.drop("type", axis=1)
# scaled_test_data = scaler.transform(test_data)

pca_train = pd.read_csv('PCA_train.csv')
pca_test = pd.read_csv('PCA_test.csv')

pca_train = pca_train.values
pca_test = pca_test.values
n_features = pca_train.shape[1]

# 데이터 로더
dataloader = DatasetLoader(pca_train, pca_test)
train_loader, test_loader = dataloader.load

# 학습 파라미터
model = predict_model.AutoEncoder(input_dim=n_features, latent_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
train(train_loader, model, criterion, optimizer)

# 예측
prediction = evaluation(test_loader, model)

# 제출
prediction_to_csv(prediction)

# plot
# anomaly_plot(pca_test, prediction)
