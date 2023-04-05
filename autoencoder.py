import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

from src.features import build_features
from src.models import predict_model
from src.train.train import train, evaluation
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg
from datetime import datetime

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = StandardScaler()

# 데이터 전처리
data = pd.read_csv(r'data\raw\train_data.csv')
data = build_features.add_air_flow_pressure(data)
data = data.drop("type", axis=1)
scaled_data = scaler.fit_transform(data)
test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = build_features.add_air_flow_pressure(test_data)
test_data = test_data.drop("type", axis=1)
scaled_test_data = scaler.transform(test_data)
n_features = data.shape[1]
print(n_features)
# 데이터 로더
dataloader = DatasetLoader(scaled_data, scaled_test_data)
train_loader, test_loader = dataloader.load

# 학습 파라미터
model = predict_model.AutoEncoder(input_dim=n_features, latent_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(train_loader, model, criterion, optimizer)
prediction = evaluation(test_loader, model)


def prediction_to_csv(prediction):
    submission = pd.read_csv(r'data\raw\answer_sample.csv')
    submission["label"] = prediction
    print(submission.label.value_counts())
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission.to_csv(f'data/submission/{current_time}submission.csv', index=False)


prediction_to_csv(prediction)
anomaly_plot(test_data, prediction)
