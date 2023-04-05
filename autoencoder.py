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

warnings.filterwarnings(action='ignore')
scaler = MinMaxScaler()

# 데이터 전처리
data = pd.read_csv(r'data\raw\train_data.csv')
data = data.drop("type", axis=1)
scaled_data = scaler.fit_transform(data)
test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = test_data.drop("type", axis=1)
scaled_test_data = scaler.transform(test_data)

# 데이터 로더
dataloader = DatasetLoader(scaled_data, scaled_test_data)
train_loader, test_loader = dataloader.load

# 학습 파라미터
model = predict_model.AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(train_loader, model, criterion, optimizer)
prediction = evaluation(test_loader, model)

# 제출
submission = pd.read_csv(r'data\raw\answer_sample.csv')
submission["label"] = prediction
print(submission.label.value_counts())
submission.to_csv(r'data\submission\submission.csv', index=False)
anomaly_plot(test_data, prediction)
