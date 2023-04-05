"""
air_inflow: 공기 흡입 유량 (^3/min)
air_end_temp: 공기 말단 온도 (°C)
out_pressure: 토출 압력 (Mpa)
motor_current: 모터 전류 (A)
motor_rpm: 모터 회전수 (rpm)
motor_temp: 모터 온도 (°C)
motor_vibe: 모터 진동 (mm/s)
type: 설비 번호

설비 번호 [0, 4, 5, 6, 7]: 30HP(마력)
설비 번호 1: 20HP
설비 번호 2: 10HP
설비 번호 3: 50HP
"""
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from src.features import build_features
from src.models import predict_model

warnings.filterwarnings(action='ignore')

data = pd.read_csv(r'data\raw\train_data.csv')
data = data.drop("type", axis=1)
window_size = 30
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
n_features = data.shape[1]

X = []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, :])
X = np.array(X)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


num_epochs = 100
hidden_size = 64
latent_dim = 16

train_set = MyDataset(X)
train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
model = predict_model.LSTMAutoencoder(n_features, 64, 2, 0.1)
# model = predict_model.AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()

        # Forward
        output = model(data)
        loss = criterion(output, data[:, -window_size:, :])

        # Backward
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = test_data.drop("type", axis=1)
scaled_test_data = scaler.transform(test_data)

x_test = []
for i in range(window_size, len(scaled_test_data)):
    x_test.append(scaled_test_data[i - window_size:i, :])
x_test = np.array(x_test)

# 테스트 데이터 예측
x_test_tensor = torch.from_numpy(x_test).float()
test_output = model(x_test_tensor)
test_loss = criterion(test_output, x_test_tensor[:, -window_size:, :])
print(f'Test Loss: {test_loss.item():.4f}')

# 이상 탐지
mse = np.mean(np.power(x_test[:, -window_size:, :] - test_output.detach().numpy(), 2), axis=1)
threshold = np.mean(mse) + 3 * np.std(mse)
anomalies = np.where(mse > threshold)[0]
print(len(anomalies))
print(anomalies)

unique_values = np.unique(anomalies)

# 새로운 열 추가 및 모든 값을 0으로 초기화
test = test_data.copy()
test['anomaly'] = 0

# 이상치가 발생한 인덱스에 대해 1로 설정
test.loc[unique_values, 'anomaly'] = 1

submission = pd.read_csv(r'data\raw\answer_sample.csv')
submission["label"] = test["anomaly"]
print(submission.label.value_counts())
submission.to_csv(r'data\submission\submission.csv', index=False)
