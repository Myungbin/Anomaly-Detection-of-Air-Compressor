import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.features import build_features
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

#  데이터 전처리
scaler = MinMaxScaler()

# 데이터 전처리
_train = pd.read_csv(cfg.TRAIN_PATH)
_train = build_features.create_derived_features(_train)
_train = _train.drop("type", axis=1)
scaled_train_data = scaler.fit_transform(_train)

test_data = pd.read_csv(cfg.TEST_PATH)
test_data = build_features.create_derived_features(test_data)
test_data = test_data.drop("type", axis=1)
scaled_test_data = scaler.transform(test_data)


def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


train_dataset, seq_len, n_features = create_dataset(_train)

n_features = _train.shape[1]

dataloader = DatasetLoader(scaled_train_data, scaled_test_data)
train_loader, test_loader = dataloader.load

model = predict_model.LSTMAutoencoder4(n_features, 128, 3)
# model = predict_model.LSTMAutoencoder(n_features, 128, 3, 0.5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(train_loader, model, criterion, optimizer)

prediction = evaluation(test_loader, model)

prediction_to_csv(prediction[0])

# plot
anomaly_plot(test_data, prediction)
