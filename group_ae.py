import warnings

import pandas as pd
import matplotlib.pyplot as plt
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
from datetime import datetime

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

train_data = pd.read_csv(r'data\raw\train_data.csv')
pca_train = pd.read_csv(r'data\processed\PCA_train.csv')
train_data = build_features.create_derived_features(train_data)
train_data = pd.concat([train_data, pca_train], axis=1)

test_data = pd.read_csv(r'data\raw\test_data.csv')
pca_test = pd.read_csv(r'data\processed\PCA_test.csv')
test_data = build_features.create_derived_features(test_data)
test_data = pd.concat([test_data, pca_test], axis=1)

grouped_train = train_data.groupby('type')

preds = []
for group_name, group_data in grouped_train:
    test_group = test_data[test_data['type'] == group_name]
    group_data = group_data.drop('type', axis=1).values
    test_group = test_group.drop('type', axis=1).values
    scaled_data = scaler.fit_transform(group_data)
    scaled_test_data = scaler.transform(test_group)
    n_features = group_data.shape[1]

    dataloader = DatasetLoader(scaled_data, scaled_test_data)
    train_loader, test_loader = dataloader.load
    model = predict_model.AutoEncoder(input_dim=n_features, latent_dim=128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(train_loader, model, criterion, optimizer)

    prediction = evaluation(test_loader, model)
    preds.append(prediction)
    print(f"finish {group_name}type")

preds = np.concatenate(preds)
prediction_to_csv(preds)

import seaborn as sns

test_data['label'] = preds
test1 = test_data[test_data["label"] == 1]
sns.pairplot(test1[['air_inflow', 'air_end_temp', 'out_pressure', 'air_flow_pressure']])
# sns.pairplot(test1[['motor_rpm', 'motor_temp', 'motor_vibe']])
plt.show()
