import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from FRUFS import FRUFS
from xgboost import XGBClassifier, XGBRegressor

from src.features import build_features_op
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

train_data = pd.read_csv(r'data\raw\train_data.csv')
train_data['motor_vibe'] = np.log1p(train_data['motor_vibe'])
pca_train = pd.read_csv(r'data\processed\PCA_train_26_feature.csv')
gmbm_train = pd.read_csv(r'data\processed\GMBM_train_feature.csv')
train_data = build_features_op.create_derived_features(train_data)
train_data = pd.concat([train_data, pca_train, gmbm_train], axis=1)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data['motor_vibe'] = np.log1p(test_data['motor_vibe'])
pca_test = pd.read_csv(r'data\processed\PCA_test_26_feature.csv')
gmbm_test = pd.read_csv(r'data\processed\GMBM_test_feature.csv')
test_data = build_features_op.create_derived_features(test_data)
test_data = pd.concat([test_data, pca_test, gmbm_test], axis=1)

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)

grouped_train = scaled_train_data.groupby('type')

preds = []
ths = []
for group_name, group_data in grouped_train:
    test_group = scaled_test_data[scaled_test_data['type'] == group_name]
    train_group = group_data.drop('type', axis=1).values
    test_group = test_group.drop('type', axis=1).values

    n_features = train_group.shape[1]
    print(n_features)
    dataloader = DatasetLoader(train_group, test_group)
    train_loader, test_loader = dataloader.load
    model = predict_model.AutoEncoder(input_dim=n_features, latent_dim=128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(train_loader, model, criterion, optimizer)

    prediction, cosine = evaluation(test_loader, model)
    preds.append(prediction)
    ths.append(cosine)
    print(f"finish {group_name}type")

threshold = np.concatenate(ths)
preds = np.concatenate(preds)
prediction_to_csv(preds)
