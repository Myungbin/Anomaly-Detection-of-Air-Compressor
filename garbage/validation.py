import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

from src.features import build_features
from src.models import predict_model
from src.train.trainer import Trainer, evaluation
from src.data.make_dataset import DatasetLoader
from src.config.config import seed_everything, cfg
from sklearn.model_selection import train_test_split

seed_everything(cfg.SEED)

scaler = MinMaxScaler()
data = pd.read_csv(r'data\raw\train_data.csv')
data = build_features.add_air_flow_pressure(data)

train, validation = train_test_split(data, test_size=0.2, random_state=42)

# 학습 및 검증
grouped_train = data.groupby('type')

preds = []
for group_name, group_data in grouped_train:
    test_group = validation[validation['type'] == group_name]
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

    train_model = Trainer(model, criterion, optimizer)
    train_model.fit(train_loader, test_loader)
    prediction = evaluation(test_loader, model)
    preds.append(prediction)

    print(f"finish {group_name}type")

preds = np.concatenate(preds)
lbael = [0] * validation.shape[0]

f1 = f1_score(lbael, preds, average="macro")
print("validation F1 score: ", f1)
