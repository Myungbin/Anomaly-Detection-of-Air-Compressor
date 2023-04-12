import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.features import build_features_op, utils, build_features
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.visualization.visual import anomaly_plot
from src.config.config import seed_everything, cfg
from datetime import datetime

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

# 데이터 전처리
train_data = pd.read_csv(r'data\raw\train_data.csv')
train_data = build_features.create_derived_features(train_data)
train_data = utils.outlier_z_score_filter_df(train_data)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = build_features.create_derived_features(test_data)

# pca_train = pd.read_csv(r'data\processed\PCA_train_26_feature.csv')
# gmbm_train = pd.read_csv(r'data\processed\GMBM_train_feature.csv')
# pca_test = pd.read_csv(r'data\processed\PCA_test_26_feature.csv')
# gmbm_test = pd.read_csv(r'data\processed\GMBM_test_feature.csv')
# train_data = pd.concat([pca_train, gmbm_train], axis=1)
# test_data = pd.concat([pca_test, gmbm_test], axis=1)

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)

n_features = scaled_train_data.shape[1]
print(n_features)

scaled_train_data = scaled_train_data.values
scaled_test_data = scaled_test_data.values

# 데이터 로더
dataloader = DatasetLoader(scaled_train_data, scaled_test_data)
train_loader, test_loader = dataloader.load

# 학습 파라미터
model = predict_model.AutoEncoder(input_dim=n_features, latent_dim=64)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
train(train_loader, model, criterion, optimizer)

# 예측
train_prediction, train_cosine = evaluation(train_loader, model)
prediction, test_cosine = evaluation(test_loader, model, max(train_cosine))

# 제출
prediction_to_csv(prediction)

# plot
# anomaly_plot(test_data, prediction)

"""
predictions = pd.read_csv(
    r"C:\MB_Project\project\Competition\Anomaly-Detection-of-Air-Compressor\data\submission\20230410_115200submission.csv")
predictions00 = predictions[:1296]
predictions01 = predictions[1296:2403]
predictions02 = predictions[2403:3501]
predictions03 = predictions[3501:4419]
predictions04 = predictions[4419:5337]
predictions05 = predictions[5337:6083]
predictions06 = predictions[6083:6831]
predictions07 = predictions[6831:]

print(len(predictions00[predictions00['label'] == 1]))
print(len(predictions01[predictions01['label'] == 1]))
print(len(predictions02[predictions02['label'] == 1]))
print(len(predictions03[predictions03['label'] == 1]))
print(len(predictions04[predictions04['label'] == 1]))
print(len(predictions05[predictions05['label'] == 1]))
print(len(predictions06[predictions06['label'] == 1]))
print(len(predictions07[predictions07['label'] == 1]))
predict_type = [predictions00, predictions01, predictions02, predictions03, predictions04, predictions05, predictions06,
                predictions07]

for type in predict_type:
    plt.plot(type['label'])
    plt.show()

"""
