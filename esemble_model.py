import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from src.features import build_features_op, utils, build_features_optim, build_features
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.config.config import seed_everything, cfg

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

# 데이터 전처리
train_data = pd.read_csv(r'data\raw\train_data.csv')
# add_train = pd.read_csv(r'data/processed/robust.csv')
# train_data = pd.concat([train_data, add_train], axis=0)
# train_data = utils.outlier_z_score_filter_df(train_data)
train_data = build_features_optim.create_derived_features(train_data)
train_data = train_data.drop('type', axis=1)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data_raw = test_data.copy()
test_data = build_features_optim.create_derived_features(test_data)
test_data = test_data.drop('type', axis=1)

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
model = predict_model.AutoEncoder(input_dim=n_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
train(train_loader, model, criterion, optimizer)

# 예측
train_prediction, train_cosine = evaluation(train_loader, model)
print(min(train_cosine))
prediction, test_cosine = evaluation(test_loader, model, min(train_cosine))


isolation_forest = IsolationForest(contamination=0.01)
isolation_forest.fit(scaled_train_data)

one_class_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
one_class_svm.fit(scaled_train_data)


isolation_forest_scores = isolation_forest.score_samples(scaled_test_data)
one_class_svm_scores = one_class_svm.score_samples(scaled_test_data)

# 제출
# prediction_to_csv(prediction)
