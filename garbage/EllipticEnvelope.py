import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import EllipticEnvelope

from src.features import build_features_op, utils, build_features_optim, build_features
from src.config.config import seed_everything, cfg
from src.train.train import prediction_to_csv

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

# 데이터 전처리
train_data = pd.read_csv(r'data\raw\train_data.csv')
# train_data = utils.outlier_z_score_filter_df(train_data)
train_data = build_features.create_derived_features(train_data)
train_data = train_data.drop('type', axis=1)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = build_features.create_derived_features(test_data)
test_data = test_data.drop('type', axis=1)

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)

n_features = scaled_train_data.shape[1]
print(n_features)

envelope = EllipticEnvelope(contamination=0.001, random_state=42)
envelope.fit(scaled_train_data)
pred = np.where(envelope.predict(scaled_test_data) == -1, 1, 0)

prediction_to_csv(pred)

