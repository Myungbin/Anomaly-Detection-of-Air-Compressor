import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
train_data = build_features_op.create_derived_features(train_data)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data['motor_vibe'] = np.log1p(test_data['motor_vibe'])
test_data = build_features_op.create_derived_features(test_data)

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(scaled_train_data)
BGM = BayesianGaussianMixture(n_components=3, random_state=42, verbose=0)
BGM.fit(scaled_train_data)

gmm_data = gmm.score_samples(scaled_train_data)
bgm_data = BGM.score_samples(scaled_train_data)

gmm_test = gmm.score_samples(scaled_test_data)
bgm_test = BGM.score_samples(scaled_test_data)

train_data_ = {'GMM': gmm_data, 'BGM': bgm_data}
test_data_ = {'GMM': gmm_test, 'BGM': bgm_test}

pca_train_df = pd.DataFrame(train_data_, columns=['GMM', 'BGM'])
pca_test_df = pd.DataFrame(test_data_, columns=['GMM', 'BGM'])

pca_train_df.to_csv('GMBM_feature.csv', index=False)
pca_test_df.to_csv('GMBM_test_feature.csv', index=False)
