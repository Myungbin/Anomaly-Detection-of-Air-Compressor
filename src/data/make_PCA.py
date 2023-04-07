import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

from src.features import build_features

warnings.filterwarnings(action='ignore')

df = pd.read_csv(r'data\raw\train_data.csv')
test = pd.read_csv(r'data\raw\test_data.csv')

df = build_features.add_air_flow_pressure(df)
test = build_features.add_air_flow_pressure(test)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
test_scaled = scaler.transform(test)

train = pd.DataFrame(data_scaled, columns=df.columns)
test = pd.DataFrame(test_scaled, columns=df.columns)
df_train = train.drop(['type'], axis=1)
df_test = test.drop(['type'], axis=1)

columns_for_pca1 = ['air_inflow', 'air_end_temp',
                    'out_pressure', 'air_flow_pressure']
columns_for_pca2 = ['motor_current', 'motor_rpm', 'motor_temp']

pca1 = PCA(n_components=1, random_state=42)
pca2 = PCA(n_components=1, random_state=42)

pca_result1_train = pca1.fit_transform(df_train[columns_for_pca1])
pca_result2_train = pca2.fit_transform(df_train[columns_for_pca2])

df_pca1_train = pd.DataFrame(pca_result1_train, columns=['pca_result1'])
df_pca2_train = pd.DataFrame(pca_result2_train, columns=['pca_result2'])

df_merged_train = pd.concat([df_pca1_train, df_pca2_train], axis=1)

pca_result1_test = pca1.transform(df_test[columns_for_pca1])
pca_result2_test = pca2.transform(df_test[columns_for_pca2])

df_pca1_test = pd.DataFrame(pca_result1_test, columns=['pca_result1'])
df_pca2_test = pd.DataFrame(pca_result2_test, columns=['pca_result2'])

df_merged_test = pd.concat([df_pca1_test, df_pca2_test], axis=1)

print(df_merged_train)
print(df_merged_test)

df_merged_train.to_csv("PCA_train.csv", index=False)
df_merged_test.to_csv("PCA_test.csv", index=False)
