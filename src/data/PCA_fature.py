from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.features import build_features

warnings.filterwarnings(action='ignore')

df = pd.read_csv(r'data\raw\train_data.csv')
test = pd.read_csv(r'data\raw\test_data.csv')

df = build_features.create_derived_features(df)
test = build_features.create_derived_features(test)

df = df.drop('type', axis=1)
test = test.drop('type', axis=1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
test_scaled = scaler.transform(test)

train = pd.DataFrame(data_scaled, columns=df.columns)
test = pd.DataFrame(test_scaled, columns=df.columns)


def pca_transform(train, test, n_components=None):
    """
    PCA를 적용하여 주성분으로 변환된 데이터를 반환합니다.
    n_components가 None이면, 모든 주성분을 사용합니다.
    """
    # PCA 적용
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train)
    pca_train = pca.transform(train)
    pca_test = pca.transform(test)

    return pca_train, pca_test


def select_n_components(data, max_components=None, explained_variance_ratio_threshold=0.95):
    """
    최적의 주성분 개수를 선택합니다.
    max_components가 None이면, 모든 주성분을 대상으로 선택합니다.
    explained_variance_ratio_threshold는 설명력 비율의 임계값으로,
    이 값 이상의 설명력 비율을 가지는 주성분을 선택합니다.
    """

    # PCA 적용
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(data)

    # 설명력 비율 계산
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

    # 최적의 주성분 개수 선택
    n_components = np.argmax(
        explained_variance_ratio_cumsum >= explained_variance_ratio_threshold) + 1

    return n_components


n_components = select_n_components(train)
print(n_components)
pca_train, pca_test = pca_transform(train, test, n_components)

pca_train_df = pd.DataFrame(pca_train, columns=['pca_result1', 'pc_result2', 'pca_result3'])
pc_test_df = pd.DataFrame(pca_test, columns=['pca_result1', 'pc_result2', 'pca_result3'])

pca_train_df.to_csv("PCA_train_26_feature.csv", index=False)
pc_test_df.to_csv("PCA_test_26_feature.csv", index=False)

# df_train = train.drop(['type'], axis=1)
# df_test = test.drop(['type'], axis=1)

# columns_for_pca1 = ['air_inflow', 'air_end_temp', 'out_pressure', 'air_flow_pressure']
# columns_for_pca2 = ['motor_current', 'motor_rpm', 'motor_temp']

# pca1 = PCA(n_components=1, random_state=42)
# pca2 = PCA(n_components=1, random_state=42)

# pca_result1_train = pca1.fit_transform(df_train[columns_for_pca1])
# pca_result2_train = pca2.fit_transform(df_train[columns_for_pca2])

# df_pca1_train = pd.DataFrame(pca_result1_train, columns=['pca_result1'])
# df_pca2_train = pd.DataFrame(pca_result2_train, columns=['pca_result2'])

# df_merged_train = pd.concat([df_pca1_train, df_pca2_train], axis=1)

# pca_result1_test = pca1.transform(df_test[columns_for_pca1])
# pca_result2_test = pca2.transform(df_test[columns_for_pca2])

# df_pca1_test = pd.DataFrame(pca_result1_test, columns=['pca_result1'])
# df_pca2_test = pd.DataFrame(pca_result2_test, columns=['pca_result2'])

# df_merged_test = pd.concat([df_pca1_test, df_pca2_test], axis=1)

# print(df_merged_train)
# print(df_merged_test)

# df_merged_train.to_csv("PCA_train.csv", index=False)
# df_merged_test.to_csv("PCA_test.csv", index=False)
