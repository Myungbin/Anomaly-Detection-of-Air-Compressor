import warnings

import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import VotingClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from src.features import utils, build_features_final, build_features
from src.models import predict_model
from src.train.train import train, evaluation, prediction_to_csv
from src.data.make_dataset import DatasetLoader
from src.config.config import seed_everything, cfg

warnings.filterwarnings(action='ignore')
seed_everything(cfg.SEED)

scaler = MinMaxScaler()

train_data = pd.read_csv(r'data\processed\PCA_train_final_feature.csv')
test_data = pd.read_csv(r'data\processed\PCA_test_final_feature.csv')

kmeans = KMeans(n_clusters=50, random_state=0).fit(train_data)

# labels = kmeans.labels_
labels = kmeans.predict(test_data)
centers = kmeans.cluster_centers_
distances = np.linalg.norm(test_data - centers[labels], axis=1)
threshold = np.percentile(distances, 95)
anomalies = test_data[distances > threshold]

result = np.zeros(len(test_data))
result[[anomalies.index]] = 1

prediction_to_csv(result)

"""submission = pd.read_csv(cfg.SUBMISSION_PATH)
submission['label'] = 0
submission.loc[anomalies.index, 'label'] = 1"""

# distances = [np.linalg.norm(x - y) for x, y in zip(test_data.values, centers[labels])]
# threshold = np.percentile(distances, 95)
# anomalies = np.zeros(len(test_data))
# anomalies1 = test_data[distances > threshold]
# anomalies[[anomalies1.index]] = 1
