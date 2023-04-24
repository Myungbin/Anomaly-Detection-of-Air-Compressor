import pandas as pd
from sklearn.covariance import MinCovDet
from src.features import build_features
import torch

data = pd.read_csv(r'data\raw\train_data.csv')
test_data = pd.read_csv(r'data\raw\test_data.csv')
data = build_features.add_air_flow_pressure(data)
test_data = build_features.add_air_flow_pressure(test_data)

mcd_model = MinCovDet(support_fraction=1, random_state=42)
mcd_model.fit(data)


def get_pred_label(model, x, k):
    prob = abs(mcd_model.mahalanobis(x))
    prob2 = torch.tensor(prob, dtype=torch.float)
    topk_indices = torch.topk(prob2, k=k, largest=True).indices

    pred = torch.zeros(len(x), dtype=torch.long)
    pred[topk_indices] = 1
    return pred.tolist(), prob2.tolist()


val_pred, val_prob = get_pred_label(mcd_model, data, 118)
data["Class"] = val_pred  # 라벨값 지정
print(data)
