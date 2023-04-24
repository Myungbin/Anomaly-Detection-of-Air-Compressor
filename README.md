# Anomaly Detection of Air Compressor

## 제4회 2023 연구개발특구 AI SPARK 챌린지 - 공기압축기 이상 판단

### 산업용 공기압축기의 이상 유무를 비지도학습 방식을 이용하여 판정  
산업용 공기압축기 및 회전기기에서 모터 및 심부 온도, 진동, 노이즈 등은 기기 피로도에 영향을 주는 요소이며, 피로도 증가는 장비가 고장에 이르는 원인이 됩니다.  
피로도 증가 시 데이터 학습을 통해 산업기기 이상 전조증상을 예측하여 기기 고장을 예방하고 그로 인한 사고를 예방하는 모델을 개발하는 것이 이번 대회의 목표입니다.  
https://aifactory.space/competition/detail/2226


## Project structure
```
Anomaly-Detection-of-Air-Compressor
├─ .gitignore
├─ data  
│  ├─ processed
│  │  └─ preprocess feature
│  ├─ raw
│  │  └─ raw data
│  └─ submission
├─ models
│  └─ model file
├─ notebooks
│  └─ EDA.ipynb
├─ src
│  ├─ config  # hyperparameter & path
│  │  └─ config.py  
│  ├─ data  
│  │  └─ make_dataset.py
│  ├─ features  # Scripts to turn raw data into features for modeling
│  │  ├─ build_features.py  
│  │  ├─ build_features_final.py
│  │  └─ utils.py  # feature utils
│  ├─ models  
│  │  ├─ loss.py  # loss function
│  │  └─ predict_model.py  # models
│  ├─ train
│  │  └─ train.py  # train, test step
│  ├─ visualization
│  │  └─ visual.py 
│  └─ __init__.py
├─ main.py  
├─ main_group.py 
├─ README.md
└─ validation.py
```

##  Getting Started <a name = "getting_started"></a>
`Python 3.9.13` 
```
python -m venv {project name}
{project name}\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Dataset
파생변수는 features/build_features.py 에 생성하여 적용하였습니다.  
`MinMaxScaler`를 이용하여 스케일링을 하였습니다.  
```
1. 데이터 구성 항목은 다음과 같습니다.
air_inflow: 공기 흡입 유량 (^3/min)
air_end_temp: 공기 말단 온도 (°C)
out_pressure: 토출 압력 (Mpa)
motor_current: 모터 전류 (A)
motor_rpm: 모터 회전수 (rpm)
motor_temp: 모터 온도 (°C)
motor_vibe: 모터 진동 (mm/s)
type: 설비 번호

2. 설비별로 다음의 특성을 갖습니다.
설비 번호 [0, 4, 5, 6, 7]: 30HP(마력)
설비 번호 1: 20HP
설비 번호 2: 10HP
설비 번호 3: 50HP
```

## Model 
프로젝트에서는 Autoencder 기반 모델을 사용하여 이상탐지를 하였습니다.  
기본적으로 모델은 다음과 같은 구조를 따릅니다.  
input data는 Encoder를 통과하여 잠재공간으로 표현되고, Decoder를 통해 원본 데이터를 재생성 합니다.  
`MSE loss`, `Adam optimizer`를 사용하였습니다.
```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x
```

## Train & Inference
추론 단계에서, threshold(cosine similarity, mse, mae)를 선택할 수 있습니다.  
기본적으로 cosine similarity를  사용하였으며, 추론에 train data의 `cosine similarity의 최솟값`을 사용합니다.  
만약 MSE, MAE를 사용할때는 tarin threshold의 최댓값을 사용해야 합니다.
```python
train_prediction, train_cosine = evaluation(train_loader, model)
prediction, test_cosine = evaluation(test_loader, model, min(train_cosine))

## main.py 
```

## Config
```
EPOCHS = 1000
BATCH_SIZE = 512
SEED = 1103
GAMMA = 0.7
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Result
다음의 정상 / 이상치 개수가 재현되어야 합니다.  
```
Normal: 7045  
Anomaly: 344
```
