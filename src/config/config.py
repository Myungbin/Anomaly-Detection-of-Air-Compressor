import os
import random

import numpy as np
import torch


class Config:
    TRAIN_PATH = r'data\raw\train_data.csv'
    TEST_PATH = r'data\raw\test_data.csv'
    SUBMISSION_PATH = r'data\raw\answer_sample.csv'

    EPOCHS = 400
    BATCH_SIZE = 512
    SEED = 1103
    WINDOW_SIZE = 20

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def seed_everything(seed):
    """
    시드 값을 설정하여 재현성을 보장합니다.

    Args:
        seed (int): 시드 값입니다.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
