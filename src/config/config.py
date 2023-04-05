import os
import random

import numpy as np
import torch


class Config:
    EPOCHS = 100
    BATCH_SIZE = 128
    SEED = 1103
    SUBMISSION_PATH = r'data\raw\answer_sample.csv'


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
