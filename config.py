import os
from custom_pytorch.custom_config import Config
import random
import torch
import numpy as np

CONFIG = Config(train_size=2000,
                train_selection='hi_weight',
                train_replacement=True,
                valid_size=500,  # by random selection, will not participate at all in training
                batch_size=4,
                valid_batch_size=6,
                random_seed=42,
                lr=1e-2,
                momentum=0.9,
                restart_period=50,
                weight_decay=0.01,
                im_size=256,
                warm_start_from='',
                identifier='SE_Xception_XUnet',
                net_params={'resolution': 3, 'depth': 3}
                )


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CONFIG.random_seed)
