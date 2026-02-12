import os
import torch
import random
import numpy as np



def set_random_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def init_all(cfg):

    print_dash()
    print(cfg)
    print_dash()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    set_random_seed(cfg['random_seed'])

    os.makedirs(cfg['save_dir'], exist_ok=True)


def print_dash(num=50):
    print(''.join(['-']*num))

