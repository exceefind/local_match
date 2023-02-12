
import os
import shutil
import time
import pprint
import torch
import numpy as np
import os.path as osp
import random

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    for k, v in file_dict.items():
        if k not in model_dict:
            print(k)
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model