import numpy as np
from torch.autograd import Variable
import torch

class PTStore:
    def __init__(self):
        self.__dict__['vars'] = {}

    def __call__(self, **kwargs):
        assert len(kwargs)==1, "You must specify just 1 variable to add"
        key, value = kwargs.items()[0]
        setattr(self, key, value)
        return value

    def __setattr__(self, key, value):
        self.__dict__['vars'][key] = value

    def __getattr__(self, key):
        if key=='_vars':
            return self.__dict__['vars']
        if key not in self.__dict__['vars']:
            raise KeyError('Key %s was not found in the pt_store! Forgot to add it?' % key)
        return self.__dict__['vars'][key]

    def __getitem__(self, key):
        if key not in self.__dict__['vars']:
            raise KeyError('Key %s was not found in the pt_store! Forgot to add it?' % key)
        cand = self.__dict__['vars'][key]
        return to_numpy(cand)

    def clear(self):
        self.__dict__['vars'].clear()

def to_numpy(cand):
    if isinstance(cand, Variable):
        return cand.data.cpu().numpy()
    elif isinstance(cand, torch._TensorBase):
        return cand.cpu().numpy()
    elif isinstance(cand, (list, tuple)):
        return map(to_numpy, cand)
    elif isinstance(cand, np.ndarray):
        return cand
    else:
        return np.array([cand])

def to_number(x):
    if isinstance(x, (int, long, float)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x[0]
    return x.data[0]

