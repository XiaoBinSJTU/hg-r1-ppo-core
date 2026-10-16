"""Deterministic seeds and GPU-first execution."""

import gc
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(mode='auto'):
    if mode in ('auto', 'cuda') and torch.cuda.is_available():
        try:
            test = torch.ones(2, device='cuda')
            (test * test).sum().item()
            return torch.device('cuda'), 'GPU'
        except RuntimeError:
            if mode == 'cuda':
                raise RuntimeError('Requested GPU is unavailable') from None
    if mode == 'cuda':
        raise RuntimeError('Requested GPU is unavailable')
    return torch.device('cpu'), 'CPU fallback' if mode == 'auto' else 'CPU'


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
