"""Causal missing-value handling and training-only normalization."""

import numpy as np


def causal_fill(values, max_gap=2):
    result = values.copy()
    last = np.zeros(values.shape[1], dtype=np.float32)
    gaps = np.zeros(values.shape[1], dtype=np.int32)
    for index, row in enumerate(values):
        valid = np.isfinite(row)
        gaps = np.where(valid, 0, gaps + 1)
        result[index] = np.where(valid, row, np.where(gaps <= max_gap, last, np.nan))
        last = np.where(valid, row, last)
    return result


def fit_scaler(features, split):
    training = features[split == 0]
    mean = np.nanmean(training, axis=0)
    scale = np.nanstd(training, axis=0)
    return mean.astype('float32'), np.maximum(scale, 1e-4).astype('float32')
