"""PyTorch dataset, normalisation and augmentation for the S1 LULC patches.
normalize() is also used by the prototype, so inputs match training exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C


def load_stats(path) -> tuple[np.ndarray, np.ndarray]:
    s = json.load(open(path))
    return np.array(s["mean"], np.float32), np.array(s["std"], np.float32)


def normalize(s1_db: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """s1_db: (3, H, W) in dB. Missing pixels (nodata) become 0 after z-scoring."""
    x = s1_db.astype(np.float32)
    missing = x[0] <= C.S1_NODATA + 1
    x = (x - mean[:, None, None]) / std[:, None, None]
    x[:, missing] = 0.0
    return x


class S1LULCDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, data_dir, mean, std, augment: bool = False):
        self.meta = meta.reset_index(drop=True)
        self.dir = Path(data_dir)
        self.mean, self.std = mean, std
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        s1 = np.load(self.dir / r.s1_path).astype(np.float32) / C.INT16_SCALE
        y = np.load(self.dir / r.label_path).astype(np.int64)
        x = normalize(s1, self.mean, self.std)
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            # small per-image gain in dB space: simulates calibration/incidence-angle differences
            x = x + np.random.normal(0, 0.1, (x.shape[0], 1, 1)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
