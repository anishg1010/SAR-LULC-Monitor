"""Temporal CNN + LSTM model for SAR-only LULC (variant D).

A single SAR image is ambiguous across seasons (a wheat field in June looks like grassland).
This model sees a SEQUENCE of seasonal images of the same place:
  1. a shared U-Net (encoder + decoder, initialised from the single-month model) turns each
     month into a 16-dim feature vector per pixel;
  2. a per-pixel LSTM reads those vectors in time order and keeps a memory of the seasonal
     pattern (lectures 10-11: RNN / LSTM);
  3. a linear head maps the last LSTM state to the 6 classes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import config as C
from .data import normalize
from .model import build_model

SEQ_MONTHS = [1, 4, 7, 10]   # one month per season, chronological


def _decode(base, x):
    feats = base.encoder(x)
    try:
        return base.decoder(feats)          # smp >= 0.5
    except TypeError:
        return base.decoder(*feats)         # older smp


class TemporalUNet(nn.Module):
    def __init__(self, base: nn.Module, hidden: int = 32):
        super().__init__()
        self.base = base                     # smp.Unet; its segmentation head is unused
        feat_ch = base.segmentation_head[0].in_channels
        self.lstm = nn.LSTM(feat_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, C.NUM_CLASSES)

    def forward(self, x):                    # x: (B, T, C, H, W)
        B, T, Cin, H, W = x.shape
        f = _decode(self.base, x.flatten(0, 1))              # (B*T, F, H, W)
        F = f.shape[1]
        seq = f.view(B, T, F, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, F)
        out, _ = self.lstm(seq.float())                      # LSTM in fp32 for stability
        logits = self.head(out[:, -1])                       # last time step
        return logits.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, classes, H, W)


def build_temporal(init_ckpt=None, hidden=32, device="cpu"):
    base = build_model("Unet", "resnet34", pretrained=init_ckpt is None)
    if init_ckpt is not None:
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        base.load_state_dict(ck["state_dict"])
    return TemporalUNet(base, hidden).to(device)


def load_temporal(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TemporalUNet(build_model("Unet", "resnet34", pretrained=False), ck.get("hidden", 32))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32), device


class TemporalDataset(Dataset):
    """One item = the 4 seasonal composites of a patch + its (annual) label.
    shift: cyclic rotation of the season order, so the sequence can END in any season
    (needed at inference, where the last month is the month being mapped).
    shift=None -> random rotation (training)."""

    def __init__(self, meta: pd.DataFrame, data_dir, mean, std, augment=False, shift=0):
        m = meta[meta.keep & meta.month.isin(SEQ_MONTHS)]
        full = m.groupby("patch").month.nunique()
        self.patches = sorted(full[full == len(SEQ_MONTHS)].index)
        self.files = {p: g.set_index("month").s1_path.to_dict() for p, g in m.groupby("patch")}
        self.labels = m.drop_duplicates("patch").set_index("patch").label_path.to_dict()
        self.dir, self.mean, self.std = Path(data_dir), mean, std
        self.augment, self.shift = augment, shift

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        p = self.patches[i]
        k = np.random.randint(len(SEQ_MONTHS)) if self.shift is None else self.shift
        months = SEQ_MONTHS[k:] + SEQ_MONTHS[:k]
        x = np.stack([normalize(np.load(self.dir / self.files[p][m]).astype(np.float32) / C.INT16_SCALE,
                                self.mean, self.std) for m in months])          # (T, C, H, W)
        y = np.load(self.dir / self.labels[p]).astype(np.int64)
        if self.augment:
            r = np.random.randint(4)
            x, y = np.rot90(x, r, (2, 3)), np.rot90(y, r)
            if np.random.rand() < 0.5:
                x, y = x[..., ::-1], y[:, ::-1]
            x = x + np.random.normal(0, 0.1, (x.shape[0], x.shape[1], 1, 1)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def build_sequence(comps, counts, t):
    """Sequence for mapped month index t (comps/counts start 9 months before the first mapped month):
    months t-9, t-6, t-3, t. A history month without Sentinel-1 data is replaced by the nearest
    month (within +-1) that has data, else by the target month itself, so the LSTM never sees an empty image."""
    seq = []
    for k in (t, t + 3, t + 6, t + 9):
        if counts[k] > 0:
            seq.append(comps[k]); continue
        alt = [j for j in (k - 1, k + 1) if 0 <= j < len(comps) and j <= t + 9 and counts[j] > 0]
        seq.append(comps[alt[0]] if alt else comps[t + 9])
    return seq


@torch.no_grad()
def predict_sequence(model, s1_seq_db, mean, std, device):
    """s1_seq_db: list of (3, H, W) dB composites in time order; returns the map for the LAST month."""
    x = torch.from_numpy(np.stack([normalize(s, mean, std) for s in s1_seq_db]))[None].to(device)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred[s1_seq_db[-1][0] <= C.S1_NODATA + 1] = C.IGNORE_INDEX
    return pred
