"""Model factory, loss and segmentation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from . import config as C


def build_model(arch: str = "Unet", encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """arch: 'Unet' | 'DeepLabV3Plus' | 'UnetPlusPlus' ...
    ImageNet weights help even for SAR: low-level edge/texture filters transfer,
    and the first conv is adapted to our 3 SAR channels."""
    Arch = getattr(smp, arch)
    return Arch(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                in_channels=len(C.S1_BANDS), classes=C.NUM_CLASSES)


class CEDiceLoss(nn.Module):
    """Weighted cross-entropy (handles class imbalance) + soft Dice (optimises overlap,
    closer to the IoU metric). Ignore pixels are excluded from both terms."""

    def __init__(self, weight=None, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=C.IGNORE_INDEX)
        self.dw = dice_weight

    def forward(self, logits, y):
        ce = self.ce(logits, y)
        valid = (y != C.IGNORE_INDEX)
        prob = logits.softmax(1) * valid.unsqueeze(1)
        oh = F.one_hot(y.clamp(max=C.NUM_CLASSES - 1), C.NUM_CLASSES).permute(0, 3, 1, 2) * valid.unsqueeze(1)
        inter = (prob * oh).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return ce + self.dw * dice


def confusion(pred: torch.Tensor, y: torch.Tensor, n: int = C.NUM_CLASSES) -> np.ndarray:
    m = y != C.IGNORE_INDEX
    idx = y[m] * n + pred[m]
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy()


def metrics(cm: np.ndarray) -> dict:
    """cm[true, pred]. Returns OA, mIoU, per-class IoU, precision, recall, F1."""
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = cm.sum(1) > 0
    return {"OA": tp.sum() / max(cm.sum(), 1), "mIoU": iou[present].mean(),
            "mF1": f1[present].mean(), "IoU": iou, "F1": f1, "precision": prec, "recall": rec}
