"""Prototype back end: fetch 12 monthly Sentinel-1 composites for a location,
run the trained model, and compute class percentages and changes.
Shared by the Streamlit app (app.py) and the case-study notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C, gee
from .data import normalize
from .model import build_model


def month_list(year: int, month: int, n: int = 12) -> list[tuple[int, int]]:
    out = []
    for k in range(n):
        m = month - 1 + k
        out.append((year + m // 12, m % 12 + 1))
    return out


def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32), device


def area_grid(lat: float, lon: float, size_px: int):
    """Square area of size_px x size_px pixels (10 m) centred on lat/lon."""
    return gee.region_grid(lat, lon, patches_per_side=1, patch=size_px)


def fetch_month(lat, lon, year, month, size_px):
    epsg, x0, y0, geom = area_grid(lat, lon, size_px)
    img = gee.monthly_composite(geom, year, month)
    s1 = gee.fetch(img, epsg, x0, y0, width=size_px)
    n = gee.scene_count(geom, year, month)
    return s1, n


def fetch_dynamic_world(lat, lon, year, month, size_px, n_months=12):
    """Independent Sentinel-2 based reference map (mode over the 12-month window)."""
    epsg, x0, y0, geom = area_grid(lat, lon, size_px)
    start = f"{year:04d}-{month:02d}-01"
    ly, lm = month_list(year, month, n_months + 1)[-1]
    end = f"{ly:04d}-{lm:02d}-01"
    return gee.fetch(gee.dynamic_world_label(geom, start, end), epsg, x0, y0, width=size_px)[0]


@torch.no_grad()
def predict(model, s1_db, mean, std, device):
    x = torch.from_numpy(normalize(s1_db, mean, std))[None].to(device)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    missing = s1_db[0] <= C.S1_NODATA + 1
    pred[missing] = C.IGNORE_INDEX
    return pred


def temporal_mode_filter(maps: np.ndarray, window: int = 3) -> np.ndarray:
    """Per-pixel majority vote over a sliding window of months.
    Suppresses single-month flips caused by speckle / seasonal noise,
    which would otherwise inflate change percentages."""
    T = maps.shape[0]
    out = maps.copy()
    h = window // 2
    for t in range(T):
        sl = maps[max(0, t - h): min(T, t + h + 1)]
        counts = np.stack([(sl == c).sum(0) for c in range(C.NUM_CLASSES)])
        out[t] = counts.argmax(0)
        out[t][(sl == C.IGNORE_INDEX).all(0)] = C.IGNORE_INDEX
    return out


def class_percentages(maps: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """% of valid pixels per class and month. A month without any Sentinel-1 data is NaN (no data),
    not 0 %, so it never shows up as a fake change."""
    rows = []
    for lab, m in zip(labels, maps):
        valid = m != C.IGNORE_INDEX
        n = valid.sum()
        rows.append({"month": lab, **{c: (100 * (m[valid] == i).sum() / n if n else np.nan)
                                      for i, c in enumerate(C.CLASSES)}})
    return pd.DataFrame(rows).set_index("month")


def changes(pct: pd.DataFrame):
    """Month-to-month change and total change (first -> last), in percentage points."""
    mom = pct.diff().iloc[1:]          # NaN where either month has no data
    mom.index = [f"{a} → {b}" for a, b in zip(pct.index[:-1], pct.index[1:])]
    ok = pct.dropna()                  # total: first -> last month WITH data
    if len(ok) < 2:
        return mom, pd.Series(np.nan, index=pct.columns, name="no data")
    total = (ok.iloc[-1] - ok.iloc[0]).rename(f"{ok.index[0]} → {ok.index[-1]}")
    return mom, total


def changed_pixel_share(maps: np.ndarray) -> float:
    """% of valid pixels whose class differs between first and last month."""
    a, b = maps[0], maps[-1]
    v = (a != C.IGNORE_INDEX) & (b != C.IGNORE_INDEX)
    return 100 * (a[v] != b[v]).mean() if v.any() else float("nan")


def agreement(pred: np.ndarray, ref: np.ndarray) -> float:
    v = (pred != C.IGNORE_INDEX) & (ref != C.IGNORE_INDEX)
    return 100 * (pred[v] == ref[v]).mean() if v.any() else float("nan")


def area_corners(lat: float, lon: float, size_px: int, scale: int = C.SCALE):
    """Lon/lat of the 4 corners of the mapped square (bottom-left, top-left, top-right, bottom-right),
    computed on the same UTM grid as the downloads, for placing the map on a web basemap."""
    from pyproj import Transformer
    epsg = gee.utm_epsg(lat, lon)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    h = size_px * scale / 2
    return [list(inv.transform(cx - h, cy - h)), list(inv.transform(cx - h, cy + h)),
            list(inv.transform(cx + h, cy + h)), list(inv.transform(cx + h, cy - h))]


def map_png_dataurl(class_map: np.ndarray, alpha: int = 255) -> str:
    """Class map -> transparent RGBA PNG as a data URL (no-data pixels fully transparent)."""
    import base64, io
    from PIL import Image
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in C.PALETTE], np.uint8)
    rgba = np.zeros(class_map.shape + (4,), np.uint8)
    valid = class_map != C.IGNORE_INDEX
    rgba[valid, :3] = pal[class_map[valid]]
    rgba[valid, 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
