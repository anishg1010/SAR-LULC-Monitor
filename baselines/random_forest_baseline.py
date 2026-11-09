"""Classical baseline: Random Forest on per-pixel SAR features (report section 4.4).

Step 1 samples pixels from s1_lulc_dataset.zip (no Earth Engine needed):
  features per month = VV, VH, VV-VH (dB) + their 7x7 local mean and std  -> 9 features
  single-month version: one month per sample; four-season version: Jan/Apr/Jul/Oct stacked -> 36 features
  train: 400 random pixels per training patch and month; test: every 8th pixel of all test patches.
Step 2 trains scikit-learn Random Forests and reports OA, mIoU, mean F1 and per-class IoU on the test regions.

Usage: python baselines/random_forest_baseline.py path/to/s1_lulc_dataset.zip
"""
import io, json, sys, time, zipfile
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

CLASSES = ["Water", "Built-up", "Tree cover", "Cropland", "Grass/Shrub/Wetland", "Bare/Sparse"]
Z = zipfile.ZipFile(sys.argv[1] if len(sys.argv) > 1 else "s1_lulc_dataset.zip")
meta = pd.read_csv(Z.open("metadata.csv")); meta = meta[meta.keep]
load = lambda n: np.load(io.BytesIO(Z.read("patches/" + n)))


def box(a, r):  # mean over a (2r+1)^2 window, edge-padded, per channel
    p = np.pad(a, ((0, 0), (r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(1).cumsum(2); k = 2 * r + 1
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / k ** 2


def feats(s):  # s: (3, H, W) in dB
    m = box(s, 3); sd = np.sqrt(np.maximum(box(s ** 2, 3) - m ** 2, 0))
    return np.concatenate([s, m, sd]).astype(np.float32)


rng = np.random.default_rng(0)
data = {}
for split, n_px, stride in [("train", 100, None), ("test", None, 8)]:
    X1, y1, X4, y4 = [], [], [], []
    for _, g in meta[meta.split == split].groupby("patch"):
        lab = load(g.label_path.iloc[0]).astype(np.int64)
        F = {r.month: feats(load(r.s1_path).astype(np.float32) / 100.0) for _, r in g.iterrows()}
        if n_px:
            idx = rng.integers(0, 256 * 256, n_px * 4); ii, jj = idx // 256, idx % 256
        else:
            ii, jj = [a.ravel() for a in np.meshgrid(np.arange(0, 256, stride), np.arange(0, 256, stride), indexing="ij")]
        yy = lab[ii, jj]; ok = yy != 255
        for f in F.values():
            X1.append(f[:, ii, jj].T[ok]); y1.append(yy[ok])
        if len(F) == 4:
            X4.append(np.concatenate([F[m][:, ii, jj].T for m in sorted(F)], 1)[ok]); y4.append(yy[ok])
    data[split] = {"1": (np.concatenate(X1), np.concatenate(y1)), "4": (np.concatenate(X4), np.concatenate(y4))}

results = {}
for k, name in [("1", "Random Forest, single month (9 features)"), ("4", "Random Forest, 4 seasons (36 features)")]:
    t = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=22, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=0).fit(*data["train"][k])
    X, y = data["test"][k]; p = rf.predict(X)
    cm = np.zeros((6, 6), int); np.add.at(cm, (y, p), 1)
    tp = np.diag(cm); iou = tp / (cm.sum(0) + cm.sum(1) - tp); f1 = 2 * tp / (cm.sum(0) + cm.sum(1))
    results[name] = {"OA": tp.sum() / cm.sum(), "mIoU": iou.mean(), "mF1": f1.mean(),
                     **{f"IoU {c}": v for c, v in zip(CLASSES, iou)}, "test_px": len(y), "fit_s": time.time() - t}
    print(name, {a: round(float(b), 3) for a, b in results[name].items()})
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in results.items()},
          open("baselines/random_forest_results.json", "w"), indent=1)
