"""Case study: run the prototype pipeline for one location and save all outputs.
Usage: python case_study.py [lat lon year month name]
Default: Mannheim, Jan 2023 - Dec 2023.
Set MODEL=temporal to use the temporal CNN + LSTM (models/temporal_lstm.pt); default: single-month U-Net."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from s1lulc import config as C, gee, predict as P

lat, lon, year, month, name = (49.4875, 8.4660, 2023, 1, "mannheim")
if len(sys.argv) == 6:
    lat, lon, year, month, name = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
SIZE = 512
OUT = Path("case_study") / name; OUT.mkdir(parents=True, exist_ok=True)
CMAP = ListedColormap(C.PALETTE)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

gee.init(os.environ.get("GEE_PROJECT", "project-5ed4e2e0-4d5f-4a56-957"))
TEMPORAL = os.environ.get("MODEL", "single") == "temporal"
if TEMPORAL:
    from s1lulc.temporal import load_temporal, predict_sequence, build_sequence
    model, mean, std, dev = load_temporal("models/temporal_lstm.pt")
    name = name + "_temporal"; OUT = Path("case_study") / name; OUT.mkdir(parents=True, exist_ok=True)
else:
    model, mean, std, dev = P.load_model("models/best_model.pt")
months = P.month_list(year, month)
labels = [f"{MON[m-1]} {y}" for y, m in months]

maps, scenes = [], []
if TEMPORAL:   # each month t is classified from t-9, t-6, t-3, t
    first = (year * 12 + month - 1) - 9
    comps, counts = [], []
    for k in range(first, first + 21):
        s1, n = P.fetch_month(lat, lon, k // 12, k % 12 + 1, SIZE); comps.append(s1); counts.append(n)
        print(f"{MON[k % 12]} {k // 12}", "scenes", n, flush=True)
    for t in range(12):
        scenes.append(counts[t + 9])
        maps.append(predict_sequence(model, build_sequence(comps, counts, t), mean, std, dev)
                    if counts[t + 9] else np.full((SIZE, SIZE), 255, np.uint8))
else:
    for (y, m), lab in zip(months, labels):
        s1, n = P.fetch_month(lat, lon, y, m, SIZE)
        scenes.append(n)
        maps.append(P.predict(model, s1, mean, std, dev) if n else np.full((SIZE, SIZE), 255, np.uint8))
        print(lab, "scenes", n, flush=True)
maps = np.stack(maps); np.save(OUT / "maps.npy", maps)
smooth = P.temporal_mode_filter(maps)

def show(ax, a, t):
    a = a.astype(float); a[a == 255] = np.nan
    ax.imshow(a, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest"); ax.set_title(t, fontsize=10); ax.axis("off")

leg = [Patch(color=c, label=n) for c, n in zip(C.PALETTE, C.CLASSES)]
for arr, tag in [(maps, "raw"), (smooth, "smoothed")]:
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.6))
    for a, mp, lab in zip(ax.ravel(), arr, labels): show(a, mp, lab)
    fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False)
    plt.suptitle(f"{name.title()} ({lat:.4f}, {lon:.4f}) — monthly LULC from Sentinel-1 ({tag})")
    plt.tight_layout(rect=(0, .06, 1, .96)); plt.savefig(OUT / f"maps_{tag}.png", dpi=130); plt.close()

summary = {"model": "temporal CNN+LSTM" if TEMPORAL else "single-month U-Net", "location": name, "lat": lat, "lon": lon, "start": labels[0], "end": labels[-1],
           "area_km": SIZE / 100, "scenes_per_month": dict(zip(labels, scenes))}
for arr, tag in [(maps, "raw"), (smooth, "smoothed")]:
    pct = P.class_percentages(arr, labels); mom, total = P.changes(pct)
    pct.round(2).to_csv(OUT / f"percentages_{tag}.csv"); mom.round(2).to_csv(OUT / f"month_to_month_{tag}.csv")
    total.round(2).to_csv(OUT / f"total_change_{tag}.csv")
    summary[f"changed_pixels_first_last_{tag}_%"] = round(P.changed_pixel_share(arr), 2)
    summary[f"mean_abs_mom_change_{tag}_pp"] = round(float(mom.abs().mean().mean()), 3)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c, col in zip(C.CLASSES, C.PALETTE): ax.plot(labels, pct[c], marker="o", color=col, lw=2, label=c)
    ax.set_ylabel("% of area"); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=3); plt.xticks(rotation=45, ha="right")
    ax.set_title(f"Class coverage over time ({tag})"); plt.tight_layout(); plt.savefig(OUT / f"coverage_{tag}.png", dpi=130); plt.close()

# Independent check: Dynamic World (Sentinel-2) over the same 12 months
dw = P.fetch_dynamic_world(lat, lon, year, month, SIZE); np.save(OUT / "dynamic_world.npy", dw)
annual = P.temporal_mode_filter(maps, window=25)[0]
ours_pct = P.class_percentages(annual[None], ["ours"]).iloc[0]; dw_pct = P.class_percentages(dw[None], ["dw"]).iloc[0]
comp = pd.DataFrame({"our_model_%": ours_pct, "dynamic_world_%": dw_pct}); comp["difference_pp"] = comp.iloc[:, 0] - comp.iloc[:, 1]
comp.round(2).to_csv(OUT / "dynamic_world_comparison.csv")
summary["agreement_with_dynamic_world_%"] = round(P.agreement(annual, dw), 2)
v = (annual != 255) & (dw != 255)
cm = np.zeros((6, 6), int); np.add.at(cm, (dw[v], annual[v]), 1)
summary["per_class_recall_vs_DW"] = {c: round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i, c in enumerate(C.CLASSES)}
fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
show(ax[0], annual, "Our model (12-month majority)"); show(ax[1], dw, "Dynamic World (12-month mode)")
fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False); plt.tight_layout(rect=(0, .07, 1, 1))
plt.savefig(OUT / "dynamic_world_comparison.png", dpi=130); plt.close()
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2)); print(comp.round(2))
