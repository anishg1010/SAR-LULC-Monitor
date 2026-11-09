"""SAR-only Land Use / Land Cover monitor — Streamlit prototype.

Run:  streamlit run app.py
Input: latitude / longitude + start month  ->  12 monthly LULC maps from Sentinel-1,
class percentages, month-to-month and total change (percentage points).
"""
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import streamlit as st

from s1lulc import config as C, gee
from s1lulc import predict as P

ROOT = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "best_model.pt"))
TEMPORAL_PATH = Path(os.environ.get("TEMPORAL_PATH", ROOT / "models" / "temporal_lstm.pt"))
CMAP = ListedColormap(C.PALETTE)
LEGEND = [Patch(color=c, label=n) for c, n in zip(C.PALETTE, C.CLASSES)]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="SAR LULC Monitor", page_icon="🛰️", layout="wide")


# ---------------------------------------------------------------- resources
@st.cache_resource
def init_ee():
    gee.init(os.environ.get("GEE_PROJECT", "project-5ed4e2e0-4d5f-4a56-957"))
    return True


@st.cache_resource
def get_model():
    return P.load_model(MODEL_PATH)


@st.cache_resource
def get_temporal():
    from s1lulc.temporal import load_temporal
    return load_temporal(TEMPORAL_PATH)


@st.cache_data(show_spinner=False)
def get_month(lat, lon, y, m, size):
    return P.fetch_month(lat, lon, y, m, size)


@st.cache_data(show_spinner=False)
def get_dw(lat, lon, y, m, size):
    return P.fetch_dynamic_world(lat, lon, y, m, size)


def show_map(ax, arr, title):
    a = arr.astype(float); a[arr == C.IGNORE_INDEX] = np.nan
    ax.imshow(a, cmap=CMAP, vmin=0, vmax=C.NUM_CLASSES - 1, interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.axis("off")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛰️ SAR LULC Monitor")
st.sidebar.caption("Land use / land cover from Sentinel-1 radar only")
# Europe = the region the model was trained for. "Outside Europe" = experimental (out of distribution).
# Prague and Porto are TEST regions (never seen in training); training regions are not offered as presets.
presets = {
    "Mannheim, DE (case study)": (49.4875, 8.4660),
    "Heidelberg, DE": (49.4093, 8.6937),
    "Berlin, DE": (52.5200, 13.4050),
    "Hamburg, DE": (53.5511, 9.9937),
    "Paris, FR": (48.8566, 2.3522),
    "Vienna, AT": (48.2082, 16.3738),
    "Prague, CZ (test region)": (50.0800, 14.4300),
    "Porto, PT (test region)": (41.1500, -8.6100),
    "Pune, IN (experimental)": (18.5204, 73.8567),
    "Mumbai, IN (experimental)": (19.0760, 72.8777),
    "Cairo, EG (experimental)": (30.0444, 31.2357),
    "New York, US (experimental)": (40.7128, -74.0060),
    "Custom": None,
}
preset = st.sidebar.selectbox("Location", list(presets))
d_lat, d_lon = presets[preset] or (49.4875, 8.4660)
if "experimental" in preset:
    st.sidebar.warning("Outside the training region (Europe): results are less reliable, especially for vegetation "
                       "classes and in monsoon/dry-season climates. Check the Dynamic World agreement.")
lat = st.sidebar.number_input("Latitude", -60.0, 75.0, d_lat, format="%.4f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, d_lon, format="%.4f")
c1, c2 = st.sidebar.columns(2)
year = c1.number_input("Start year", 2016, date.today().year - 1, 2023)
month = c2.selectbox("Start month", range(1, 13), format_func=lambda i: MONTHS[i - 1])
size_km = st.sidebar.select_slider("Area size", [2.56, 5.12, 7.68], value=5.12, format_func=lambda v: f"{v} km")
MODEL_CHOICES = ["Temporal CNN + LSTM (recommended)", "Single-month U-Net"]
model_choice = st.sidebar.radio("Model", MODEL_CHOICES if TEMPORAL_PATH.exists() else MODEL_CHOICES[1:],
                                help="Temporal: each month is classified together with the 3 seasons before it "
                                     "(t-9, t-6, t-3, t months), which separates crops from grassland. "
                                     "Single-month: each month on its own.")
temporal = model_choice.startswith("Temporal")
smooth = st.sidebar.checkbox("Temporal smoothing (3-month majority)", value=False,
                             help="Removes single-month class flips caused by speckle or seasonal effects.")
compare_dw = st.sidebar.checkbox("Compare with Dynamic World (independent)", value=True)
run = st.sidebar.button("Run analysis", type="primary", width="stretch")

st.title("Monthly land use / land cover from Sentinel-1")
st.caption("Model input: Sentinel-1 GRD (VV, VH, VV−VH), monthly median composite. "
           "Models: U-Net (single month) or U-Net + LSTM over 4 seasons, trained on ESA WorldCover 2021 labels.")

params = (round(float(lat), 4), round(float(lon), 4), int(year), int(month), float(size_km), temporal)
if run:
    st.session_state["params"] = params
if st.session_state.get("params") != params:
    st.info("Choose a location and start month in the sidebar, then click **Run analysis**.")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)
    st.stop()

# ---------------------------------------------------------------- pipeline
if not MODEL_PATH.exists():
    st.error(f"Model weights not found: {MODEL_PATH}"); st.stop()
init_ee()
size = int(size_km * 100)
months = P.month_list(int(year), int(month))
labels = [f"{MONTHS[m-1]} {y}" for y, m in months]


@st.cache_data(show_spinner=False, max_entries=8)
def run_pipeline(lat, lon, year, month, size, temporal):
    """Retrieve composites and classify 12 months. Cached, so moving a slider does not recompute."""
    months = P.month_list(year, month)
    maps, scenes = [], []
    bar = st.progress(0.0, "Retrieving Sentinel-1 composites …")
    if not temporal:
        model, mean, std, device = get_model()
        for i, (y, m) in enumerate(months):
            s1, n = get_month(lat, lon, y, m, size)
            scenes.append(n)
            maps.append(P.predict(model, s1, mean, std, device) if n > 0 else np.full((size, size), C.IGNORE_INDEX, np.uint8))
            bar.progress((i + 1) / 12, f"{MONTHS[m-1]} {y}: {n} scenes, classified")
    else:
        from s1lulc.temporal import predict_sequence, build_sequence
        model, mean, std, device = get_temporal()
        # 9 months of history before the start month, so every mapped month t has the sequence t-9, t-6, t-3, t
        first = (year * 12 + month - 1) - 9
        allm = [(k // 12, k % 12 + 1) for k in range(first, first + 21)]
        comps, counts = [], []
        for i, (y, m) in enumerate(allm):
            s1, n = get_month(lat, lon, y, m, size)
            comps.append(s1); counts.append(n)
            bar.progress((i + 1) / 21, f"Retrieving 21 months (9 of history + 12): {MONTHS[m-1]} {y}, {n} scenes")
        for t in range(12):
            n = counts[t + 9]
            scenes.append(n)
            maps.append(predict_sequence(model, build_sequence(comps, counts, t), mean, std, device)
                        if n > 0 else np.full((size, size), C.IGNORE_INDEX, np.uint8))
    bar.empty()
    return np.stack(maps), scenes


maps, scenes = run_pipeline(*params[:4], size, temporal)
if smooth:
    maps = P.temporal_mode_filter(maps)

pct = P.class_percentages(maps, labels)
mom, total = P.changes(pct)

# ---------------------------------------------------------------- output
k1, k2, k3, k4 = st.columns(4)
k1.metric("Area", f"{size_km} × {size_km} km")
k2.metric("S1 scenes used", f"{sum(scenes)}", f"min {min(scenes)}/month", delta_color="off")
k3.metric("Pixels changed (first → last)", f"{P.changed_pixel_share(maps):.1f} %")
dom = pct.mean().idxmax()
k4.metric("Dominant class", dom, f"{pct[dom].mean():.1f} % avg", delta_color="off")

if min(scenes) == 0:
    st.warning("Some months have no Sentinel-1 acquisitions; they are shown blank, marked 'no data' in the tables "
               "and skipped in the change calculations.")
elif min(scenes) <= 2:
    st.warning(f"Only {min(scenes)} Sentinel-1 scene(s) in some months: those composites are noisy (speckle is not "
               "averaged out), so month-to-month changes are less reliable.")

st.subheader("LULC maps — 12 months")
fig, axes = plt.subplots(2, 6, figsize=(18, 6.6))
for ax, mp, lab in zip(axes.ravel(), maps, labels):
    show_map(ax, mp, lab)
fig.legend(handles=LEGEND, loc="lower center", ncol=6, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1))
st.pyplot(fig)

st.subheader("Map view")
import json
import streamlit.components.v1 as components
mv1, mv2 = st.columns([3, 1])
sel = mv1.select_slider("Month shown on the map", options=labels, value=labels[0])
opacity = mv2.slider("Overlay opacity", 0.0, 1.0, 0.65, 0.05)
corners = P.area_corners(lat, lon, size)          # [lon, lat] x 4, exact UTM square
lons, lats = [c[0] for c in corners], [c[1] for c in corners]
bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
outline = [[c[1], c[0]] for c in corners]
leaflet = f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="map" style="height:510px;border-radius:6px"></div>
<script>
var map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
var b = {json.dumps(bounds)};
L.imageOverlay({json.dumps(P.map_png_dataurl(maps[labels.index(sel)]))}, b, {{opacity: {opacity}}}).addTo(map);
L.polygon({json.dumps(outline)}, {{color: '#222', weight: 2, fill: false}}).addTo(map);
map.fitBounds(b, {{padding: [20, 20]}});
</script>"""
components.html(leaflet, height=520)
st.caption("LULC map of the selected month over an OpenStreetMap basemap. Drag the slider to step "
           "through the 12 months; scroll to zoom. Colours: " +
           ", ".join(f"{n}" for n in C.CLASSES) + " as in the legend above.")

st.subheader("Class coverage over time")
c_left, c_right = st.columns([3, 2])
fig2, ax2 = plt.subplots(figsize=(9, 4))
for c, col in zip(C.CLASSES, C.PALETTE):
    ax2.plot(pct.index, pct[c], marker="o", label=c, color=col, lw=2)
ax2.set_ylabel("% of area"); ax2.grid(alpha=.3); ax2.legend(fontsize=8, ncol=2)
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
c_left.pyplot(fig2)
c_right.markdown("**Total change, first → last month** (percentage points)")
c_right.caption(f"Months compared: {total.name}")
c_right.dataframe(total.to_frame("Δ pp").style.format("{:+.2f}", na_rep="no data")
                  .background_gradient(cmap="RdBu", vmin=-10, vmax=10), width="stretch")

st.markdown("**Class percentages per month (%)**")
st.dataframe(pct.style.format("{:.2f}", na_rep="no data"), width="stretch")
st.markdown("**Month-to-month change (percentage points)**")
st.dataframe(mom.style.format("{:+.2f}", na_rep="no data").background_gradient(cmap="RdBu", vmin=-5, vmax=5), width="stretch")
st.caption("Percentage of area = class pixels / valid pixels × 100. Change = difference of percentages "
           "(percentage points), not relative %. Pixel-level misclassifications that flip between months "
           "inflate apparent change; use temporal smoothing to see the effect.")

if compare_dw:
    st.subheader("Sanity check against Dynamic World (Sentinel-2, not used for training)")
    dw = get_dw(lat, lon, int(year), int(month), size)
    mode_pred = P.temporal_mode_filter(maps, window=25)[0]  # 12-month majority map
    dw_pct = P.class_percentages(dw[None], ["Dynamic World"]).iloc[0]
    ours = P.class_percentages(mode_pred[None], ["Our model (12-mo mode)"]).iloc[0]
    a, b = st.columns([2, 3])
    fig3, ax3 = plt.subplots(1, 2, figsize=(9, 4.6))
    show_map(ax3[0], mode_pred, "Our model (12-month majority)")
    show_map(ax3[1], dw, "Dynamic World (12-month mode)")
    plt.tight_layout(); a.pyplot(fig3)
    comp = pd.DataFrame({"Our model %": ours, "Dynamic World %": dw_pct})
    comp["Difference pp"] = comp.iloc[:, 0] - comp.iloc[:, 1]
    b.metric("Pixel agreement with Dynamic World", f"{P.agreement(mode_pred, dw):.1f} %")
    b.dataframe(comp.style.format("{:.2f}"), width="stretch")

csv = pct.join(pct.diff().add_suffix(" Δpp (vs previous month)")).round(3)
csv.index = [f"{y}-{m:02d}" for y, m in P.month_list(int(year), int(month))]   # ISO months: Excel keeps them as text
csv.index.name = "month"
csv.insert(0, "S1 scenes", scenes)
meta = (f"# lat={lat:.4f}, lon={lon:.4f}, area={size / 100:.2f} km, model={model_choice}, "
        f"smoothing={'3-month majority' if smooth else 'off'}\n")
fname = f"lulc_{lat:.4f}_{lon:.4f}_{labels[0].replace(' ', '')}_{'temporal' if temporal else 'single'}{'_smoothed' if smooth else ''}.csv"
st.download_button("Download results (CSV)", (meta + csv.to_csv()).encode(), fname, "text/csv")
