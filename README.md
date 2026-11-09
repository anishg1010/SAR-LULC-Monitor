# SAR-only Land Use / Land Cover Segmentation (Sentinel-1)

Deep Learning project · M.Sc. Applied Data Science and Analytics · SRH University Heidelberg · Anish Gaware

A deep learning model that maps **land use / land cover (LULC)** from **Sentinel-1 radar only**, and a web
prototype that turns a **latitude/longitude + start month** into **12 monthly LULC maps**, the **percentage of
area per class**, and the **month-to-month and total change in percentage points**.

Classes: Water · Built-up · Tree cover · Cropland · Grass/Shrub/Wetland · Bare/Sparse

## Links

| Item | Link |
|---|---|
| Dataset (`s1_lulc_dataset.zip`, 862 MB) | [Google Drive folder](https://drive.google.com/drive/folders/1OHT8kTIMf6AFuNg6crv4tyUOWJI01Nfg?usp=sharing) |
| Model weights (`temporal_lstm.pt`, `best_model.pt`) | [Google Drive folder](https://drive.google.com/drive/folders/1OHT8kTIMf6AFuNg6crv4tyUOWJI01Nfg?usp=sharing) (put the `.pt` files into `models/`) |
| Training + prototype code | this repository |

## Results (test set: Prague, Porto, Jutland — regions never seen in training)

| Model | Overall accuracy | mIoU | mean F1 |
|---|---|---|---|
| A · U-Net / ResNet-34, ImageNet pretrained (single month) | 72.1 % | 0.509 | 0.625 |
| B · DeepLabV3+ / ResNet-34, ImageNet pretrained | 71.5 % | 0.499 | 0.613 |
| C · U-Net / ResNet-34 from scratch | 68.2 % | 0.480 | 0.600 |
| **D · Temporal CNN + LSTM (4 seasons)** | **75.3 %** | **0.541** | **0.657** |

Variant D feeds four seasonal composites through the U-Net and reads the per-pixel features with an LSTM; it
mainly improves cropland (+0.080 IoU) and grassland (+0.066 IoU), which look alike in any single month.
Case studies (2023, 5.12 × 5.12 km), checked against Google Dynamic World (independent, Sentinel-2 based):

| Area | Model | Agreement with Dynamic World | Mean month-to-month change (raw) |
|---|---|---|---|
| Mannheim | single-month U-Net | 90.7 % | 0.63 pp |
| Mannheim | temporal CNN + LSTM | 88.0 % | 0.33 pp |
| Pune, India (outside training range) | single-month U-Net | 89.7 % | 1.07 pp |
| Pune, India (outside training range) | temporal CNN + LSTM | 86.6 % | 0.45 pp |

In a city centre almost no real change is expected within a year, so the month-to-month change mostly measures
noise: the temporal model roughly halves it. In Pune the city and river are mapped well, but urban vegetation is
missed (tree recall about 0.1), and only 0–3 Sentinel-1 scenes per month are available.

## Repository layout
```
app.py                     Streamlit prototype (the website)
case_study.py              runs the prototype pipeline headless for one location, saves maps + CSVs
start_website.bat          Windows one-click start of the website
s1lulc/config.py           classes, WorldCover -> 6-class mapping, 21 regions + region-based split, settings
s1lulc/gee.py              Earth Engine: Sentinel-1 monthly composites (speckle filter, median, dB), labels, download
s1lulc/data.py             PyTorch dataset, normalisation, augmentation
s1lulc/model.py            model factory (segmentation_models_pytorch), CE + Dice loss, OA / IoU / F1
s1lulc/temporal.py         variant D: temporal CNN + LSTM and its sequence dataset
s1lulc/predict.py          prototype back end: inference, smoothing, percentages, changes, agreement
notebooks/01_data_collection.ipynb   builds the dataset (executed, with outputs)
notebooks/02_training.ipynb          trains + evaluates variants A-C (executed, with training curves)
notebooks/03_temporal_lstm.ipynb     trains + evaluates variant D (executed, with training curves)
models/                    training curves, confusion matrices, examples, test results (*.png, *.csv)
data/                      class distribution, class weights, normalisation statistics, scene counts
case_study/<name>/         maps, percentages, changes, Dynamic World comparison for one area
                           (mannheim, mannheim_temporal, pune, pune_temporal)
case_study/website_exports/  CSV exports from website runs
```
Model weights (`models/*.pt`) and the patch files (`data/patches/`) are not in the repository; see **Links**.

## Required accounts / API keys
- **Google Earth Engine** account with a Cloud project registered for noncommercial use
  (https://code.earthengine.google.com/register). Authentication is OAuth (`earthengine authenticate`); no key file.
- No other accounts or keys.

## Setup
Tested with Python 3.11 (training, Linux, NVIDIA H200) and Python 3.14 (prototype, Windows 11, CPU).

**Windows**
```bat
cd path\to\project
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
earthengine authenticate
```

**Linux / macOS**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # for GPU training install the CUDA build of torch first
earthengine authenticate               # on a server without a browser: --auth_mode=notebook
export GEE_PROJECT=your-ee-project-id
```

Download the model weights (see **Links**) into `models/`.

## Run the prototype
Windows: set your Earth Engine project ID in `start_website.bat` (`GEE_PROJECT=...`) and double-click it.
Other systems: `streamlit run app.py` with `GEE_PROJECT` set. Then open http://localhost:8501.

1. Pick a location (preset or latitude/longitude), start year and month, area size.
2. Choose the model:
   - **Temporal CNN + LSTM** (default): month *t* is classified from the composites of *t−9, t−6, t−3, t*
     (one per season), so 21 months are retrieved (9 of history + 12).
   - **Single-month U-Net**: each month on its own.
3. Optional: 3-month temporal smoothing; comparison with Dynamic World.
4. Click **Run analysis**. The page shows the 12 maps as a grid and in an interactive **map view** (one month
   at a time over OpenStreetMap, with a month slider and opacity control), the class percentages, the
   month-to-month and total change, and the Dynamic World check.
5. **Download results (CSV)**: the first line records location, area, model and smoothing; then one row per
   month (`YYYY-MM`) with the number of Sentinel-1 scenes, class percentages and month-to-month changes.

A month without any Sentinel-1 scene is shown as "no data" (not 0 %), and the temporal model replaces a missing
history month with the nearest month that has data. Outside Europe (e.g. the Pune, Mumbai, Cairo, New York
presets) results are experimental: the model was trained on European regions only.

Headless alternative: `python case_study.py 49.4875 8.4660 2023 1 mannheim` writes maps and tables to `case_study/<name>/`.

## Reproduce training
1. `notebooks/01_data_collection.ipynb` downloads the dataset from Earth Engine into `data/`
   (or unzip `s1_lulc_dataset.zip` into `data/` and skip this step).
2. `notebooks/02_training.ipynb` trains variants A-C and writes `models/best_model.pt`.
3. `notebooks/03_temporal_lstm.ipynb` trains variant D from `best_model.pt` and writes `models/temporal_lstm.pt`.

## How the numbers are calculated
- **Class percentage** (month *t*, class *c*) = pixels predicted as *c* / valid pixels × 100 (pixels without Sentinel-1 data excluded).
- **Month-to-month change** = percentage(*t*) − percentage(*t*−1), in percentage points (pp).
- **Total change** = percentage(last month) − percentage(first month), in pp, using the first and last months that have data.
- **Pixels changed** = share of pixels whose class differs between the first and last month.
- **Temporal smoothing** (optional): per-pixel majority vote over a 3-month window.
- **Sanity check**: 12-month majority map vs **Dynamic World** (independent of the WorldCover training labels).

## Data sources
- Sentinel-1 GRD (`COPERNICUS/S1_GRD`), Copernicus / ESA, via Google Earth Engine (model input)
- ESA WorldCover 10 m 2021 v200 (`ESA/WorldCover/v200`) (training labels)
- Google Dynamic World V1 (`GOOGLE/DYNAMICWORLD/V1`) (independent reference, case study only)

## AI assistance
Parts of the code and documentation were drafted with the help of an AI assistant (Claude, Anthropic).
Data collection, training, evaluation and prototype runs were carried out and checked by the author.
