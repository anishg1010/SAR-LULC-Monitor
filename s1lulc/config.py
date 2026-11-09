"""Shared configuration: classes, label mapping, regions and data settings."""

# ---------------------------------------------------------------------------
# Classes
# ESA WorldCover has 11 classes. Several are hard or impossible to tell apart
# with C-band SAR alone, so they are merged into 6 SAR-separable classes.
# ---------------------------------------------------------------------------
CLASSES = [
    "Water",                 # 0
    "Built-up",              # 1
    "Tree cover",            # 2
    "Cropland",              # 3
    "Grass/Shrub/Wetland",   # 4
    "Bare/Sparse",           # 5
]
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

PALETTE = [
    "#1f78b4",  # Water
    "#e31a1c",  # Built-up
    "#33a02c",  # Tree cover
    "#ffd92f",  # Cropland
    "#b2df8a",  # Grass/Shrub/Wetland
    "#bdbdbd",  # Bare/Sparse
]

# ESA WorldCover v200 (2021) code -> our class index
WORLDCOVER_REMAP = {
    10: 2,    # Tree cover
    20: 4,    # Shrubland
    30: 4,    # Grassland
    40: 3,    # Cropland
    50: 1,    # Built-up
    60: 5,    # Bare / sparse vegetation
    70: 255,  # Snow and ice -> ignored (rare, SAR-ambiguous)
    80: 0,    # Permanent water bodies
    90: 4,    # Herbaceous wetland
    95: 2,    # Mangroves
    100: 4,   # Moss and lichen
}

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
LABEL_YEAR = 2021          # WorldCover v200 reference year
TRAIN_MONTHS = [1, 4, 7, 10]  # one month per season -> model sees seasonal variation
PATCH_SIZE = 256           # pixels
SCALE = 10                 # metres per pixel
PATCHES_PER_SIDE = 6       # region = 6 x 256 px x 10 m = 15.36 km square
S1_BANDS = ["VV", "VH", "VV_VH"]   # VV dB, VH dB, VV-VH (dB ratio)
S1_NODATA = -50.0          # value written where SAR data is missing
INT16_SCALE = 100          # dB values stored as int16 (dB * 100)

# ---------------------------------------------------------------------------
# Regions (centre lat, lon). Split is by REGION, not by patch, so the test set
# is geographically independent from training (no spatial leakage).
# Mannheim/Heidelberg is deliberately NOT included: it is reserved for the
# prototype case study.
# ---------------------------------------------------------------------------
REGIONS = {
    # ---- train ----
    "munich":            (48.14, 11.58, "train"),
    "lyon":              (45.76, 4.84, "train"),
    "milan":             (45.46, 9.19, "train"),
    "warsaw":            (52.23, 21.01, "train"),
    "seville":           (37.39, -5.98, "train"),
    "amsterdam":         (52.37, 4.90, "train"),
    "bucharest":         (44.43, 26.10, "train"),
    "black_forest":      (48.00, 8.20, "train"),
    "beauce_cropland":   (48.30, 1.70, "train"),
    "hungarian_plain":   (47.00, 20.20, "train"),
    "scottish_highlands": (57.10, -4.70, "train"),
    "lake_balaton":      (46.85, 17.70, "train"),
    "almeria":           (37.00, -2.30, "train"),
    "tampere_lakes":     (61.50, 23.80, "train"),
    "camargue_wetland":  (43.55, 4.55, "train"),
    # ---- validation ----
    "leipzig":           (51.34, 12.37, "val"),
    "valencia":          (39.47, -0.38, "val"),
    "bavarian_forest":   (49.00, 13.40, "val"),
    # ---- test ----
    "prague":            (50.08, 14.43, "test"),
    "porto":             (41.15, -8.61, "test"),
    "jutland_cropland":  (55.60, 9.50, "test"),
}
