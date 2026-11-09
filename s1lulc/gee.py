"""Google Earth Engine helpers: Sentinel-1 monthly composites, WorldCover labels,
and pixel download on a fixed UTM grid.

Used by BOTH the dataset notebook and the prototype, so training data and
prototype inputs go through exactly the same preprocessing.
"""
from __future__ import annotations

import ee
import numpy as np
from pyproj import Transformer

from . import config as C


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def init(project: str) -> None:
    """Authenticate (first time only) and initialise Earth Engine."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project)


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def region_grid(lat: float, lon: float, patches_per_side: int = C.PATCHES_PER_SIDE,
                patch: int = C.PATCH_SIZE, scale: int = C.SCALE):
    """Return (epsg, x0, y0, geometry) for a square region centred on lat/lon.
    x0/y0 is the upper-left corner in UTM metres."""
    epsg = utm_epsg(lat, lon)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = tf.transform(lon, lat)
    half = patches_per_side * patch * scale / 2
    x0, y0 = cx - half, cy + half
    geom = ee.Geometry.Rectangle([x0, y0 - 2 * half, x0 + 2 * half, y0],
                                 proj=f"EPSG:{epsg}", geodesic=False)
    return epsg, x0, y0, geom


# ---------------------------------------------------------------------------
# Sentinel-1
# ---------------------------------------------------------------------------
def s1_collection(geom: ee.Geometry, start, end) -> ee.ImageCollection:
    """Sentinel-1 GRD, IW mode, dual-pol VV+VH, both orbit directions.

    COPERNICUS/S1_GRD in Earth Engine is already preprocessed with the
    Sentinel-1 Toolbox: orbit file applied, GRD border noise removal,
    thermal noise removal, radiometric calibration (sigma0) and terrain
    correction (orthorectification with SRTM 30 m / ASTER DEM), stored in dB.
    """
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))


def _speckle_filter(img: ee.Image) -> ee.Image:
    """3x3 focal median in linear power units (simple, robust speckle filter).
    Temporal median compositing afterwards reduces speckle further."""
    lin = ee.Image(10).pow(img.divide(10))
    lin = lin.focalMedian(radius=1, kernelType="square", units="pixels")
    return lin.copyProperties(img, img.propertyNames())


def monthly_composite(geom: ee.Geometry, year: int, month: int) -> ee.Image:
    """Median monthly composite -> bands VV, VH, VV_VH (all dB)."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = s1_collection(geom, start, end).map(_speckle_filter)
    lin = col.median()
    db = lin.log10().multiply(10)
    comp = db.addBands(db.select("VV").subtract(db.select("VH")).rename("VV_VH"))
    comp = comp.select(["VV", "VH", "VV_VH"])
    empty = ee.Image.constant([C.S1_NODATA] * 3).rename(C.S1_BANDS)
    out = ee.Image(ee.Algorithms.If(col.size().gt(0), comp, empty))
    return out.unmask(C.S1_NODATA).toFloat()


def scene_count(geom: ee.Geometry, year: int, month: int) -> int:
    start = ee.Date.fromYMD(year, month, 1)
    return s1_collection(geom, start, start.advance(1, "month")).size().getInfo()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def worldcover_label() -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    src = list(C.WORLDCOVER_REMAP.keys())
    dst = list(C.WORLDCOVER_REMAP.values())
    return wc.remap(src, dst, C.IGNORE_INDEX).unmask(C.IGNORE_INDEX).rename("label").toUint8()


def dynamic_world_label(geom: ee.Geometry, start, end) -> ee.Image:
    """Independent reference (Sentinel-2 based) for the prototype sanity check.
    Dynamic World classes are remapped to our 6 classes."""
    # DW: 0 water,1 trees,2 grass,3 flooded_veg,4 crops,5 shrub_scrub,6 built,7 bare,8 snow
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(geom).filterDate(start, end).select("label").mode())
    return dw.remap([0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [0, 2, 4, 4, 3, 4, 1, 5, C.IGNORE_INDEX],
                    C.IGNORE_INDEX).unmask(C.IGNORE_INDEX).rename("label").toUint8()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def fetch(image: ee.Image, epsg: int, x0: float, y0: float,
          width: int = C.PATCH_SIZE, height: int | None = None,
          scale: int = C.SCALE) -> np.ndarray:
    """Download an image on an exact UTM grid -> array (bands, H, W)."""
    height = height or width
    req = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -scale, "translateY": y0},
            "crsCode": f"EPSG:{epsg}",
        },
    }
    arr = ee.data.computePixels(req)
    return np.stack([arr[b] for b in arr.dtype.names])
