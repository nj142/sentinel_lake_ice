"""
Sentinel_DCC.py — MPI + multiprocessing Random Forest classification
                      for Sentinel-2 imagery on the DCC

WHAT THE SCRIPT DOES
--------------------
Point it at one study site's folder of S2 images and a single lake mask,
and it will:

  1. Split the images in that folder evenly across MPI ranks (one rank per
     CPU task in run.sh — ranks live on the supercomputer's compute nodes).
  2. Within each rank, split the lakes for that study site evenly across
     worker processes (16 workers/rank by default — set in run.sh).
  3. Run the same ordered list of per-lake steps on every lake — see
     `process_lake()` below.  By default STEP 5 calls a Random Forest, but
     you can replace it with band thresholding, NDWI, or any other
     algorithm; you can also insert extra masks as additional STEPs.

USAGE
-----
    mpirun -n 8 python Sentinel_DCC.py
    mpirun -n 8 python Sentinel_DCC.py --nimages 10 --workers 30

LAYOUT OF THIS FILE
-------------------
The script has two sections separated by a clearly-marked divider:

  1. USER-EDITABLE SECTION (just below this docstring)
       Input paths, band configuration, and the per-lake processing
       steps.  Edit these to point at different data, swap algorithms,
       or add masks.  No HPC knowledge required.

  2. HPC MACHINERY (below the divider)
       MPI work distribution, shared memory, file discovery, progress
       CSVs, timing instrumentation.  You should not need to edit
       anything here.

OUTPUT
------
    {OUTPUT_DIR}/progress/{image_folder}.csv         per-image, written live
    {OUTPUT_DIR}/Sentinel_DCC_<TIMESTAMP>.csv     final combined CSV
"""

import os
import sys
import re
import glob
import time
import json
import csv
import shutil
import argparse
import random
import tempfile
import subprocess
import multiprocessing as mp
import multiprocessing.shared_memory as shm_mod
from collections import defaultdict
from multiprocessing import resource_tracker
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.transform import Affine, rowcol
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from osgeo import ogr, osr
osr.UseExceptions()
ogr.UseExceptions()

try:
    from mpi4py import MPI
    COMM    = MPI.COMM_WORLD
    RANK    = COMM.Get_rank()
    SIZE    = COMM.Get_size()
    HAS_MPI = True

except ImportError:
    print("WARNING: mpi4py not found — running serial (rank 0 of 1)")
    COMM, RANK, SIZE, HAS_MPI = None, 0, 1, False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                      USER-EDITABLE SECTION                           ║
# ║                                                                      ║
# ║  Four things to edit in this section:                                ║
# ║    1. INPUT FILEPATHS  · point at your study site's data             ║
# ║    2. BAND CONFIG      · which bands and SCL codes to use            ║
# ║    3. process_lake()   · the ordered STEPs run on every lake         ║
# ║    4. OUTPUT_COLUMNS   · which columns appear in the output CSV      ║
# ║                                                                      ║
# ║  Everything below the END-OF-USER-SECTION divider is HPC machinery   ║
# ║  (MPI, shared memory, work distribution) and does not need editing.  ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# -----------------------------------------------------------------------
# INPUT FILEPATHS
# -----------------------------------------------------------------------
# Folder containing one sub-folder per Sentinel-2 image for the study
# site.  Each sub-folder must hold the band TIFs named exactly as the
# entries in FEATURE_BANDS plus SCL.tif.  Every image must share the
# same band layout.
S2_FOLDER  = "/work/nj142/S2/StudySite"

# Single shapefile of lake polygons covering the study site.  Lakes inside
# each S2 image's footprint are split evenly across the worker pool.
LAKE_MASK  = "/work/nj142/ALPOD_Tiles/StudySite_ALPOD/StudySite_ALPOD.shp"

# Trained Random Forest model (joblib package containing model,
# label_encoder, and feature_columns).
RF_MODEL   = "/work/nj142/S2/Models/StudySite/StudySite_freezeup_RFmodel.joblib"

# Where to write per-image progress CSVs and the final combined output.
OUTPUT_DIR = "/hpc/home/nj142/Output"


# -----------------------------------------------------------------------
# BAND CONFIGURATION
# -----------------------------------------------------------------------
# Feature bands present in each S-2 image folder, named exactly as is.
FEATURE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# 10 m reference band — every other band is resampled to its grid, and
# its bounding box is used as the image footprint.  (B02 is always 10 m.)
REF_BAND = "B02"

# Sentinel-2 SCL codes considered invalid in STEP 3 (clouds, shadows,
# saturation, no-data, ...).
#   0=no data  1=saturated  2=dark area  3=cloud shadow
#   7=unclassified  8=cloud medium  9=cloud high  10=thin cirrus
SCL_INVALID_CODES = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.uint16)

# Column in the lake mask shapefile holding each lake's unique integer ID.
LAKE_ID_COL = "id"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                       OUTPUT CSV LAYOUT                              ║
# ║                                                                      ║
# ║  One row is written per lake per image.  Columns come from two       ║
# ║  places:                                                             ║
# ║                                                                      ║
# ║    PER-IMAGE columns  (filled in by HPC machinery — same value for   ║
# ║                        every lake in a given image):                 ║
# ║      rank              MPI rank that processed the image             ║
# ║      study_site        basename of S2_FOLDER                         ║
# ║      year              YYYY parsed from the image folder name        ║
# ║      image_folder      S2 image folder name                          ║
# ║      unix_timestamp    midnight UTC of the acquisition date          ║
# ║      n_lakes_in_image  lakes inside this image's footprint           ║
# ║      read_time_s       wall seconds reading + resampling the bands   ║
# ║      rf_time_s         wall seconds in pool.map for this image       ║
# ║      error             error message, or "" on success               ║
# ║                                                                      ║
# ║    PER-LAKE columns   (returned from STEP 6 of process_lake — every  ║
# ║                        key in that dict becomes a CSV column):       ║
# ║      lake_id            unique lake ID from the lake mask            ║
# ║      ice_pixels         ice pixel count from STEP 5                  ║
# ║      water_pixels       water pixel count from STEP 5                ║
# ║      n_scl_valid_pixels ice_pixels + water_pixels                    ║
# ║                                                                      ║
# ║  Header + sample row:                                                ║
# ║    rank,study_site,year,image_folder,unix_timestamp,                 ║
# ║      n_lakes_in_image,read_time_s,rf_time_s,error,                   ║
# ║      lake_id,ice_pixels,water_pixels,n_scl_valid_pixels              ║
# ║    3,StudySite,2019,S2A_4WFD_20190903_0_L2A,1567468800,              ║
# ║      180,4.21,8.15,,12345,512,2048,2560                              ║
# ║                                                                      ║
# ║  To add a column:                                                    ║
# ║    1. Compute its value in process_lake() (in any STEP).             ║
# ║    2. Add it to the dict returned in STEP 6.                         ║
# ║    3. Add its name to OUTPUT_COLUMNS at the end of this section.     ║
# ║                                                                      ║
# ║  To remove a column: delete its name from OUTPUT_COLUMNS.  (You can  ║
# ║  leave it in the STEP 6 dict — extras are silently dropped.)         ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                   PER-LAKE PROCESSING STEPS                          ║
# ║                                                                      ║
# ║  Each worker calls `process_lake()` ONCE PER LAKE assigned to it.    ║
# ║  Steps run in the order they appear below — change, reorder, add,    ║
# ║  or remove steps to alter the analysis.                              ║
# ║                                                                      ║
# ║  Common edits:                                                       ║
# ║    · Replace STEP 5 (Random Forest) with a band threshold, NDWI,     ║
# ║      or any other classifier — just set ice_pixels & water_pixels.   ║
# ║    · Add a new STEP that ANDs an extra mask into combined_mask       ║
# ║      (e.g. terrain mask, water-body type, NDWI threshold).           ║
# ║    · Reorder steps if the data dependencies still hold.              ║
# ║    · Add a new column in STEP 6 (then add it to OUTPUT_COLUMNS).     ║
# ║                                                                      ║
# ║  band_data layout (set up for you by the HPC machinery):             ║
# ║    shape : (len(FEATURE_BANDS) + 1, rows, cols)   dtype: uint16      ║
# ║    slices 0 .. len(FEATURE_BANDS)-1   feature bands in order         ║
# ║    slice  len(FEATURE_BANDS)          SCL band                       ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def process_lake(lake_id, lake_geom, band_data, transform, nodata):
    """
    Run the full per-lake analysis on shared raster data.

    Parameters
    ----------
    lake_id    : int                       unique lake identifier
    lake_geom  : shapely.geometry.Polygon  polygon in the raster's CRS
    band_data  : np.ndarray, uint16        shape (n_feature_bands + 1, rows, cols)
                                           — last slice is SCL
    transform  : rasterio.transform.Affine raster-to-world transform
    nodata     : int or None               instrument fill value (None if not set)

    Returns
    -------
    dict — one entry per per-lake CSV column.  See STEP 6 below and
    OUTPUT_COLUMNS at the bottom of this section for the column list.
    """
    n_feature  = len(FEATURE_BANDS)
    rows, cols = band_data.shape[1], band_data.shape[2]

    # Defaults for the per-lake CSV fields.  STEP 5 overwrites
    # ice_pixels / water_pixels when valid pixels survive STEPs 2–4.
    ice_pixels   = 0
    water_pixels = 0

    # ┌────────────────────────────────────────────────────────────────┐
    # │  STEP 1  ·  COMPUTE PIXEL-SPACE WINDOW AROUND THIS LAKE        │
    # └────────────────────────────────────────────────────────────────┘
    # Translate the lake polygon's geographic bounds into pixel
    # row/col indices, giving us a small rectangular window that
    # contains the lake.  Every later step works only on this window
    # — not the whole 10 980 × 10 980 image — which is the dominant
    # per-lake speedup.
    minx, miny, maxx, maxy = lake_geom.bounds
    row_top, col_left  = rowcol(transform, minx, maxy)
    row_bot, col_right = rowcol(transform, maxx, miny)

    row_off = max(0,    int(row_top))
    col_off = max(0,    int(col_left))
    row_end = min(rows, int(row_bot)   + 1)
    col_end = min(cols, int(col_right) + 1)

    # If the lake bbox doesn't overlap the image, skip STEPs 2–5 and
    # fall through to STEP 6 with ice_pixels = water_pixels = 0.
    if row_off < row_end and col_off < col_end:
        win_rows, win_cols = row_end - row_off, col_end - col_off
        win_transform      = transform * Affine.translation(col_off, row_off)

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 2  ·  BURN LAKE POLYGON TO A BOOLEAN PIXEL MASK          │
        # └────────────────────────────────────────────────────────────────┘
        # rasterio.features.geometry_mask returns True for pixels INSIDE
        # the polygon (because invert=True).  We rasterize only into the
        # window, not the whole image.
        inside_mask = geometry_mask(
            [lake_geom],
            transform=win_transform,
            invert=True,
            out_shape=(win_rows, win_cols),
        )

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 3  ·  REMOVE CLOUDY / INVALID PIXELS USING SCL           │
        # └────────────────────────────────────────────────────────────────┘
        # The Sentinel-2 Scene Classification Layer (SCL) flags clouds,
        # shadows, saturation, and no-data pixels.  Mask anything in
        # SCL_INVALID_CODES out of the lake.  AND the result into the
        # combined mask of pixels that survive into STEP 4.
        scl_window    = band_data[n_feature, row_off:row_end, col_off:col_end]
        scl_valid     = ~np.isin(scl_window, SCL_INVALID_CODES)
        combined_mask = inside_mask & scl_valid

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 4  ·  EXTRACT BAND VALUES FOR SURVIVING PIXELS           │
        # └────────────────────────────────────────────────────────────────┘
        # Slice ALL bands within the window — feature bands plus SCL — and
        # keep only the columns that pass combined_mask.  Result shape:
        #   (n_feature_bands + 1, n_valid_pixels)
        # SCL is included as a feature because the RF model was trained
        # with it.  If your downstream algorithm in STEP 5 doesn't want
        # SCL as a feature, drop the last row here.
        pixel_data = band_data[:, row_off:row_end, col_off:col_end][:, combined_mask]

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 5  ·  CLASSIFY EACH PIXEL WITH RANDOM FOREST             │
        # └────────────────────────────────────────────────────────────────┘
        # Hand the (bands × pixels) matrix to the trained RF model and
        # tally ice vs. water predictions.  Replace the body of this block
        # with another classifier (band thresholding, NDWI, deep learning,
        # ...) — just set ice_pixels and water_pixels.
        if pixel_data.size > 0:
            package      = RF_MODEL_PACKAGE
            model        = package["model"]
            le           = package["label_encoder"]
            feature_cols = package["feature_columns"]

            # Drop residual all-zero columns (instrument fill value) as
            # a safety net for anything SCL didn't catch.
            if nodata is not None:
                valid_cols = ~np.all(pixel_data == nodata, axis=0)
                pixels_np  = pixel_data[:, valid_cols].T
            else:
                pixels_np  = pixel_data.T

            if pixels_np.size > 0:
                if pixels_np.shape[1] != len(feature_cols):
                    raise ValueError(
                        f"Band count mismatch: stack has {pixels_np.shape[1]} "
                        f"bands but model expects {len(feature_cols)} "
                        f"features {feature_cols}"
                    )

                pixels_df   = pd.DataFrame(pixels_np, columns=feature_cols)
                predictions = model.predict(pixels_df)

                ice_idx   = int(np.where(le.classes_ == "ice")[0][0])
                water_idx = int(np.where(le.classes_ == "water")[0][0])

                ice_pixels   = int(np.sum(predictions == ice_idx))
                water_pixels = int(np.sum(predictions == water_idx))

    # ┌────────────────────────────────────────────────────────────────┐
    # │  STEP 6  ·  BUILD OUTPUT ROW (per-lake CSV columns)            │
    # └────────────────────────────────────────────────────────────────┘
    # Each key returned here becomes a per-lake column in the CSV.
    # Per-image columns (rank, year, image_folder, ...) are added
    # automatically by the HPC machinery — don't include them here.
    # If you add a key, also add its name to OUTPUT_COLUMNS below.
    return {
        "lake_id":            lake_id,
        "ice_pixels":         ice_pixels,
        "water_pixels":       water_pixels,
        "n_scl_valid_pixels": ice_pixels + water_pixels,
    }


# -----------------------------------------------------------------------
# OUTPUT CSV COLUMN ORDER
# -----------------------------------------------------------------------
# Final list of column names written to the CSV, in the order they will
# appear.  Per-image columns (filled in by HPC machinery) are listed
# first, then per-lake columns from STEP 6 of process_lake.  Reorder,
# drop, or add entries to change the output.
OUTPUT_COLUMNS = [
    # ── Per-image columns (set automatically by HPC machinery) ────────
    "rank",              # MPI rank that processed the image
    "study_site",        # basename of S2_FOLDER
    "year",              # YYYY parsed from the image folder name
    "image_folder",      # S2 image folder name
    "unix_timestamp",    # midnight UTC of the acquisition date
    "n_lakes_in_image",  # lakes inside this image's footprint
    "read_time_s",       # wall seconds reading + resampling the bands
    "rf_time_s",         # wall seconds in pool.map for this image
    "error",             # error message, or "" on success

    # ── Per-lake columns (returned from STEP 6 of process_lake) ───────
    "lake_id",
    "ice_pixels",
    "water_pixels",
    "n_scl_valid_pixels",
]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                    END OF USER-EDITABLE SECTION                      ║
# ║                                                                      ║
# ║  Below: HPC machinery — MPI ranks, multiprocessing pool, POSIX       ║
# ║  shared memory, ogr2ogr lake clipping, file discovery, progress      ║
# ║  CSVs, timing report.  You shouldn't need to edit anything here.     ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# DERIVED PATHS / GLOBALS
# ---------------------------------------------------------------------------

STUDY_SITE       = os.path.basename(os.path.normpath(S2_FOLDER))
PROGRESS_DIR     = os.path.join(OUTPUT_DIR, "progress")
RF_MODEL_PACKAGE = None        # populated by load_rf_model() before pool fork


# ---------------------------------------------------------------------------
# TIMING INFRASTRUCTURE
# ---------------------------------------------------------------------------

TIMINGS: defaultdict = defaultdict(float)


def _tick(key: str, elapsed: float) -> None:
    TIMINGS[key] += elapsed


def _timed(key: str):
    """Context manager — records wall time of the enclosed block into TIMINGS."""
    class _Ctx:
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            _tick(key, time.perf_counter() - self._t0)
    return _Ctx()


_TIMING_LABELS = {
    "footprint_extract":   "Raster footprint extraction (B02)",
    "clip_vector":         "ogr2ogr lake clip",
    "read_clipped_shp":    "Read clipped shapefile (gpd)",
    "raster_read":         "Band read + resample to 10 m (rasterio/GDAL)",
    "shm_alloc":           "Shared-memory alloc + copy",
    "geom_project":        "Lake CRS reproject",
    "build_jobs":          "Build worker job list",
    "pool_map":            "pool.map wall time (all lakes)",
    "shm_cleanup":         "Shared-memory cleanup",
    "worker_shm_attach":   "[worker] shm attach/detach",
    "worker_process_lake": "[worker] process_lake() total",
    "load_rf_model":       "Load RF model (joblib)",
    "write_progress_csv":  "Write per-image progress CSVs",
    "write_csv":           "Write final combined CSV",
}


def _print_timing_report(all_timings: dict, total_wall: float) -> None:
    sorted_items    = sorted(all_timings.items(), key=lambda kv: kv[1], reverse=True)
    total_accounted = sum(v for _, v in sorted_items)
    col_w           = 46
    print(f"\n{'='*72}", flush=True)
    print(f"  WALL-TIME BREAKDOWN  (summed across all {SIZE} rank(s))", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'Stage':<{col_w}} {'Seconds':>10}  {'% of sum':>9}", flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    for key, secs in sorted_items:
        label = _TIMING_LABELS.get(key, key)
        pct   = 100.0 * secs / total_accounted if total_accounted > 0 else 0.0
        print(f"  {label:<{col_w}} {secs:>10.2f}s  {pct:>8.1f}%", flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    print(f"  {'Sum of timed stages':<{col_w}} {total_accounted:>10.2f}s", flush=True)
    print(f"  {'Total script wall time':<{col_w}} {total_wall:>10.2f}s", flush=True)
    print(f"{'='*72}\n", flush=True)


# ---------------------------------------------------------------------------
# RF MODEL LOADING
# ---------------------------------------------------------------------------

def load_rf_model() -> None:
    """
    Load the trained RF package into RF_MODEL_PACKAGE before the
    multiprocessing pool is forked.  Workers inherit the model via
    copy-on-write — no re-loading per worker.

    n_jobs is forced to 1 to avoid contention with the worker pool.
    """
    global RF_MODEL_PACKAGE
    t0 = time.perf_counter()
    if not os.path.isfile(RF_MODEL):
        raise FileNotFoundError(f"RF model not found: {RF_MODEL}")
    package = joblib.load(RF_MODEL)
    package["model"].set_params(n_jobs=1)
    RF_MODEL_PACKAGE = package
    print(f"[rank {RANK}] Loaded RF model  <-  {RF_MODEL}", flush=True)
    _tick("load_rf_model", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# MULTIPROCESSING WORKER (thin wrapper around process_lake)
# ---------------------------------------------------------------------------

def _rf_worker(args: tuple) -> tuple:
    """
    Multiprocessing worker — handles the shared-memory dance and then
    calls the user-defined process_lake() exactly once for one lake.

    Shared memory layout
    --------------------
    dtype : uint16
    shape : (len(FEATURE_BANDS) + 1, rows, cols)
      slices 0 .. len(FEATURE_BANDS)-1   feature bands in FEATURE_BANDS order
      slice  len(FEATURE_BANDS)          SCL (cast to uint16)

    Returns
    -------
    (lake_id, lake_row, worker_timings)
        lake_row        dict from process_lake() STEP 6, or None on error
        worker_timings  dict mapping timing-key -> seconds
    """
    (lake_id, geom_wkt, transform_tuple, shm_name,
     raster_shape, raster_dtype_str, nodata) = args

    wtimings: dict = defaultdict(float)
    existing_shm   = None

    try:
        # Attach to shared memory
        t = time.perf_counter()
        existing_shm = shm_mod.SharedMemory(name=shm_name)
        # Prevent the resource tracker from trying to unlink a block
        # owned by the main process (only the main needs to clean up).
        try:
            resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        except (KeyError, ValueError):
            pass
        band_data = np.ndarray(
            raster_shape,
            dtype=np.dtype(raster_dtype_str),
            buffer=existing_shm.buf,
        )
        wtimings["worker_shm_attach"] += time.perf_counter() - t

        transform = Affine(*transform_tuple)
        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(geom_wkt)

        # Run the user-defined per-lake pipeline
        t = time.perf_counter()
        lake_row = process_lake(int(lake_id), geom, band_data, transform, nodata)
        wtimings["worker_process_lake"] += time.perf_counter() - t

        existing_shm.close()
        return (int(lake_id), lake_row, dict(wtimings))

    except Exception as exc:
        if existing_shm is not None:
            try: existing_shm.close()
            except Exception: pass
        print(f"[worker] lake {lake_id} error: {exc}", flush=True)
        return (int(lake_id), None, dict(wtimings))


# ---------------------------------------------------------------------------
# RASTER + VECTOR HELPERS
# ---------------------------------------------------------------------------

def get_footprint_from_band(band_path: str) -> tuple:
    """
    Open the reference band and return its footprint (in WGS84) plus
    metadata used by the rest of the pipeline.

    Returns
    -------
    footprint_4326, crs, transform, height, width, nodata
    """
    with rasterio.open(band_path) as src:
        crs       = src.crs
        transform = src.transform
        height    = src.height
        width     = src.width
        nodata    = src.nodata
        bounds    = src.bounds

    l, b, r, t = transform_bounds(crs, "EPSG:4326", *bounds)
    return (mapping(Polygon([(l, b), (r, b), (r, t), (l, t)])),
            crs, transform, height, width, nodata)


def read_band_stack(img_folder: str, height: int, width: int) -> np.ndarray:
    """
    Read FEATURE_BANDS + SCL from *img_folder*, resampled to (height, width)
    on the 10 m grid using nearest-neighbour (matches training pipeline).
    """
    all_bands = FEATURE_BANDS + ["SCL"]
    stack     = np.empty((len(all_bands), height, width), dtype=np.uint16)

    for i, bname in enumerate(all_bands):
        bpath = os.path.join(img_folder, f"{bname}.tif")
        if not os.path.isfile(bpath):
            raise FileNotFoundError(f"Missing band file: {bpath}")
        with rasterio.open(bpath) as src:
            stack[i] = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest,
            ).astype(np.uint16)

    return stack


def clip_vector_with_geometry(vector_path: str, geometry: dict,
                               output_path: str) -> int:
    """
    Clip *vector_path* to features entirely within *geometry* (WGS84 dict).
    Writes a new shapefile to *output_path*; returns its feature count.
    """
    geom    = ogr.CreateGeometryFromJson(json.dumps(geometry))
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom.AssignSpatialReference(srs4326)

    src_ds     = ogr.Open(vector_path)
    src_lyr    = src_ds.GetLayer()
    vec_srs    = src_lyr.GetSpatialRef()
    vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    layer_name = src_lyr.GetName()

    if not vec_srs.IsSame(srs4326):
        geom.Transform(osr.CoordinateTransformation(srs4326, vec_srs))

    auth_code = vec_srs.GetAuthorityCode(None)
    epsg      = int(auth_code) if auth_code else 4326
    sql       = (
        f"SELECT * FROM {layer_name} "
        f"WHERE ST_Within(Geometry, GeomFromText('{geom.ExportToWkt()}', {epsg}))"
    )
    subprocess.check_call([
        "ogr2ogr", "-f", "ESRI Shapefile",
        output_path, vector_path,
        "-dialect", "SQLite", "-sql", sql,
    ])

    out_ds = ogr.Open(output_path)
    count  = out_ds.GetLayer(0).GetFeatureCount()
    src_ds = None
    out_ds = None
    return count


def extract_year_from_folder(folder_name: str) -> str:
    m = re.search(r'(\d{8})', folder_name)
    return m.group(1)[:4] if m else "unknown"


def extract_unix_time_from_folder(folder_name: str) -> int:
    m = re.search(r'(\d{8})', folder_name)
    if m:
        try:
            return int(datetime.strptime(m.group(1), "%Y%m%d").timestamp())
        except ValueError:
            pass
    return 0


def _cleanup_shp(shp_path: str):
    base = os.path.splitext(shp_path)[0]
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"):
        try: os.remove(base + ext)
        except FileNotFoundError: pass


# ---------------------------------------------------------------------------
# PROGRESS / RESUME HELPERS
# ---------------------------------------------------------------------------

def get_progress_path(folder_name: str) -> str:
    return os.path.join(PROGRESS_DIR, f"{folder_name}.csv")


def image_already_done(folder_name: str) -> bool:
    return os.path.isfile(get_progress_path(folder_name))


def write_progress_csv(rows: list, folder_name: str) -> str:
    """Write per-image results immediately after pool.map() returns."""
    path = get_progress_path(folder_name)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", restval="",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def combine_progress_csvs(output_path: str) -> None:
    """Merge every per-image CSV into a single sorted master CSV."""
    progress_files = sorted(glob.glob(os.path.join(PROGRESS_DIR, "*.csv")))
    if not progress_files:
        print("[rank 0] WARNING: no progress CSVs found to combine.", flush=True)
        return

    all_rows = []
    for pf in progress_files:
        try:
            with open(pf, newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception as exc:
            print(f"[rank 0] WARNING: could not read {pf}: {exc}", flush=True)

    all_rows.sort(key=lambda r: (
        r.get("study_site", ""),
        r.get("year", ""),
        r.get("image_folder", ""),
        int(r.get("lake_id") or 0),
    ))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", restval="",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[rank 0] Combined {len(progress_files)} progress files "
        f"({len(all_rows)} rows) -> {output_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# FILE DISCOVERY (rank 0 only)
# ---------------------------------------------------------------------------

def discover_files(n_per_site: int | None, seed: int = 42) -> list:
    """
    Scan S2_FOLDER for valid image folders (each must contain every
    FEATURE_BANDS tif plus SCL.tif).  Optionally subsample *n_per_site*.

    Images that already have a progress CSV are skipped automatically
    so resumed runs don't reprocess completed work.
    """
    rng = random.Random(seed)

    if not os.path.isdir(S2_FOLDER):
        print(f"[rank 0] WARNING: {S2_FOLDER} not found", flush=True)
        return []

    candidates = []
    for img_folder in sorted(glob.glob(os.path.join(S2_FOLDER, "*"))):
        if not os.path.isdir(img_folder):
            continue
        folder_name = os.path.basename(img_folder)

        if not os.path.isfile(os.path.join(img_folder, f"{REF_BAND}.tif")):
            continue

        missing = [
            b for b in FEATURE_BANDS + ["SCL"]
            if not os.path.isfile(os.path.join(img_folder, f"{b}.tif"))
        ]
        if missing:
            print(
                f"[rank 0] WARNING: {folder_name} missing bands {missing} — skipping",
                flush=True,
            )
            continue

        candidates.append((img_folder, folder_name, extract_year_from_folder(folder_name)))

    if n_per_site is None:
        sample = candidates
        print(f"[rank 0] {STUDY_SITE}: using all {len(sample)} images", flush=True)
    else:
        sample = rng.sample(candidates, min(n_per_site, len(candidates)))
        print(
            f"[rank 0] {STUDY_SITE}: {len(candidates)} images available, "
            f"sampling {len(sample)}",
            flush=True,
        )

    records, n_skipped = [], 0
    for (img_folder, folder_name, year) in sample:
        if image_already_done(folder_name):
            n_skipped += 1
            continue
        records.append({
            "path":        img_folder,
            "year":        year,
            "folder_name": folder_name,
        })

    if n_skipped:
        print(
            f"[rank 0] {STUDY_SITE}: skipping {n_skipped} already-completed "
            f"image(s) (progress CSVs found in {PROGRESS_DIR})",
            flush=True,
        )

    return records


# ---------------------------------------------------------------------------
# WORK DISTRIBUTION
# ---------------------------------------------------------------------------

def scatter_work(records: list) -> list:
    if not HAS_MPI:
        return records
    if RANK == 0:
        chunks = [[] for _ in range(SIZE)]
        for i, rec in enumerate(records):
            chunks[i % SIZE].append(rec)
        print("\n[rank 0] Work distribution:", flush=True)
        for r, chunk in enumerate(chunks):
            print(f"         rank {r:>2}: {len(chunk):>3} images", flush=True)
        print("", flush=True)
    else:
        chunks = None
    return COMM.scatter(chunks, root=0)


# ---------------------------------------------------------------------------
# PER-IMAGE PROCESSING
# ---------------------------------------------------------------------------

def process_image(rec: dict, pool: mp.Pool, tmp_dir: str) -> list:
    """
    Full S2 pipeline for one image folder.  Returns a list of row dicts
    and writes a per-image progress CSV immediately so partial results
    survive a job kill.
    """
    img_folder  = rec["path"]
    folder_name = rec["folder_name"]
    t0          = time.perf_counter()

    print(f"\n[rank {RANK}] START {STUDY_SITE}/{folder_name}", flush=True)

    base_row = {
        "rank":           RANK,
        "study_site":     STUDY_SITE,
        "year":           rec["year"],
        "image_folder":   folder_name,
        "unix_timestamp": extract_unix_time_from_folder(folder_name),
        "error":          "",
    }

    def err(msg, read_t=None):
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] ERROR {folder_name}: {msg}  elapsed={elapsed:.2f}s",
            flush=True,
        )
        # Image-level failure: write one row with the error message.
        # Per-lake fields (lake_id, ice_pixels, ...) are left blank by
        # the CSV writer's restval — no lakes were processed.
        rows = [{**base_row,
                 "n_lakes_in_image": 0,
                 "read_time_s":      round(read_t or elapsed, 4),
                 "rf_time_s":        -1,
                 "error":            msg}]
        t_csv = time.perf_counter()
        write_progress_csv(rows, folder_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return rows

    # 1. Footprint from the 10 m reference band
    ref_band_path = os.path.join(img_folder, f"{REF_BAND}.tif")
    try:
        with _timed("footprint_extract"):
            footprint, crs, transform, height, width, nodata = \
                get_footprint_from_band(ref_band_path)
    except Exception as exc:
        return err(f"Footprint extraction failed: {exc}")

    # 2. Clip the lake mask to the image footprint
    clip_out = os.path.join(tmp_dir, f"clip_r{RANK}_{folder_name}.shp")
    try:
        with _timed("clip_vector"):
            n_lakes = clip_vector_with_geometry(LAKE_MASK, footprint, clip_out)
    except Exception as exc:
        return err(f"clip_vector_with_geometry failed: {exc}")

    if n_lakes == 0:
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] DONE  {folder_name}  no lakes within footprint  "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
        _cleanup_shp(clip_out)
        t_csv = time.perf_counter()
        write_progress_csv([], folder_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return []

    # 3. Read clipped lakes
    try:
        with _timed("read_clipped_shp"):
            lakes_gdf = gpd.read_file(clip_out)
    except Exception as exc:
        _cleanup_shp(clip_out)
        return err(f"Could not read clipped shapefile: {exc}")
    _cleanup_shp(clip_out)

    # 4. Read all feature bands + SCL, resampled to 10 m
    try:
        with _timed("raster_read"):
            band_stack = read_band_stack(img_folder, height, width)
    except Exception as exc:
        return err(f"Band read/resample failed: {exc}")

    read_time = time.perf_counter() - t0

    # 5. Copy band stack into POSIX shared memory
    raster_shape = band_stack.shape
    raster_dtype = str(band_stack.dtype)
    shm_name     = f"rf_s2_rank{RANK}_{folder_name}"

    try:
        with _timed("shm_alloc"):
            image_shm  = shm_mod.SharedMemory(
                name=shm_name, create=True, size=band_stack.nbytes
            )
            shared_arr = np.ndarray(band_stack.shape, dtype=band_stack.dtype,
                                    buffer=image_shm.buf)
            shared_arr[:] = band_stack
            del band_stack, shared_arr
    except Exception as exc:
        return err(f"Shared memory allocation failed: {exc}")

    transform_tuple = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )

    # 6. Reproject lakes to raster CRS, build per-lake job list
    with _timed("geom_project"):
        lakes_proj = lakes_gdf.to_crs(crs)

    with _timed("build_jobs"):
        jobs = [
            (int(lake_row[LAKE_ID_COL]),
             lake_row.geometry.wkt,
             transform_tuple,
             shm_name,
             raster_shape,
             raster_dtype,
             nodata)
            for _, lake_row in lakes_proj.iterrows()
        ]

    n_workers_actual = pool._processes
    chunksize        = max(1, len(jobs) // max(1, n_workers_actual))

    print(
        f"[rank {RANK}]   distributing {n_lakes} lakes to {n_workers_actual} "
        f"workers (chunksize={chunksize}  read={read_time:.2f}s)",
        flush=True,
    )

    # 7. Dispatch lakes to the worker pool
    t_rf = time.perf_counter()
    with _timed("pool_map"):
        raw_results = pool.map(_rf_worker, jobs, chunksize=chunksize)
    rf_time = time.perf_counter() - t_rf

    results = []
    for lake_id, lake_row, wtimings in raw_results:
        for k, v in wtimings.items():
            _tick(k, v)
        results.append((lake_id, lake_row))

    # 8. Release shared memory
    with _timed("shm_cleanup"):
        try:
            image_shm.close()
            image_shm.unlink()
        except Exception as exc:
            print(
                f"[rank {RANK}] WARNING: shared memory cleanup failed: {exc}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0

    print(
        f"[rank {RANK}] DONE  {STUDY_SITE}/{folder_name}  "
        f"{n_lakes} lakes classified  "
        f"read={read_time:.2f}s  rf={rf_time:.2f}s  elapsed={elapsed_total:.2f}s",
        flush=True,
    )

    per_image_metrics = {
        "n_lakes_in_image": n_lakes,
        "read_time_s":      round(read_time, 4),
        "rf_time_s":        round(rf_time,   4),
    }

    rows = []
    for lake_id, lake_row in results:
        if lake_row is None:
            # Worker raised — keep lake_id and flag the row; per-lake
            # fields stay blank (handled by the CSV writer's restval).
            rows.append({**base_row, **per_image_metrics,
                         "lake_id": lake_id,
                         "error":   "worker exception"})
        else:
            rows.append({**base_row, **per_image_metrics, **lake_row})

    # 9. Write per-image progress CSV immediately
    t_csv = time.perf_counter()
    csv_path = write_progress_csv(rows, folder_name)
    _tick("write_progress_csv", time.perf_counter() - t_csv)
    print(f"[rank {RANK}] PROGRESS  {folder_name}  -> {csv_path}", flush=True)

    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if RANK == 0:
        parser = argparse.ArgumentParser(
            description="RF benchmark S2 — MPI + multiprocessing"
        )
        parser.add_argument("--nimages", type=int, default=None,
                            help="Images to sample from S2_FOLDER (default: all)")
        parser.add_argument("--workers", type=int,
                            default=max(1, mp.cpu_count() - 1),
                            help="Pool workers per rank (default: cpu_count - 1)")
        args = parser.parse_args()
        cfg  = {"nimages": args.nimages, "workers": args.workers}
    else:
        cfg = None

    if HAS_MPI:
        cfg = COMM.bcast(cfg, root=0)

    n_sample, n_workers = cfg["nimages"], cfg["workers"]

    if RANK == 0:
        if not os.path.isfile(LAKE_MASK):
            print(f"[rank 0] WARNING: lake mask not found: {LAKE_MASK}", flush=True)
        else:
            print(f"[rank 0] Lake mask  ->  {LAKE_MASK}", flush=True)

    tmp_dir = tempfile.mkdtemp(prefix=f"rf_s2_rank{RANK}_")

    if RANK == 0:
        global_start = time.time()
        n_label = str(n_sample) if n_sample is not None else "ALL"
        print(f"\n{'='*72}", flush=True)
        print(f"  Sentinel_DCC.py  |  {SIZE} MPI ranks  |  {n_workers} workers/rank",
              flush=True)
        print(f"  Study site : {STUDY_SITE}", flush=True)
        print(f"  S2 folder  : {S2_FOLDER}", flush=True)
        print(f"  Images     : {n_label}", flush=True)
        print(f"  Feat. bands: {FEATURE_BANDS}", flush=True)
        print(f"  SCL excl.  : {SCL_INVALID_CODES.tolist()}", flush=True)
        print(f"  Progress   : {PROGRESS_DIR}", flush=True)
        print(f"{'='*72}\n", flush=True)

        records = discover_files(n_sample)
        n_remaining = len(records)
        print(
            f"[rank 0] {n_remaining} image(s) remaining to process "
            f"(completed images already skipped)",
            flush=True,
        )
        if not records:
            print(
                "[rank 0] All images already complete.  "
                "Combining progress CSVs into final output...",
                flush=True,
            )
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"Sentinel_DCC_{timestamp}.csv")
            t_csv = time.perf_counter()
            combine_progress_csvs(output_path)
            _tick("write_csv", time.perf_counter() - t_csv)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            (COMM.Abort(0) if HAS_MPI else sys.exit(0))
    else:
        records      = None
        global_start = None

    local_records = scatter_work(records)

    # Load RF model before forking the pool — workers inherit via copy-on-write
    load_rf_model()

    ctx        = mp.get_context("fork")
    local_rows = []
    with ctx.Pool(processes=n_workers) as pool:
        for rec in local_records:
            local_rows.extend(process_image(rec, pool, tmp_dir))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if HAS_MPI:
        all_timings_nested = COMM.gather(dict(TIMINGS), root=0)
    else:
        all_timings_nested = [dict(TIMINGS)]

    if RANK == 0:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"Sentinel_DCC_{timestamp}.csv")

        t_csv = time.perf_counter()
        combine_progress_csvs(output_path)
        _tick("write_csv", time.perf_counter() - t_csv)

        total_elapsed = time.time() - global_start

        combined: defaultdict = defaultdict(float)
        for rank_timings in all_timings_nested:
            for k, v in rank_timings.items():
                combined[k] += v

        print(f"\n{'='*72}", flush=True)
        print(f"  Total wall time : {total_elapsed:.2f}s  "
              f"({total_elapsed/60:.1f} min)", flush=True)
        print(f"  Output          : {output_path}", flush=True)
        print(f"{'='*72}\n", flush=True)

        _print_timing_report(combined, total_elapsed)


if __name__ == "__main__":
    mp.freeze_support()
    main()
