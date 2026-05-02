Sentinel_Lake_Ice - README for CEE690 Class Final

---

**Team Members:** Noah Jacobs  
**Semester:** Spring 2026  
**Instructor:** Dr. Nathaniel Chaney  
**Institution:** Duke University

---

## Project Summary

This project implements a **parallelized HPC pipeline** for classifying global lake ice cover from Sentinel-2 imagery on the **Duke Compute Cluster (DCC)**. The script (`Sentinel_DCC.py`) uses **MPI across multiple compute nodes** in order to run a trained Random Forest classifier over thousands of lake polygons and hundreds of satellite images, producing per-lake, per-image ice pixel counts as time series CSVs. This is a modular adaptation of my previous Sentinel-2 script, improved for readibility, repeatability, and use by fellow labmates for accelerating Sentinel-2 lake phenology workflows.  It has backstops for errors midway through runtime, is packaged on my Github for direct deployment, and is set up for automation of processing across dozens of study sites.

![alt text](<Study_Sites.png>)

I will be using this script to build lake ice time series across the global cryosphere, which will help us understand the role of lake geometry, landscape characteristics, and biome in controlling ice phenology worldwide.  I selected the 44 50x50km  cells shown above from a global equal-area Mollweide sample grid, using the following characteristics: 

    1. Cell must have winter isotherm of 2 months below 0* celsius 
    
    2. Cell must have PLD lake density > 300
    
    3. Choose 1% of cells from each biome class remaining from the WWF global biome dataset

These 44 grid cells cells consist of more than 42,000 lakes, for which I will download all available satellite imagery in the Sentinel-2 record, build freeze-up and break-up time series for each, and output csv files for ice timing analysis using the DCC.

This project is designed to be modular, so that in the future my lab members can adapt it to run various tasks on lake polygons with the power of the Duke Compute Cluster, without having to build an MPI script from scratch.


---

## Problem Statement and Objectives

Classifying lake ice cover from Sentinel-2 imagery at scale requires processing hundreds of images, each containing thousands of individual lake polygons. Running this sequentially is prohibitively slow-- even just a few sites would take months to run on a typical lab PC. The pipeline efficiently distributes both image-level and lake-level work across HPC resources while writing fault-tolerant progress outputs that survive runtime stoppage.

**Objectives / Expected Outcomes:**

1. Distribute Sentinel-2 image processing evenly across **MPI ranks** (one rank per CPU task), enabling multi-node parallelism on the DCC.

2. Within each MPI rank, parallelize **per-lake Random Forest classification** across a configurable worker pool using POSIX shared memory to avoid redundant band data copies.

3. Write **per-image progress CSVs** immediately after processing so that partial results are preserved on job interruption, and support **resume** by automatically skipping already-completed images on rerun.

4. Output a **combined time series CSV** of per-lake ice and water pixel counts suitable for downstream freeze-up date extraction.

---

## Methods

**Step 0. Input Discovery and Work Distribution**
   - Rank 0 scans `S2_FOLDER` for valid image subdirectories (each must contain all `FEATURE_BANDS` TIFs plus `SCL.tif`).
   - Images already present in the progress directory are skipped automatically to support job resumption.
   - The remaining image list is scattered evenly across all MPI ranks via `MPI.COMM_WORLD.scatter()`.

**Step 1. RF Model Loading**
   - Each rank loads the trained Random Forest package (`joblib`) before forking the worker pool.
   - Workers inherit the model via copy-on-write to avoid redundant per-worker reload tasks.
   - `n_jobs` is forced to 1 on the loaded model to avoid thread contention with the worker pool.

**Step 2. Per-Image Raster + Vector Preparation**
   - Extract the image footprint from the 10 m reference band (`B02`) and reproject to WGS84.
   - Clip the lake mask shapefile to the image footprint using `ogr2ogr` with an SQLite spatial query.
   - Read and resample all feature bands (`B02`, `B03`, `B04`, `B08`, `B11`, `B12`) plus `SCL` to the 10 m grid using nearest-neighbour resampling.
   - Copy the full band stack into **POSIX shared memory** so all worker processes can read it without duplication.

**Step 3. Per-Lake Classification (Worker Pool)**
   - Each worker attaches to the shared memory block and runs `process_lake()` for one lake at a time.
   - Per lake: compute a pixel-space bounding window -> burn lake polygon to mask -> remove invalid pixels via SCL -> extract surviving band values -> classify with the Random Forest -> tally ice vs. water predictions.
   - Workers return `(lake_id, result_dict, worker_timings)` tuples; results are aggregated by the parent rank.

**Step 4. Output and Resume**
   - Per-image results are written to `{OUTPUT_DIR}/progress/{image_folder}.csv` immediately after `pool.map()` returns.
   - After all ranks complete, rank 0 merges all progress CSVs into a single sorted master CSV timestamped at `{OUTPUT_DIR}/Sentinel_DCC_<TIMESTAMP>.csv`.
   - A wall-time breakdown report is printed, summed across all MPI ranks.

---

## Repository / File Structure

```
.
├── Sentinel_DCC.py          # Main HPC pipeline script
├── run.sh                   # SLURM job submission script
├── Output/
│   ├── progress/            # Per-image intermediate CSVs (written live)
│   └── Sentinel_DCC_*.csv   # Final combined output CSV
```

---

## Configuration (User-Editable Section, designed so my labmates can use this script in the future)

All user-facing settings are contained in the clearly marked `USER-EDITABLE SECTION` at the top of the script. No HPC machinery needs to be touched for typical use, which I hope will be a useful tool for future lab mates looking to utilize the power of the DCC in their work.

| Variable | Description |
|---|---|
| `S2_FOLDER` | Path to folder of Sentinel-2 image subdirectories for the study site |
| `LAKE_MASK` | Path to the lake polygon shapefile covering the study site |
| `RF_MODEL` | Path to the trained `.joblib` Random Forest package |
| `OUTPUT_DIR` | Directory for progress CSVs and the final combined output |
| `OUTPUT_COLUMNS` | Ordered list of columns written to the output CSV |

---

## Usage

```
#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=freezeup_s2
#SBATCH --output=/hpc/home/nj142/Output/Sentinel_DCC_%j.out

module purg
module load OpenMPI/4.1.6 

source /hpc/home/nj142/miniconda3/etc/profile.d/conda.sh
conda activate python311

mpirun -n 25 python ~/Scripts/Sentinel_DCC.py --workers 15

```

A typical SLURM `run.sh` sets the number of MPI tasks and workers/rank to match the node allocation. DCC maxes out at 400 CPUs

---

## Output CSV Schema

One row is written per lake per image. Columns can be easily adapted based on what you want your output to look like:

| Column | Source | Description |
|---|---|---|
| `rank` | HPC machinery | MPI rank that processed the image |
| `study_site` | HPC machinery | Basename of `S2_FOLDER` |
| `year` | HPC machinery | YYYY parsed from the image folder name |
| `image_folder` | HPC machinery | S2 image folder name |
| `unix_timestamp` | HPC machinery | Midnight UTC of the acquisition date |
| `n_lakes_in_image` | HPC machinery | Lakes inside this image's footprint |
| `read_time_s` | HPC machinery | Wall seconds for band read + resample |
| `rf_time_s` | HPC machinery | Wall seconds for `pool.map` across all lakes |
| `error` | HPC machinery | Error message, or `""` on success |
| `lake_id` | `process_lake()` | Unique lake ID from the lake mask |
| `ice_pixels` | `process_lake()` | Ice pixel count from Random Forest |
| `water_pixels` | `process_lake()` | Water pixel count from Random Forest |
| `n_scl_valid_pixels` | `process_lake()` | Total valid (non-cloudy) pixels classified |

To add a column: compute its value in `process_lake()`, include it in the returned dict, and add its name to `OUTPUT_COLUMNS`.

---

## Datasets

| Dataset | Description | Source / Link |
|---|---|---|
| **Sentinel-2 SR Imagery (2017–2025)** | Multispectral imagery (L2A) used for ice classification | AWS STAC API |
| **Prior Lake Database (PLD)** | Global lake mask dataset including lakes ≥ 0.01 km² | [Wang et al. 2025, WRR](https://doi.org/10.1029/2023WR036896) |
| **WWF Terrestrial Ecoregions** | Global map of 867 terrestrial ecoregions used to categorize diverse climatic and ecological zones | [Olson et al. 2001, BioScience](https://doi.org/10.1641/0006-3568(2001)051[0933:TEOTWA]2.0.CO;2) |
| **Labelbox Annotations** | Manual delineations used to supplement lake masks | Generated from prior PhD repository by Noah Jacobs, Annie Cushman, and Lauren Coleman |
| **Trained RF Model** | `.joblib` package containing model, label encoder, and feature columns | Generated from prior PhD classification workflow |

---

## Python Packages

```python
numpy
pandas
scikit-learn
joblib
rasterio
geopandas
shapely
mpi4py
gdal / osgeo (ogr, osr)
multiprocessing (stdlib)
```
