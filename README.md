# Sub-seasonal forecasting of cropland productivity with satellite soil moisture

Code for Adebayo and Nakalembe (2026), *Sub-seasonal forecasting of cropland productivity
anomalies using satellite soil moisture in water-limited environments*, Remote Sensing of
Environment 347, 115645. https://doi.org/10.1016/j.rse.2026.115645

A ConvLSTM forecasts VIIRS NIRv anomalies 8 to 40 days ahead from past NIRv and SMAP L4
root-zone soil moisture (RZSM) over cropland in Eastern and Southern Africa.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12. Training uses a GPU when one is available.

## Data

All inputs are public. `data/` holds the 32 x 32 cell study grid and the GEOGLAM crop
calendar attributes of each cell; everything else is downloaded or derived:

```bash
python preprocessing/export_patches.py --grids data/grid/study_grid.shp --bucket BUCKET --project PROJECT
mkdir -p work/patches && gsutil -m cp 'gs://BUCKET/patches/*.tif' work/patches/
python preprocessing/cropland_mask.py --grids data/grid/study_grid.shp --cropland cropland/africa_*.tif --output_dir work/cropland_masks
python preprocessing/composite.py --patches work/patches --cropland_masks work/cropland_masks --output_dir work/composites
python preprocessing/standardize.py --composites work/composites --output_dir work/zscore
python preprocessing/lag_analysis.py --composites work/composites --output_dir work/lags --data_root work/zscore
python preprocessing/aridity.py --aridity aridity_classes_9km.tif --composites work/composites --output_dir work/aridity
```

`export_patches.py` writes daily VIIRS VNP43IA4 and SMAP SPL4SMGP stacks to Google Cloud
Storage through Earth Engine. The cropland maps are those of Khan et al. (2026); the aridity
raster holds Global Aridity Index v3 classes (P/ET0 < 0.5 water-limited) on the 9 km grid.
Run `lag_analysis.py --target NDVI` as well for the NDVI experiments.

## Training

```bash
python train.py --config configs/nirv_rzsm.yaml --data_root work/zscore --aridity_dir work/aridity \
    --output_dir runs/huber/nirv_rzsm_t6
```

This trains five seeds and writes per-seed and summary test metrics. `scripts/run_experiments.sh`
trains every model in the paper.

The scripts in `analysis/` and `supplementary/` make the figures and tables from those runs;
`analysis/common.py` and the comments in `run_experiments.sh` list which runs each one uses.

## Citation

```bibtex
@article{adebayo2026subseasonal,
  title   = {Sub-seasonal forecasting of cropland productivity anomalies using satellite soil moisture in water-limited environments},
  author  = {Adebayo, Adebowale Daniel and Nakalembe, Catherine},
  journal = {Remote Sensing of Environment},
  volume  = {347},
  pages   = {115645},
  year    = {2026},
  doi     = {10.1016/j.rse.2026.115645}
}
```
