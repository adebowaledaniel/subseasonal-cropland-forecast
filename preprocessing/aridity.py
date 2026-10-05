"""
Per-cell aridity class rasters for the regime-stratified evaluation.

Input is the aridity classification on the 9 km EASE-Grid 2.0 (EPSG:6933) of the patches:
1 = water-limited (Global Aridity Index v3 P/ET0 < 0.5), 2 = energy-limited, -9999 elsewhere.
It is cut to each cell's 32 x 32 footprint and written to {output_dir}/{LOC}/binary_ai.tif.

    python preprocessing/aridity.py --aridity work/aridity_classes_9km.tif \
        --composites work/composites --output_dir work/aridity
"""

import argparse
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aridity', required=True, help='classified aridity raster in EPSG:6933')
    p.add_argument('--composites', required=True, help='output of composite.py, for cell footprints')
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    with rasterio.open(args.aridity) as src:
        for loc_dir in sorted(d for d in Path(args.composites).iterdir() if d.is_dir()):
            with rasterio.open(next(loc_dir.glob('*.tif'))) as ref:
                window = from_bounds(*ref.bounds, src.transform)
            classes = src.read(1, window=window, boundless=True)
            profile = src.meta.copy()
            profile.update(height=classes.shape[0], width=classes.shape[1], count=1,
                           transform=src.window_transform(window))
            out = Path(args.output_dir) / loc_dir.name
            out.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out / 'binary_ai.tif', 'w', **profile) as dst:
                dst.write(classes, 1)


if __name__ == '__main__':
    main()
