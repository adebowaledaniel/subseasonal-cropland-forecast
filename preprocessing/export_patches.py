"""
Export daily VIIRS (VNP43IA4) NDVI/NIRv and SMAP L4 soil moisture stacks from Google Earth
Engine for every 32 x 32 grid cell (9 km, EPSG:6933), one GeoTIFF per cell and year.

Bands are named {VAR}_{YYYY-MM-DD} with VAR in NDVI, NIRV, sm_surface, sm_rootzone.
Files are written to gs://BUCKET/patches/{ISO3}_G_{GID}_viirs_{YEAR}.tif.

    python preprocessing/export_patches.py --grids data/grid/study_grid.shp \
        --bucket my-bucket --project my-gcp-project --start 2016 --end 2025
"""

import argparse
from datetime import date, timedelta

import ee
import geopandas as gpd

VIIRS = 'NASA/VIIRS/002/VNP43IA4'
SMAP = 'NASA/SMAP/SPL4SMGP/008'
EASE2 = 'EPSG:6933'


def viirs_daily(day: date, region) -> ee.Image:
    """Mean NDVI and NIRv for one day, keeping only full BRDF inversions (QA 0 or 1)."""

    def prepare(img):
        qa1 = img.select('BRDF_Albedo_Band_Mandatory_Quality_I1')
        qa2 = img.select('BRDF_Albedo_Band_Mandatory_Quality_I2')
        img = img.updateMask(qa1.lte(1).And(qa2.lte(1)))
        red, nir = img.select('Nadir_Reflectance_I1'), img.select('Nadir_Reflectance_I2')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        # NIRv = (NDVI - 0.08) * NIR, the bare-soil offset of Badgley et al. (2017)
        nirv = ndvi.subtract(0.08).multiply(nir).rename('NIRV')
        return img.addBands([ndvi, nirv])

    start, end = day.isoformat(), (day + timedelta(days=1)).isoformat()
    col = ee.ImageCollection(VIIRS).filterDate(start, end).filterBounds(region).map(prepare)
    return col.select(['NDVI', 'NIRV']).mean().toFloat().rename([f'NDVI_{start}', f'NIRV_{start}'])


def smap_daily(day: date, region) -> ee.Image:
    """Daily mean of the 3-hourly SMAP L4 surface and root-zone soil moisture."""
    start, end = day.isoformat(), (day + timedelta(days=1)).isoformat()
    col = ee.ImageCollection(SMAP).filterDate(start, end).filterBounds(region)
    return (col.select(['sm_surface', 'sm_rootzone']).mean().toFloat()
            .rename([f'sm_surface_{start}', f'sm_rootzone_{start}']))


def yearly_stack(region, year: int) -> ee.Image:
    days = [date(year, 1, 1) + timedelta(days=i)
            for i in range((date(year + 1, 1, 1) - date(year, 1, 1)).days)]
    return ee.Image.cat([ee.Image.cat([viirs_daily(d, region), smap_daily(d, region)])
                         for d in days]).toFloat()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grids', required=True, help='grid shapefile (EPSG:6933) with ISO3 and GID columns')
    p.add_argument('--bucket', required=True)
    p.add_argument('--project', required=True, help='Google Cloud project for Earth Engine')
    p.add_argument('--start', type=int, default=2016)
    p.add_argument('--end', type=int, default=2025)
    args = p.parse_args()

    ee.Initialize(project=args.project)
    grids = gpd.read_file(args.grids).to_crs(EASE2)

    for _, cell in grids.iterrows():
        coords = [list(xy) for xy in cell.geometry.exterior.coords]
        region = ee.Geometry.Polygon(coords, proj=EASE2, geodesic=False)
        for year in range(args.start, args.end + 1):
            name = f'{cell.ISO3}_G_{cell.GID}_viirs_{year}'
            ee.batch.Export.image.toCloudStorage(
                image=yearly_stack(region, year).clip(region),
                description=name,
                bucket=args.bucket,
                fileNamePrefix=f'patches/{name}',
                region=region,
                dimensions='32x32',
                crs=EASE2,
                maxPixels=1e13,
            ).start()
            print('submitted', name)


if __name__ == '__main__':
    main()
