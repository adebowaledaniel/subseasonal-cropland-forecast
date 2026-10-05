"""
Growing-season months per cell from the GEOGLAM Crop Monitor sub-national crop calendars.

The attribute table gives the planting and harvest days of year of the dominant crop of
the sub-national unit under each cell. The growing season is every month from planting to
harvest, wrapping across the year end.

    python preprocessing/crop_calendar.py --attributes data/grid_attributes.csv \
        --output configs/growing_season.yml
"""

import argparse
import csv
from datetime import date, timedelta


def month_of_day(day: int) -> int:
    return (date(2023, 1, 1) + timedelta(days=max(int(day), 1) - 1)).month


def season_months(start_day: int, end_day: int):
    start, end = month_of_day(start_day), month_of_day(end_day)
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 13)) + list(range(1, end + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--attributes', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    with open(args.attributes) as f:
        rows = list(csv.DictReader(f))
    with open(args.output, 'w') as f:
        f.write('# Growing-season months (planting to harvest) of the dominant crop per location,\n'
                '# from the GEOGLAM Crop Monitor sub-national crop calendars.\n')
        for row in rows:
            f.write(f"{row['Name']}: {season_months(row['planting'], row['harvest'])}\n")


if __name__ == '__main__':
    main()
