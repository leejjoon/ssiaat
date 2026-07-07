"""Query the exposures overlapping NGC 628 and save the result as CSV.

Moved out of ssiaat.query.
"""
from pathlib import Path

from ssiaat.query import query_overlapping


def main():
    root = "ngc628"
    ra_deg = 24.1770616
    dec_deg = 15.7869095
    side_deg = 0.9
    frame = "icrs"  # "galactic"
    df = query_overlapping(ra_deg, dec_deg, side_deg, frame=frame,
                           release="qr2", good_astrometry_only=False)

    rootdir = Path(".")
    df.to_csv(rootdir / f"{root}.csv")


if __name__ == '__main__':
    main()
