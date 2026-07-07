import pandas as pd
import requests

# this can take a minute or two.
URI_ROOT = "https://spherex-utils-api-111857047506.us-central1.run.app"

def query_overlapping(ra_deg, dec_deg, side_deg, release="qr2", good_astrometry_only=False, frame="icrs",
                      uri_root=None):

    uri_root = URI_ROOT if uri_root is None else uri_root

    if good_astrometry_only:
        good_astrometry_only = "true"
    else:
        good_astrometry_only = "false"

    uri = f"{uri_root}/overlapping?lon={ra_deg}&lat={dec_deg}&side1={side_deg}&release={release}&good_astrometry_only={good_astrometry_only}&frame={frame}"

    j = requests.get(uri)

    df_query = pd.DataFrame(j.json())

    return df_query

# TODO: compare fresh query results against saved ones to find which weeks
# need reprocessing (see git history: check_weeks).

def main():
    from pathlib import Path

    root = "ngc628"
    ra_deg = 24.1770616
    dec_deg = 15.7869095
    side_deg = 0.9
    frame = "icrs"  # "galactic"
    df = query_overlapping(ra_deg, dec_deg, side_deg, frame=frame, release="qr2", good_astrometry_only=False)

    rootdir = Path(".") # f"/content/drive/MyDrive/spherex_examples/{root}")
    df.to_csv(rootdir / f"{root}.csv")


if __name__ == '__main__':
    main()
