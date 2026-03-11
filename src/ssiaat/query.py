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

def check_weeks(df_query):
    """This is to compare the query results from the saved one and check which weeks need to be reprocessed."""

    # FIXME still in progress.
    df_to_save = df_query[["filename", "OBSID"]].copy()
    df_to_save["weekname"] = df_to_save["OBSID"].str.split("_").str.get(0)
    df_to_save.to_csv(rootdir / "weekgroup.csv", index=False)

    new_record = df_to_save.set_index("filename")
    # Comparing a slice against the same slice to ensure identical labeling
    old_record = dff.iloc[:2400, :]
    # result = dff[~dff['filename'].isin(subset['filename'])]
    only_in_old = old_record[~old_record.index.isin(new_record.index)]
    only_in_new = new_record[~new_record.index.isin(old_record.index)]
    print("only in old: {} files, {} weeks".format(len(only_in_old),
                                                   len(only_in_old["weekname"].unique())))
    print("only in new: {} files, {} weeks".format(len(only_in_new),
                                                   len(only_in_new["weekname"].unique())))

    # print(result)

    weeks_to_process = set(only_in_new["weekname"].unique())
    weeks_to_process.update(only_in_old["weekname"].unique())
    weeks_to_process

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
