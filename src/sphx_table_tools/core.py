import pandas as pd
from urllib.parse import urlparse
import os
import boto3
import requests
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config
import re

def _get_table_from_filenames(filenames):
    # filenames should be a
    unique_filenames = pd.Series(filenames)

    _root = unique_filenames.str.split("_spx_").str[0]

    split = _root.str.split("_")
    plan = split.apply(lambda s: f"{s[1]}_{s[2]}")
    band = split.apply(lambda s: s[4][-1])
    root = split.apply(lambda s: f"{s[3]}_{s[4]}")

    df = pd.DataFrame(dict(filename=unique_filenames, plan=plan, band=band, root=root))

    return df

def _find_latest_s3(df, bucket, release):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    results = []
    for _, row in df.iterrows():
        prefix_for_listing_pipe_vers = f"{release}/level2/{row['plan']}/"
        try:
            response = s3.list_objects_v2(
                Bucket=bucket, Prefix=prefix_for_listing_pipe_vers, Delimiter="/"
            )
        except ClientError as e:
            print(f"Error listing objects for {row['plan']}/{row['band']}/{row['root']}: {e}")
            results.append(None)
            continue

        if "CommonPrefixes" not in response:
            results.append(None)
            continue

        pipe_vers = sorted(
            [p["Prefix"].split("/")[-2] for p in response["CommonPrefixes"]],
            reverse=True,
        )

        found_uri = None
        for pipe_ver in pipe_vers:
            file_key = f"{release}/level2/{row['plan']}/{pipe_ver}/{row['band']}/level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits"
            try:
                s3.head_object(Bucket=bucket, Key=file_key)
                found_uri = f"s3://{bucket}/{file_key}"
                break
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    continue
                else:
                    pass
        results.append(found_uri)
    return pd.Series(results, index=df.index)

def _find_latest_local(df, root_path, release):
    results = []
    for _, row in df.iterrows():
        plan_path = os.path.join(root_path, release, "level2", row["plan"])
        if not os.path.isdir(plan_path):
            results.append(None)
            continue

        pipe_vers = sorted(
            [d for d in os.listdir(plan_path) if os.path.isdir(os.path.join(plan_path, d))],
            reverse=True,
        )

        found_path = None
        for pipe_ver in pipe_vers:
            file_path = os.path.join(
                plan_path,
                pipe_ver,
                row["band"],
                f"level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits",
            )
            if os.path.exists(file_path):
                found_path = file_path
                break
        results.append(found_path)
    return pd.Series(results, index=df.index)

def _find_latest_http(df, root_url, release):
    results = []
    for _, row in df.iterrows():
        plan_url = f"{root_url.rstrip('/')}/{release}/level2/{row['plan']}/"
        try:
            response = requests.get(plan_url)
            response.raise_for_status()
        except requests.RequestException:
            results.append(None)
            continue

        pipe_vers = sorted(
            re.findall(r'href="([^/]+)/"', response.text), reverse=True
        )

        found_url = None
        for pipe_ver in pipe_vers:
            file_url = f"{plan_url}{pipe_ver}/{row['band']}/level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits"
            try:
                head_response = requests.head(file_url, allow_redirects=True)
                if head_response.status_code == 200:
                    found_url = file_url
                    break
            except requests.RequestException:
                pass
        results.append(found_url)
    return pd.Series(results, index=df.index)


def find_latest_uri(filenames, root_uri, release="qr2"):
    """
    Find the latest file based on the pipe_ver for different backends.
    """
    df = _get_table_from_filenames(filenames)
    parsed_root = urlparse(root_uri)

    if parsed_root.scheme in ["", "file"]:
        return _find_latest_local(df, parsed_root.path, release)
    elif parsed_root.scheme == "s3":
        bucket = parsed_root.netloc
        return _find_latest_s3(df, bucket, release)
    elif parsed_root.scheme in ["http", "https"]:
        return _find_latest_http(df, root_uri, release)
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed_root.scheme}")


def check_uri(df: pd.DataFrame, root_uri: str) -> pd.Series:
    """
    Given a pandas dataframe with a "uri" column, check if each uri
    does exist under the root uri.

    :param df: DataFrame with a "uri" column.
    :param root_uri: The root URI to check against (local path, s3://, or http(s)://).
    :return: A pandas Series with the actual path, None if it fails to find a path.
    """

    def check_individual_uri(uri, parsed_root):
        if parsed_root.scheme in ["", "file"]:
            # Local file path
            path = os.path.join(parsed_root.path, uri)
            return path if os.path.exists(path) else None
        elif parsed_root.scheme == "s3":
            # S3 path
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            bucket = parsed_root.netloc
            key = os.path.join(parsed_root.path.lstrip('/'), uri)
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return f"s3://{bucket}/{key}"
            except Exception:
                return None
        elif parsed_root.scheme in ["http", "httpshttps"]:
            # HTTP/HTTPS path
            url = f"{root_uri.rstrip('/')}/{uri}"
            try:
                response = requests.head(url, allow_redirects=True)
                if response.status_code == 200:
                    return url
            except requests.RequestException:
                pass
            return None
        else:
            raise ValueError(f"Unsupported URI scheme: {parsed_root.scheme}")

    parsed_root = urlparse(root_uri)
    return df["uri"].apply(lambda uri: check_individual_uri(uri, parsed_root))
