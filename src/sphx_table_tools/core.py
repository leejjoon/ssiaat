import pandas as pd
from urllib.parse import urlparse
import os
import boto3
import asyncio
import aiohttp
import aioboto3
import requests
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config
import re
from tqdm.asyncio import tqdm

# Default concurrency limit
MAX_CONCURRENT_TASKS = 20

def _get_table_from_filenames(filenames):
    """Parse filenames into a DataFrame with plan, band, and root components."""
    unique_filenames = pd.Series(filenames)
    _root = unique_filenames.str.split("_spx_").str[0]
    split = _root.str.split("_")
    plan = split.apply(lambda s: f"{s[1]}_{s[2]}")
    band = split.apply(lambda s: s[4][-1])
    root = split.apply(lambda s: f"{s[3]}_{s[4]}")
    return pd.DataFrame(dict(filename=unique_filenames, plan=plan, band=band, root=root))

async def _find_latest_s3_single_async(row, bucket, release, session, semaphore):
    """Asynchronously find the latest S3 URI for a single row."""
    async with semaphore:
        prefix = f"{release}/level2/{row['plan']}/"
        async with session.client("s3", config=Config(signature_version=UNSIGNED)) as s3:
            try:
                response = await s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
            except Exception:
                return None

            if "CommonPrefixes" not in response:
                return None

            pipe_vers = sorted(
                [p["Prefix"].split("/")[-2] for p in response["CommonPrefixes"]],
                reverse=True,
            )

            for pipe_ver in pipe_vers:
                file_key = f"{release}/level2/{row['plan']}/{pipe_ver}/{row['band']}/level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits"
                try:
                    await s3.head_object(Bucket=bucket, Key=file_key)
                    return f"s3://{bucket}/{file_key}"
                except Exception:
                    continue
    return None

async def _find_latest_http_single_async(row, root_url, release, session, semaphore):
    """Asynchronously find the latest HTTP URI for a single row."""
    async with semaphore:
        plan_url = f"{root_url.rstrip('/')}/{release}/level2/{row['plan']}/"
        try:
            async with session.get(plan_url) as response:
                if response.status != 200:
                    return None
                text = await response.text()
        except Exception:
            return None

        pipe_vers = sorted(re.findall(r'href="([^/]+)/"', text), reverse=True)

        for pipe_ver in pipe_vers:
            file_url = f"{plan_url}{pipe_ver}/{row['band']}/level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits"
            try:
                async with session.head(file_url, allow_redirects=True) as head_resp:
                    if head_resp.status == 200:
                        return file_url
            except Exception:
                pass
    return None

def _find_latest_local_single(row, root_path, release):
    """Find the latest local path for a single row."""
    plan_path = os.path.join(root_path, release, "level2", row["plan"])
    if not os.path.isdir(plan_path):
        return None
    pipe_vers = sorted(
        [d for d in os.listdir(plan_path) if os.path.isdir(os.path.join(plan_path, d))],
        reverse=True,
    )
    for pipe_ver in pipe_vers:
        file_path = os.path.join(plan_path, pipe_ver, row["band"], f"level2_{row['plan']}_{row['root']}_spx_{pipe_ver}.fits")
        if os.path.exists(file_path):
            return file_path
    return None

async def find_latest_uri_async(filenames, root_uri, release="qr2", progress: bool = False, max_concurrency: int = MAX_CONCURRENT_TASKS):
    """Async engine to find latest URIs."""
    df = _get_table_from_filenames(filenames)
    parsed_root = urlparse(root_uri)
    semaphore = asyncio.Semaphore(max_concurrency)

    if parsed_root.scheme in ["", "file"]:
        # Local is fast enough to stay synchronous or use threads, but we'll just run it
        return df.apply(lambda row: _find_latest_local_single(row, parsed_root.path, release), axis=1)

    tasks = []
    if parsed_root.scheme == "s3":
        session = aioboto3.Session()
        async with session.client("s3", config=Config(signature_version=UNSIGNED)) as _: # Just to ensure session is usable
            tasks = [
                _find_latest_s3_single_async(row, parsed_root.netloc, release, session, semaphore)
                for _, row in df.iterrows()
            ]
            if progress:
                results = await tqdm.gather(*tasks, desc="Checking S3")
            else:
                results = await asyncio.gather(*tasks)
    
    elif parsed_root.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                _find_latest_http_single_async(row, root_uri, release, session, semaphore)
                for _, row in df.iterrows()
            ]
            if progress:
                results = await tqdm.gather(*tasks, desc="Checking HTTP")
            else:
                results = await asyncio.gather(*tasks)
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed_root.scheme}")

    return pd.Series(results, index=df.index)

def find_latest_uri(filenames, root_uri, release="qr2", progress: bool = False, max_concurrency: int = MAX_CONCURRENT_TASKS):
    """Synchronous wrapper for find_latest_uri_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(find_latest_uri_async(filenames, root_uri, release, progress, max_concurrency))

def check_uri(df: pd.DataFrame, root_uri: str, progress: bool = False) -> pd.Series:
    """Check existence of URIs (kept simple for now, could also be asyncified)."""
    # ... (omitted for brevity in this step, keeping previous logic but could asyncify if requested)
    parsed_root = urlparse(root_uri)
    s3_client = None
    if parsed_root.scheme == "s3":
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def _check_single(uri, parsed_root, s3_client):
        if parsed_root.scheme in ["", "file"]:
            path = os.path.join(parsed_root.path, uri)
            return path if os.path.exists(path) else None
        elif parsed_root.scheme == "s3":
            key = os.path.join(parsed_root.path.lstrip('/'), uri)
            try:
                s3_client.head_object(Bucket=parsed_root.netloc, Key=key)
                return f"s3://{parsed_root.netloc}/{key}"
            except Exception: return None
        elif parsed_root.scheme in ["http", "https"]:
            url = f"{root_uri.rstrip('/')}/{uri}"
            try:
                if requests.head(url, allow_redirects=True).status_code == 200: return url
            except Exception: pass
            return None
        return None

    if progress:
        tqdm.pandas(desc="Checking URIs")
        return df["uri"].progress_apply(lambda x: _check_single(x, parsed_root, s3_client))
    return df["uri"].apply(lambda x: _check_single(x, parsed_root, s3_client))
