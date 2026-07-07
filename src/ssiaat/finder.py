"""
Finder utilities for SPHEREx table tools.

This module provides functionality to find the latest version of data files 
within a structured storage system (S3, HTTP, or local). 

The logic for finding the latest file follows these steps:
1. Parse input filenames to extract components like 'plan', 'pointing', and 'band'.
2. List directories in the storage root (e.g., '{root}/{release}/level2/{plan}/') 
   to identify available pipeline versions ('l2b-v*').
3. Sort these versions in descending order (latest first).
4. For each version, reconstruct the expected filename and check for its existence.
5. Return the URI of the first (latest) existing file found.
"""
import pandas as pd
from urllib.parse import urlparse
import logging
import os
import asyncio
import re
from tqdm.asyncio import tqdm
import fsspec
from fsspec.core import url_to_fs
import aiohttp
from .fs import register_fs

# Register custom filesystems
register_fs()

logger = logging.getLogger(__name__)

# Default concurrency limit
MAX_CONCURRENT_TASKS = 20

_PIPE_VER_RE = re.compile(r"l2b-v(\d+)-(\d+)-(\d+)$")

def _parse_pipe_version(name):
    """'l2b-v20-2025-335' -> (20, 2025, 335); None if not a pipeline version.

    Versions must be compared numerically: as strings, 'l2b-v9-...' sorts
    after 'l2b-v20-...' and "find latest" silently returns old data.
    """
    m = _PIPE_VER_RE.match(name)
    if m is None:
        return None
    return tuple(int(g) for g in m.groups())

def _get_table_from_filenames(filenames):
    """
    Parse filenames into a DataFrame with plan, pointing, step, band, and pipeline_run components.

    Example filename: level2_2025W48_1A_0516_2D6_spx_l2b-v20-2025-335.fits
    Parsing logic:
    - plan: '2025W48_1A' (from indices 1 and 2)
    - pointing: '0516' (from index 3)
    - step: '2' (first char of index 4 '2D6')
    - band: '6' (last char of index 4 '2D6')
    - pipeline_run: 'l2b-v20-2025-335' (from index 6)
    """
    unique_filenames = pd.Series(filenames)
    # Example: level2_2025W48_1A_0516_2D6_spx_l2b-v20-2025-335.fits
    _root = unique_filenames.str.split(".fits").str[0]
    split = _root.str.split("_")
    
    plan = split.apply(lambda s: f"{s[1]}_{s[2]}")
    pointing = split.apply(lambda s: s[3])
    # split[4] is e.g. '2D6'
    step = split.apply(lambda s: s[4][0])
    band = split.apply(lambda s: s[4][-1])
    pipeline_run = split.apply(lambda s: s[6])
    
    return pd.DataFrame(dict(
        filename=unique_filenames, 
        plan=plan, 
        pointing=pointing, 
        step=step, 
        band=band, 
        pipeline_run=pipeline_run
    ))


from pathlib import Path

# def get_readpath(rootdir, plan, band, root):

def find_local_uri(fn, release="qr2", rootdir=None):

    # path in olaf
    # "/proj/internal_group/spherex/Shared/prod-qr2/repo//level2"
    # release = "qr2"

    if rootdir is None:
        rootdir = Path("/proj/internal_group/spherex/Shared/") / f"prod-{release}" / "repo"
    else:
        rootdir = Path(rootdir)

    # FIXME we should have a simpler way
    k = _get_table_from_filenames([fn]).iloc[0]
    plan = k["plan"]
    band = k["band"]
    root = "{}_{}D{}".format(k["pointing"], k["step"], k["band"])

    pipe_version_candidates = sorted(
        (p.name for p in (rootdir / f"level2/{plan}").iterdir()
         if _parse_pipe_version(p.name) is not None),
        key=_parse_pipe_version, reverse=True)

    for pipe_ver in pipe_version_candidates:
        file_key = f"level2/{plan}/{pipe_ver}/{band}/level2_{plan}_{root}_spx_{pipe_ver}.fits"
        cand = rootdir / file_key
        if cand.exists():
            return cand

    return None

async def _find_latest_single_async(row, fs, root_path, release, semaphore):
    """Asynchronously find the latest URI for a single row using fsspec."""
    async with semaphore:
        # Use plan from the row (parsed directly from filename)
        plan_dir = f"{root_path.rstrip('/')}/{release}/level2/{row['plan']}/"
        
        try:
            if hasattr(fs, '_ls'):
                items = await fs._ls(plan_dir, detail=False)
            else:
                items = await asyncio.to_thread(fs.ls, plan_dir, detail=False)
            
            pipe_vers = []
            for item in items:
                name = item.rstrip('/').split('/')[-1]
                if _parse_pipe_version(name) is not None:
                    pipe_vers.append(name)
            pipe_vers.sort(key=_parse_pipe_version, reverse=True)

            for pipe_ver in pipe_vers:
                # Based on the confirmed structure: {plan}/{pipe_ver}/{band}/{filename}
                # Use the pipe_ver to reconstruct the filename
                base_filename = row['filename'].rsplit('_', 1)[0]
                new_filename = f"{base_filename}_{pipe_ver}.fits"
                file_path = f"{plan_dir}{pipe_ver}/{row['band']}/{new_filename}"
                
                if hasattr(fs, '_exists'):
                    exists = await fs._exists(file_path)
                else:
                    exists = await asyncio.to_thread(fs.exists, file_path)
                
                if exists:
                    return file_path
        except FileNotFoundError:
            # plan directory absent: genuinely "not found", stay quiet
            return None
        except Exception as e:
            # credentials, bad root URI, network, ... -- do not let these
            # masquerade as "file not found"
            logger.warning("finder failed for %s: %r", row["filename"], e)
            return None
    return None


async def find_latest_uri_async(filenames, root_uri, release="qr2", progress: bool = False, 
                                max_concurrency: int = MAX_CONCURRENT_TASKS, 
                                storage_options: dict = None):
    """Async engine to find latest URIs using fsspec."""
    df = _get_table_from_filenames(filenames)
    if storage_options is None and root_uri.startswith("s3://"):
        storage_options = {"anon": True}
    fs, root_path = url_to_fs(root_uri, asynchronous=True, **(storage_options or {}))
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    tasks = [
        _find_latest_single_async(row, fs, root_path, release, semaphore) 
        for _, row in df.iterrows()
    ]
    if progress:
        results = await tqdm.gather(*tasks, desc=f"Checking {fs.protocol}")
    else:
        results = await asyncio.gather(*tasks)
            
    return pd.Series(results, index=df.index)


def find_latest_uri(filenames, root_uri, release="qr2", progress: bool = False,
                    max_concurrency: int = MAX_CONCURRENT_TASKS,
                    storage_options: dict = None):
    """Synchronous wrapper for find_latest_uri_async.

    Only usable where no event loop is running. Inside Jupyter/IPython
    (which runs its own loop) call the async variant directly instead:

        result = await find_latest_uri_async(filenames, root_uri, ...)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(find_latest_uri_async(
            filenames, root_uri, release, progress, max_concurrency,
            storage_options))
    raise RuntimeError(
        "find_latest_uri() cannot be used while an event loop is running"
        " (e.g. inside Jupyter/IPython)."
        " Use: await find_latest_uri_async(...)")


async def get_uri_updated_dataframe(df_query, release, root_uri=None):
  if root_uri is None:
    root_uri = "s3://nasa-irsa-spherex"

  filenames = df_query["filename"].unique()

  latest_uris = await find_latest_uri_async(filenames, root_uri,
                                            release=release,
                                            progress=True,
                                            max_concurrency=30)

  query_results = pd.DataFrame({
              'filename': filenames,
              'uri_decorated': latest_uris
  })
  df = query_results.merge(df_query.set_index("filename")["DETECTOR"], how="inner", left_on="filename", right_index=True)

  return df, query_results


def check_uri(df: pd.DataFrame, root_uri: str, progress: bool = False, storage_options: dict = None) -> pd.Series:
    """Check existence of URIs using fsspec."""
    if storage_options is None and root_uri.startswith("s3://"):
        storage_options = {"anon": True}
    fs, root_path = url_to_fs(root_uri, **(storage_options or {}))
    def _check_single(uri):
        full_path = f"{root_path.rstrip('/')}/{uri}"
        if fs.exists(full_path): return full_path
        return None
    if progress:
        from tqdm import tqdm
        tqdm.pandas(desc="Checking URIs")
        return df["uri"].progress_apply(_check_single)
    return df["uri"].apply(_check_single)


def test_s3():

    async def find_files(root_uri, filenames, release="qr2", progress=True):

        # using the async implementation behind the scenes
        latest_uris = await find_latest_uri_async(filenames, root_uri, release=release, progress=progress, max_concurrency=30)

        results = pd.DataFrame({
            'filename': filenames,
            'uri_decorated': latest_uris
        })

        return results


    root_uri = "s3://nasa-irsa-spherex"
    # #root_uri = "webfsd://100.103.128.7:3000"

    filenames = ['level2_2025W24_1A_0405_1D1_spx_l2b-v19-2025-252.fits',
                 'level2_2025W24_1A_0405_2D1_spx_l2b-v19-2025-252.fits',
                 'level2_2025W24_1A_0405_3D1_spx_l2b-v19-2025-252.fits',
                 ]

    import asyncio
    results = asyncio.run(find_files(root_uri, filenames, release="qr2"))
    print(len(results))


if __name__ == '__main__':
    test_s3()
