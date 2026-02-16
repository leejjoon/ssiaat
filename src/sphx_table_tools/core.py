import pandas as pd
from urllib.parse import urlparse
import os
import asyncio
import re
from tqdm.asyncio import tqdm
import fsspec
from fsspec.core import url_to_fs
import aiohttp

# Default concurrency limit
MAX_CONCURRENT_TASKS = 20

def _get_table_from_filenames(filenames):
    """Parse filenames into a DataFrame with plan, pointing, step, band, and pipeline_run components."""
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

async def _find_latest_single_async(row, fs, root_path, release, semaphore, http_session=None):
    """Asynchronously find the latest URI for a single row using fsspec/aiohttp."""
    async with semaphore:
        # Use plan from the row (parsed directly from filename)
        plan_dir = f"{root_path.rstrip('/')}/{release}/level2/{row['plan']}/"
        
        try:
            protocols = fs.protocol if isinstance(fs.protocol, (list, tuple)) else [fs.protocol]
            if any(p in ['http', 'https'] for p in protocols):
                async with http_session.get(plan_dir) as response:
                    if response.status != 200:
                        return None
                    text = await response.text()
                
                pipe_vers = set()
                hrefs = re.findall(r"href=['\"]?([^'\"\s>]+)['\"]?", text)
                for href in hrefs:
                    name = href.rstrip('/').split('/')[-1]
                    if name.startswith('l2b-v'):
                        pipe_vers.add(name)
                names = re.findall(r'(l2b-v[\w-]+)', text)
                for name in names:
                    pipe_vers.add(name)
                pipe_vers = sorted(list(pipe_vers), reverse=True)
            else:
                if hasattr(fs, '_ls'):
                    items = await fs._ls(plan_dir, detail=False)
                else:
                    items = fs.ls(plan_dir, detail=False)
                
                pipe_vers = []
                for item in items:
                    name = item.rstrip('/').split('/')[-1]
                    if name.startswith('l2b-v'):
                        pipe_vers.append(name)
                pipe_vers.sort(reverse=True)
            for pipe_ver in pipe_vers:
                # Based on the confirmed structure: {plan}/{pipe_ver}/{band}/{filename}
                # Use the pipe_ver to reconstruct the filename
                base_filename = row['filename'].rsplit('_', 1)[0]
                new_filename = f"{base_filename}_{pipe_ver}.fits"
                file_path = f"{plan_dir}{pipe_ver}/{row['band']}/{new_filename}"
                if hasattr(fs, '_exists'):
                    exists = await fs._exists(file_path)
                else:
                    exists = fs.exists(file_path)
                
                if exists:
                    return file_path
        except Exception:
            raise
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
    
    async with aiohttp.ClientSession() as http_session:
        tasks = [
            _find_latest_single_async(row, fs, root_path, release, semaphore, http_session=http_session) 
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
    """Synchronous wrapper for find_latest_uri_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(find_latest_uri_async(filenames, root_uri, release, progress, max_concurrency, storage_options))

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
