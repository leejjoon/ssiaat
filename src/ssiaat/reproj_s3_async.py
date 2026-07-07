"""Async bulk reprojection.

Fetches L2 files concurrently (S3, HTTP, local, ...) and reprojects each
onto a template WCS in a process pool: async workers keep the network
busy while the CPU-bound reproject_adaptive step scales across cores.

A failing file never aborts or hangs the run: failures are collected and
returned alongside the results (see AsyncCollector).
"""
import asyncio
import logging
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import fsspec
from fsspec.core import split_protocol

from .async_collector import AsyncCollector
from .reproj import get_df_from_buffer, get_metadata_from_filename

logger = logging.getLogger(__name__)


def _read_sync(fs, uri):
    with fs.open(uri, "rb") as f:
        return f.read()


async def async_read_file(uri, **storage_options):
    """Read the full content of a URI, asynchronously when the filesystem
    supports it (e.g. s3), via a worker thread otherwise (e.g. file://)."""
    protocol, _ = split_protocol(uri)
    protocol = protocol or "file"

    fs_cls = fsspec.get_filesystem_class(protocol)
    if fs_cls.async_impl:
        fs = fsspec.filesystem(protocol, asynchronous=True, **storage_options)
        async with await fs.open_async(uri, "rb") as f:
            return await f.read()

    fs = fsspec.filesystem(protocol, **storage_options)
    return await asyncio.to_thread(_read_sync, fs, uri)


def _process_buffer(buffer, wcs_tmpl, aux_metadata, zodi_corrector):
    # Module-level so it can be pickled into the process pool.
    return get_df_from_buffer(buffer, wcs_tmpl, aux_metadata=aux_metadata,
                              zodi_corrector=zodi_corrector)


def _get_mp_context():
    # forkserver avoids both fork-with-threads hazards (we run under an
    # event loop) and spawn's per-worker import cost; fall back to spawn
    # where forkserver is unavailable.
    try:
        return multiprocessing.get_context("forkserver")
    except ValueError:
        return multiprocessing.get_context("spawn")


async def run_reproj_tasks(uri_list, wcs_tmpl, *, num_fetchers=4,
                           num_workers=None, progress=True,
                           storage_options=None, zodi_corrector=None,
                           num_tasks=None):
    """Fetch and reproject every URI onto wcs_tmpl.

    Fetching runs on `num_fetchers` async workers; the CPU-bound
    reprojection of each exposure runs in a `num_workers`-process pool,
    so the run scales with cores instead of serializing on the GIL.

    Parameters
    ----------
    uri_list : iterable of str
        Full URIs (s3://..., https://..., file://..., or plain local
        paths). None entries are skipped.
    num_fetchers : int
        Concurrent downloads. (`num_tasks` is a deprecated alias.)
    num_workers : int, optional
        Reprojection worker processes; defaults to cpu_count() - 1.
        0 disables the pool and processes inline on the event-loop
        thread (no pickling constraints; useful for debugging).
    storage_options : dict, optional
        Passed to fsspec. Defaults to {"anon": True} for s3:// URIs when
        not given (public-bucket convention, same as the finder).
    zodi_corrector : callable, optional
        Maps the ZODI array to a corrected one before subtraction.
        With num_workers > 0 it must be picklable: use e.g.
        ``functools.partial(ZodiCorrection().get_corrected_zodi, band)``,
        not a lambda or local closure.

    Returns
    -------
    (dfl, failures)
        dfl: list of DataFrames (exposures with an empty template
        footprint are silently skipped). failures: list of
        (uri, exception) pairs; the run itself never raises for
        per-item errors.
    """
    if num_tasks is not None:
        warnings.warn("num_tasks is deprecated; use num_fetchers",
                      DeprecationWarning, stacklevel=2)
        num_fetchers = num_tasks
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    uris = [uri for uri in uri_list if uri is not None]

    async def fetch(uri):
        options = storage_options
        if options is None and uri.startswith("s3://"):
            options = {"anon": True}
        buffer = await async_read_file(uri, **(options or {}))
        return buffer, get_metadata_from_filename(uri)

    if num_workers == 0:
        async def process(uri):
            buffer, aux_metadata = await fetch(uri)
            return _process_buffer(buffer, wcs_tmpl, aux_metadata,
                                   zodi_corrector)

        collector = AsyncCollector(uris, process, total=len(uris))
        results = await collector.run(num_tasks=num_fetchers,
                                      progress=progress)
    else:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=num_workers,
                                 mp_context=_get_mp_context()) as pool:
            async def process(uri):
                buffer, aux_metadata = await fetch(uri)
                return await loop.run_in_executor(
                    pool, _process_buffer, buffer, wcs_tmpl, aux_metadata,
                    zodi_corrector)

            collector = AsyncCollector(uris, process, total=len(uris))
            results = await collector.run(num_tasks=num_fetchers,
                                          progress=progress)

    failures = collector.failures

    logger.info("reprojection run finished: %d succeeded, %d failed",
                len(results), len(failures))
    for uri, exc in failures:
        logger.warning("reprojection failed for %s: %r", uri, exc)

    return results, failures
