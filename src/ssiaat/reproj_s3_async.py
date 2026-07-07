"""Async bulk reprojection.

Fetches L2 files concurrently (S3, HTTP, local, ...) and reprojects each
onto a template WCS. Concurrency helps the fetch; the reprojection itself
is CPU-bound and still runs on the event-loop thread (process-based
parallelism is planned separately).

A failing file never aborts or hangs the run: failures are collected and
returned alongside the results (see AsyncCollector).
"""
import asyncio
import logging

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


async def run_reproj_tasks(uri_list, wcs_tmpl, *, num_tasks=4, progress=True,
                           storage_options=None, zodi_corrector=None):
    """Fetch and reproject every URI onto wcs_tmpl.

    Parameters
    ----------
    uri_list : iterable of str
        Full URIs (s3://..., https://..., file://..., or plain local
        paths). None entries are skipped.
    storage_options : dict, optional
        Passed to fsspec. Defaults to {"anon": True} for s3:// URIs when
        not given (public-bucket convention, same as the finder).
    zodi_corrector : callable, optional
        Maps the ZODI array to a corrected one before subtraction.

    Returns
    -------
    (dfl, failures)
        dfl: list of DataFrames (exposures with an empty template
        footprint are silently skipped). failures: list of
        (uri, exception) pairs; the run itself never raises for
        per-item errors.
    """
    uris = [uri for uri in uri_list if uri is not None]

    async def process(uri):
        options = storage_options
        if options is None and uri.startswith("s3://"):
            options = {"anon": True}
        buffer = await async_read_file(uri, **(options or {}))
        aux_metadata = get_metadata_from_filename(uri)
        return get_df_from_buffer(buffer, wcs_tmpl, aux_metadata=aux_metadata,
                                  zodi_corrector=zodi_corrector)

    collector = AsyncCollector(uris, process, total=len(uris))
    results = await collector.run(num_tasks=num_tasks, progress=progress)
    failures = collector.failures

    logger.info("reprojection run finished: %d succeeded, %d failed",
                len(results), len(failures))
    for uri, exc in failures:
        logger.warning("reprojection failed for %s: %r", uri, exc)

    return results, failures
