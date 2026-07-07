"""ssiaat -- SPHEREx Spectral Image As A Table.

Importing this package registers the ``.spectral`` (DataFrame) and
``.itable`` (Series) pandas accessors.
"""
from .spherex_table import (
    read_stable,
    promote_to_stable,
    SsiaatConverter,
    Image,
    Model,
    FitResults,
)
from .model import sed
from .reproj import (
    SphxReprojector,
    get_df_from_uri,
    merge_to_stable,
    ingest_hdul,
)
from .reproj_s3_async import run_reproj_tasks
from .query import query_overlapping
from .wcs_helper import get_wcs, get_wcs_from_shape, TemplateHeaderCards
from .indexing import make_pixel_index, get_src_yx
from .finder import find_latest_uri, find_latest_uri_async, check_uri
from .async_collector import AsyncCollector

__all__ = [
    "read_stable",
    "promote_to_stable",
    "SsiaatConverter",
    "Image",
    "Model",
    "FitResults",
    "sed",
    "SphxReprojector",
    "get_df_from_uri",
    "merge_to_stable",
    "ingest_hdul",
    "run_reproj_tasks",
    "query_overlapping",
    "get_wcs",
    "get_wcs_from_shape",
    "TemplateHeaderCards",
    "make_pixel_index",
    "get_src_yx",
    "find_latest_uri",
    "find_latest_uri_async",
    "check_uri",
    "AsyncCollector",
]
