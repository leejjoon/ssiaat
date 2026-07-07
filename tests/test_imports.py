"""Every submodule must at least import (reproj_hips historically could not)."""
import importlib

import pytest

SUBMODULES = [
    "ssiaat",
    "ssiaat.async_collector",
    "ssiaat.finder",
    "ssiaat.flags",
    "ssiaat.fs",
    "ssiaat.query",
    "ssiaat.reproj",
    "ssiaat.reproj_hips",
    "ssiaat.reproj_s3_async",
    "ssiaat.spherex_table",
    "ssiaat.tabular_bandpass_lite",
    "ssiaat.wcs_helper",
    "ssiaat.zodi_correction",
    "ssiaat.model.sed",
    "ssiaat.model.vectorized_lstsq",
]


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports(module):
    importlib.import_module(module)


def test_ingest_hdul_shared_not_duplicated():
    from ssiaat import reproj, reproj_hips
    assert reproj_hips.ingest_hdul is reproj.ingest_hdul
