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
    "ssiaat.model",
    "ssiaat.model.fitting",
    "ssiaat.model.sed",
    "ssiaat.model.vectorized_lstsq",
]


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports(module):
    importlib.import_module(module)


def test_ingest_hdul_shared_not_duplicated():
    from ssiaat import reproj, reproj_hips
    assert reproj_hips.ingest_hdul is reproj.ingest_hdul


def test_public_api_names_resolve():
    import ssiaat
    for name in ssiaat.__all__:
        assert getattr(ssiaat, name) is not None


def test_bare_import_registers_accessors():
    # `import ssiaat` alone must be enough for df.spectral / s.itable.
    import subprocess, sys
    code = (
        "import ssiaat, pandas as pd\n"
        "df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[0, 0, 1])\n"
        "df.spectral\n"
        "s = pd.Series([1.0], index=[0])\n"
        "s.itable\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
