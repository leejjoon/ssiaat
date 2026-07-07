"""finder: filename parsing and latest-pipeline-version selection.

The version-selection tests run against a local fixture tree via fsspec's
file:// protocol — no S3/network needed.
"""
import asyncio
import logging

import pandas as pd
import pytest

from ssiaat.finder import (
    _find_latest_single_async,
    _get_table_from_filenames,
    _parse_pipe_version,
    find_latest_uri,
    find_local_uri,
)

FN = "level2_2025W48_1A_0516_2D6_spx_l2b-v19-2025-252.fits"
BASE = "level2_2025W48_1A_0516_2D6_spx"
PLAN = "2025W48_1A"
BAND = "6"

# Deliberately mixes single- and double-digit versions: lexically
# 'l2b-v9-...' sorts after 'l2b-v20-...', which is the bug under test.
VERSIONS = ["l2b-v9-2025-100", "l2b-v19-2025-252", "l2b-v20-2025-335"]
LATEST = "l2b-v20-2025-335"


def test_parse_pipe_version():
    assert _parse_pipe_version("l2b-v20-2025-335") == (20, 2025, 335)
    assert _parse_pipe_version("l2b-v9-2025-100") == (9, 2025, 100)
    assert _parse_pipe_version("not-a-version") is None
    assert _parse_pipe_version("l2b-v20-2025-335.bak") is None


def test_parse_pipe_version_orders_numerically():
    assert (_parse_pipe_version("l2b-v9-2025-100")
            < _parse_pipe_version("l2b-v20-2025-335"))
    latest = max(VERSIONS, key=_parse_pipe_version)
    assert latest == LATEST


def test_get_table_from_filenames():
    fn = "level2_2025W48_1A_0516_2D6_spx_l2b-v20-2025-335.fits"
    row = _get_table_from_filenames([fn]).iloc[0]
    assert row["filename"] == fn
    assert row["plan"] == "2025W48_1A"
    assert row["pointing"] == "0516"
    assert row["step"] == "2"
    assert row["band"] == "6"
    assert row["pipeline_run"] == "l2b-v20-2025-335"


def _make_tree(root, versions, with_file=None, release="qr2"):
    """Build {root}/{release}/level2/{plan}/{ver}/{band}/{file} for each
    version; the file itself is created only for versions in `with_file`
    (all versions if None)."""
    with_file = versions if with_file is None else with_file
    for ver in versions:
        d = root / release / "level2" / PLAN / ver / BAND
        d.mkdir(parents=True)
        if ver in with_file:
            (d / f"{BASE}_{ver}.fits").touch()


def test_find_latest_uri_prefers_numerically_latest(tmp_path):
    _make_tree(tmp_path, VERSIONS)
    result = find_latest_uri([FN], f"file://{tmp_path}")
    assert result.iloc[0].endswith(f"{BASE}_{LATEST}.fits")


def test_find_latest_uri_falls_back_to_existing_file(tmp_path):
    # v20 directory exists but holds no file; v19 is the latest available.
    _make_tree(tmp_path, VERSIONS, with_file=["l2b-v9-2025-100",
                                              "l2b-v19-2025-252"])
    result = find_latest_uri([FN], f"file://{tmp_path}")
    assert result.iloc[0].endswith(f"{BASE}_l2b-v19-2025-252.fits")


def test_find_latest_uri_missing_plan_gives_none(tmp_path, caplog):
    (tmp_path / "qr2" / "level2").mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger="ssiaat.finder"):
        result = find_latest_uri([FN], f"file://{tmp_path}")
    assert result.iloc[0] is None
    # a genuinely missing plan is "not found", not an error worth warning
    assert not caplog.records


def test_finder_warns_on_unexpected_errors(caplog):
    # Errors other than "not found" (credentials, bad root, network) must
    # not silently collapse into None.
    class BrokenFS:
        async def _ls(self, path, detail=False):
            raise RuntimeError("simulated credentials error")

    row = {"filename": FN, "plan": PLAN, "band": BAND}

    async def run():
        return await _find_latest_single_async(
            row, BrokenFS(), "/root", "qr2", asyncio.Semaphore(1))

    with caplog.at_level(logging.WARNING, logger="ssiaat.finder"):
        result = asyncio.run(run())

    assert result is None
    assert any(FN in r.getMessage() and "credentials" in r.getMessage()
               for r in caplog.records)


def test_find_local_uri_prefers_numerically_latest(tmp_path):
    # find_local_uri uses a repo layout without the {release} segment.
    for ver in VERSIONS:
        d = tmp_path / "level2" / PLAN / ver / BAND
        d.mkdir(parents=True)
        (d / f"level2_{PLAN}_0516_2D{BAND}_spx_{ver}.fits").touch()

    result = find_local_uri(FN, rootdir=tmp_path)
    assert result is not None
    assert str(result).endswith(f"spx_{LATEST}.fits")
