"""AsyncCollector and the async bulk-reprojection runner.

Written with plain asyncio.run so no pytest-asyncio plugin is needed. The
timeout wrappers matter: before failures were caught inside the worker, a
raising processor killed workers and the run deadlocked on queue.join().
"""
import asyncio

import pandas as pd
import pytest

from ssiaat.async_collector import AsyncCollector
from ssiaat.reproj_s3_async import run_reproj_tasks

TIMEOUT = 60  # generous; a deadlock regression fails via TimeoutError


async def _double(item):
    await asyncio.sleep(0)
    return 2 * item


def test_collects_all_results():
    collector = AsyncCollector(range(10), _double)
    results = asyncio.run(asyncio.wait_for(collector.run(num_tasks=4), TIMEOUT))
    assert sorted(results) == [2 * i for i in range(10)]
    assert collector.failures == []


def test_accepts_async_iterable():
    async def agen():
        for i in range(5):
            yield i

    async def run():
        collector = AsyncCollector(agen(), _double, total=5)
        return await asyncio.wait_for(collector.run(num_tasks=2), TIMEOUT)

    assert sorted(asyncio.run(run())) == [0, 2, 4, 6, 8]


def test_none_results_skipped():
    async def keep_even(item):
        return item if item % 2 == 0 else None

    collector = AsyncCollector(range(10), keep_even)
    results = asyncio.run(asyncio.wait_for(collector.run(num_tasks=3), TIMEOUT))
    assert sorted(results) == [0, 2, 4, 6, 8]


def test_failures_recorded_without_deadlock():
    # 6 failing items with only 4 workers: under the old behavior every
    # worker dies and queue.join() waits forever (TimeoutError here).
    async def flaky(item):
        await asyncio.sleep(0)
        if item < 6:
            raise ValueError(f"bad item {item}")
        return item

    collector = AsyncCollector(range(10), flaky)
    results = asyncio.run(asyncio.wait_for(collector.run(num_tasks=4), TIMEOUT))

    assert sorted(results) == [6, 7, 8, 9]
    assert len(collector.failures) == 6
    failed_items = sorted(item for item, _ in collector.failures)
    assert failed_items == [0, 1, 2, 3, 4, 5]
    assert all(isinstance(exc, ValueError) for _, exc in collector.failures)


def test_progress_smoke():
    collector = AsyncCollector(range(4), _double)
    results = asyncio.run(
        asyncio.wait_for(collector.run(num_tasks=2, progress=True), TIMEOUT))
    assert len(results) == 4


def test_run_reproj_tasks_collects_results_and_failures(
        synthetic_l2_path, template_wcs, tmp_path, caplog):
    # Two good exposures plus one missing file: the run must complete
    # (bug-first: the old worker died on the error and queue.join() hung),
    # return both DataFrames, and report the bad URI as a failure.
    import logging
    caplog.set_level(logging.INFO, logger="ssiaat.reproj_s3_async")

    good = f"file://{synthetic_l2_path}"
    bad = f"file://{tmp_path}/does_not_exist.fits"

    dfl, failures = asyncio.run(asyncio.wait_for(
        run_reproj_tasks([good, good, None, bad], template_wcs,
                         num_fetchers=2, num_workers=0, progress=False),
        TIMEOUT))

    assert len(dfl) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dfl)

    assert len(failures) == 1
    failed_uri, exc = failures[0]
    assert failed_uri == bad
    assert isinstance(exc, Exception)

    assert any("1 failed" in r.getMessage() for r in caplog.records)


def test_run_reproj_tasks_process_pool(synthetic_l2_path, template_wcs,
                                       tmp_path):
    # Pooled run: same results as inline, bad URI still a failure, run
    # completes under timeout.
    good = f"file://{synthetic_l2_path}"
    bad = f"file://{tmp_path}/does_not_exist.fits"

    inline, _ = asyncio.run(asyncio.wait_for(
        run_reproj_tasks([good], template_wcs, num_workers=0,
                         progress=False),
        TIMEOUT))
    pooled, failures = asyncio.run(asyncio.wait_for(
        run_reproj_tasks([good, good, bad], template_wcs, num_fetchers=2,
                         num_workers=2, progress=False),
        TIMEOUT))

    assert len(pooled) == 2
    assert len(failures) == 1
    for df in pooled:
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), inline[0].reset_index(drop=True))


def test_run_reproj_tasks_unpicklable_corrector(synthetic_l2_path,
                                                template_wcs):
    # A lambda corrector cannot cross the process boundary: it must land
    # in failures (with the run completing), not hang or crash the run.
    good = f"file://{synthetic_l2_path}"

    dfl, failures = asyncio.run(asyncio.wait_for(
        run_reproj_tasks([good], template_wcs, num_workers=2,
                         progress=False, zodi_corrector=lambda z: z),
        TIMEOUT))

    assert dfl == []
    assert len(failures) == 1
