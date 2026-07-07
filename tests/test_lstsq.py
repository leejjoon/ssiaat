"""vectorized_lstsq and the Model/FitResults fitting path.

Safety net for the planned Phase 5 rewrite: known-answer recovery on the
synthetic stable, an independent np.linalg.lstsq reference, a hardcoded
regression pin on real-ish example data, and chunked==unchunked.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssiaat.model.vectorized_lstsq import (
    vectorized_lstsq_numpy,
    vectorized_lstsq_chunked,
)
from ssiaat.spherex_table import Model

from conftest import PIXELS, TRUE_LINE_AMPS, TRUE_CONT_AMPS, TMPL_SHAPE

EXAMPLE_CSV = Path(__file__).parent / "data" / "lstsq_example.csv"
EXAMPLE_MODEL_COLUMNS = ["model0", "model1", "model2"]

# Regression pin: computed with vectorized_lstsq_numpy as of 2026-07-07
# (pre-Phase-5 implementation) on lstsq_example.csv, grouped by the
# tmpl_ind column, unweighted. Freezes observable behavior (values and
# group ordering) across the Phase 5 rewrite.
EXAMPLE_PINNED_INDEX = [1088, 1089, 1090]
EXAMPLE_PINNED_COEFFS = [
    [1.9920865896085427, 1.4817168867243980, 2.1875453211285340],
    [1.7945221222951935, 1.4912152482981502, 2.0002577622425837],
    [1.6481822045935535, 1.4917593004459653, 1.8793446749351300],
]


def _with_model_columns(stable, line_model, cont_model):
    df = stable.copy()
    df["model0"] = line_model(df["wvl"].values)
    df["model1"] = cont_model(df["wvl"].values)
    return df


@pytest.mark.parametrize("variance_column", [None, "variance"])
def test_known_answer_recovery(synthetic_stable, line_model, cont_model,
                               variance_column):
    df = _with_model_columns(synthetic_stable, line_model, cont_model)
    coeffs, idx = vectorized_lstsq_numpy(df, ["model0", "model1"],
                                         variance_column=variance_column)

    assert list(idx) == sorted(PIXELS)
    for k, pix in enumerate(idx):
        np.testing.assert_allclose(
            coeffs[k], [TRUE_LINE_AMPS[pix], TRUE_CONT_AMPS[pix]], rtol=1e-6)


def test_against_numpy_lstsq_reference():
    # Independent per-group reference on the example data (real-ish
    # numbers, unlike the well-conditioned synthetic stable).
    df = pd.read_csv(EXAMPLE_CSV, index_col=0)
    coeffs, idx = vectorized_lstsq_numpy(df, EXAMPLE_MODEL_COLUMNS,
                                         group_column="tmpl_ind")

    for k, group in enumerate(idx):
        sub = df[df["tmpl_ind"] == group]
        A = sub[EXAMPLE_MODEL_COLUMNS].values
        expected, *_ = np.linalg.lstsq(A, sub["image"].values, rcond=None)
        np.testing.assert_allclose(coeffs[k], expected, rtol=1e-6)


def test_regression_pin():
    df = pd.read_csv(EXAMPLE_CSV, index_col=0)
    coeffs, idx = vectorized_lstsq_numpy(df, EXAMPLE_MODEL_COLUMNS,
                                         group_column="tmpl_ind")

    np.testing.assert_array_equal(idx, EXAMPLE_PINNED_INDEX)
    np.testing.assert_allclose(coeffs, EXAMPLE_PINNED_COEFFS, rtol=1e-7)


def test_large_synthetic_matches_reference():
    # 200k rows / 4k groups: catches accumulation/indexing bugs the
    # 75-row CSV can't. Deterministic (seeded), unweighted.
    rng = np.random.default_rng(42)
    n_rows, n_groups = 200_000, 4_000
    df = pd.DataFrame({
        "g": rng.integers(0, n_groups, n_rows),
        "m0": rng.normal(size=n_rows),
        "m1": rng.normal(size=n_rows),
        "m2": np.ones(n_rows),
    })
    df["image"] = 1.5 * df["m0"] - 0.5 * df["m1"] + 3.0 + \
        0.01 * rng.normal(size=n_rows)

    coeffs, idx = vectorized_lstsq_numpy(df, ["m0", "m1", "m2"],
                                         group_column="g")

    for k in rng.choice(len(idx), size=20, replace=False):
        sub = df[df["g"] == idx[k]]
        expected, *_ = np.linalg.lstsq(sub[["m0", "m1", "m2"]].values,
                                       sub["image"].values, rcond=None)
        np.testing.assert_allclose(coeffs[k], expected, rtol=1e-8)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_chunked_matches_unchunked():
    # Unlike the old inline example(), pass the same group_column to both
    # implementations (chunked defaults to "tmpl_ind", numpy to the index).
    df = pd.read_csv(EXAMPLE_CSV, index_col=0)

    coeffs, errs, idx = vectorized_lstsq_numpy(
        df, EXAMPLE_MODEL_COLUMNS, group_column="tmpl_ind",
        return_error=True)
    coeffs_c, errs_c, idx_c = vectorized_lstsq_chunked(
        df, EXAMPLE_MODEL_COLUMNS, chunk_size=2, group_column="tmpl_ind",
        return_error=True)

    np.testing.assert_array_equal(idx, idx_c)
    np.testing.assert_allclose(coeffs, coeffs_c, rtol=1e-12)
    np.testing.assert_allclose(errs, errs_c, rtol=1e-12)


def test_model_least_square_fit(synthetic_stable, line_model, cont_model):
    model = Model([line_model], [cont_model])
    fitted = model.least_square_fit(synthetic_stable, return_error=True)

    assert len(fitted.C) == 1
    assert len(fitted.contC) == 1
    assert list(fitted.C[0].index) == sorted(PIXELS)

    for pix in PIXELS:
        np.testing.assert_allclose(fitted.C[0][pix], TRUE_LINE_AMPS[pix],
                                   rtol=1e-6)
        np.testing.assert_allclose(fitted.contC[0][pix], TRUE_CONT_AMPS[pix],
                                   rtol=1e-6)

    assert fitted.Cerr[0].shape == fitted.C[0].shape

    # Template metadata must propagate into the coefficient Series so
    # coefficient maps can be turned back into images.
    assert (fitted.C[0].attrs["ssiaat_template_header"]
            == synthetic_stable.attrs["ssiaat_template_header"])
    coef_image = fitted.C[0].itable.to_image()
    assert coef_image.shape == TMPL_SHAPE
    assert np.isfinite(np.asarray(coef_image)).sum() == len(PIXELS)
