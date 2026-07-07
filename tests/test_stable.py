"""promote_to_stable, read_stable, and the .spectral accessor.

The parquet round-trip tests are the key guard: the whole workflow relies
on pandas/pyarrow carrying df.attrs (the template header cards) through
to_parquet / read_parquet.
"""
import numpy as np
import pandas as pd
import pytest

from ssiaat.spherex_table import (
    check_index_stable,
    check_index_itable,
    promote_to_stable,
    read_stable,
)
from ssiaat.wcs_helper import TemplateHeaderCards

from conftest import TMPL_SHAPE, PIXELS


def test_check_index_predicates():
    dup_int = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])
    unique_int = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
    dup_float = pd.DataFrame({"a": [1, 2, 3]}, index=[1.0, 1.0, 2.0])

    assert check_index_stable(dup_int)
    assert not check_index_itable(dup_int)

    assert not check_index_stable(unique_int)
    assert check_index_itable(unique_int)

    assert not check_index_stable(dup_float)
    assert not check_index_itable(dup_float)


def test_promote_sets_index_and_attrs(template_header):
    df = pd.DataFrame({"tmpl_ind": [0, 0, 1, 1], "image": [1.0, 2.0, 3.0, 4.0]})
    stable = promote_to_stable(df, header=template_header)

    assert stable.index.name == "tmpl_ind"
    assert check_index_stable(stable)
    expected = TemplateHeaderCards.from_header(template_header)
    assert stable.attrs["ssiaat_template_header"] == expected.header_image_list
    assert stable.attrs["ssiaat_template_header_hash"] == expected.hash


def test_promote_without_header_or_attrs_raises():
    df = pd.DataFrame({"tmpl_ind": [0, 0, 1], "image": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="template header"):
        promote_to_stable(df)


def test_promote_rejects_unique_index(template_header):
    df = pd.DataFrame({"tmpl_ind": [0, 1, 2], "image": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="index"):
        promote_to_stable(df, header=template_header)
    # ... unless explicitly overridden.
    stable = promote_to_stable(df, header=template_header,
                               ignore_index_check=True)
    assert stable.index.name == "tmpl_ind"


def test_attrs_survive_parquet_roundtrip(synthetic_stable, stable_parquet):
    df = pd.read_parquet(stable_parquet)

    assert (df.attrs["ssiaat_template_header"]
            == synthetic_stable.attrs["ssiaat_template_header"])
    assert (df.attrs["ssiaat_template_header_hash"]
            == synthetic_stable.attrs["ssiaat_template_header_hash"])
    # The duplicated integer index must round-trip too.
    assert df.index.name == "tmpl_ind"
    assert check_index_stable(df)
    pd.testing.assert_frame_equal(df, synthetic_stable)


def test_read_stable_single_and_concat(synthetic_stable, stable_parquet):
    stable = read_stable(stable_parquet)
    assert len(stable) == len(synthetic_stable)
    assert check_index_stable(stable)

    stable2 = read_stable(stable_parquet, stable_parquet)
    assert len(stable2) == 2 * len(synthetic_stable)


def test_read_stable_accepts_list(stable_parquet):
    # Scripts pass a single list; varargs and list forms must agree.
    from_list = read_stable([stable_parquet, stable_parquet])
    from_varargs = read_stable(stable_parquet, stable_parquet)
    pd.testing.assert_frame_equal(from_list, from_varargs)


def test_read_stable_columns(stable_parquet):
    stable = read_stable(stable_parquet, columns=["wvl", "image"])
    assert list(stable.columns) == ["wvl", "image"]
    assert stable.index.name == "tmpl_ind"


def test_read_stable_wvl_range(synthetic_stable, stable_parquet):
    w1, w2 = 4.0, 4.1
    stable = read_stable(stable_parquet, wvl_range=(w1, w2))
    expected = synthetic_stable.query("(@w1 < wvl) and (wvl < @w2)")
    assert len(stable) == len(expected)
    assert stable["wvl"].between(w1, w2, inclusive="neither").all()


def test_converter_raises_without_metadata():
    df = pd.DataFrame({"image": [1.0, 2.0, 3.0]}, index=[0, 0, 1])
    with pytest.raises(ValueError, match="promote_to_stable"):
        df.spectral.converter


def test_read_stable_rejects_mismatched_headers(stable_parquet, tmp_path,
                                                template_header):
    other_header = template_header.copy()
    other_header["NAXIS1"] = 8
    df = pd.DataFrame({"tmpl_ind": [0, 0, 1], "image": [1.0, 2.0, 3.0]})
    df = promote_to_stable(df, header=other_header)
    fn2 = tmp_path / "other.parquet"
    df.to_parquet(fn2)

    with pytest.raises(ValueError, match="inconsistent"):
        read_stable(stable_parquet, fn2)


def test_converter_read_stable_caches_converter(converter, stable_parquet):
    stable = converter.read_stable(stable_parquet)
    assert stable._ssiaat_converter is converter
    assert stable.spectral.converter is converter


def test_spectral_converter_reconstructed_from_attrs(stable_parquet):
    # A freshly read parquet has no live converter; the accessor must
    # rebuild one from the attrs metadata alone.
    df = pd.read_parquet(stable_parquet)
    conv = df.spectral.converter
    assert conv is not None
    assert conv.tmpl_shape == TMPL_SHAPE


def test_make_simple_image(synthetic_stable):
    image = synthetic_stable.spectral.make_simple_image(3.9, 4.2)
    assert image.shape == TMPL_SHAPE

    flat = np.ravel(np.asarray(image))
    finite_pixels = np.flatnonzero(np.isfinite(flat))
    np.testing.assert_array_equal(finite_pixels, sorted(PIXELS))

    means = synthetic_stable.groupby(synthetic_stable.index)["image"].mean()
    np.testing.assert_allclose(flat[finite_pixels], means.loc[sorted(PIXELS)])
