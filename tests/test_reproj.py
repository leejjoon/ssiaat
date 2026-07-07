"""End-to-end reprojection: synthetic L2 file -> get_df_from_uri -> stable -> image.

The synthetic exposure is a constant field (signal 2.0 + zodi 1.0), so
after zodi subtraction and adaptive reprojection every template pixel
must come back ~2.0.
"""
import numpy as np
import pytest

from ssiaat.reproj import get_df_from_uri, merge_to_stable

from conftest import TMPL_SHAPE, L2_SHAPE, L2_SIGNAL, L2_BAND

N_TMPL_PIX = TMPL_SHAPE[0] * TMPL_SHAPE[1]


def test_get_df_from_uri(synthetic_l2_path, template_wcs):
    df = get_df_from_uri(template_wcs, f"file://{synthetic_l2_path}")

    # Template lies fully inside the detector footprint.
    assert len(df) == N_TMPL_PIX
    for col in ["tmpl_ind", "image", "variance", "src_x", "src_y", "wvl"]:
        assert col in df.columns

    # Zodi (1.0) subtracted from the constant 3.0 image.
    np.testing.assert_allclose(df["image"], L2_SIGNAL, rtol=1e-5)
    assert (df["variance"] > 0).all()

    assert df["tmpl_ind"].between(0, N_TMPL_PIX - 1).all()
    assert df["tmpl_ind"].is_unique
    assert df["src_x"].between(0, L2_SHAPE[1] - 1).all()
    assert df["src_y"].between(0, L2_SHAPE[0] - 1).all()
    assert np.isfinite(df["wvl"]).all()

    # Metadata columns from the header and the filename.
    assert (df["band"] == L2_BAND).all()
    assert (df["expidn"] == 12345).all()
    assert (df["pipe_run"] == "l2b-v19-2025-252").all()

    # Template header cards attached for the stable machinery.
    assert df.attrs["ssiaat_template_header"]


def test_process_single_reproject_kwargs(synthetic_l2_path, template_wcs):
    # Extra kwargs flow through to reproject_adaptive; the constant field
    # must survive a different kernel setting unchanged.
    from astropy.io import fits
    from ssiaat.reproj import SphxReprojector, get_metadata_from_filename

    with fits.open(synthetic_l2_path) as hdul:
        hdul["IMAGE"].data -= hdul["ZODI"].data
        reprojector = SphxReprojector(
            hdul, aux_metadata=get_metadata_from_filename(synthetic_l2_path))
        out = reprojector.process_single(template_wcs, kernel="gaussian")

    np.testing.assert_allclose(out[0].data, L2_SIGNAL, rtol=1e-5)


def test_get_df_from_uri_zodi_corrector(synthetic_l2_path, template_wcs):
    # Doubling the zodi (1.0 -> 2.0) leaves 3.0 - 2.0 = 1.0 of signal.
    df = get_df_from_uri(template_wcs, f"file://{synthetic_l2_path}",
                         zodi_corrector=lambda z: 2.0 * z)
    np.testing.assert_allclose(df["image"], 1.0, rtol=1e-5)


def test_merge_to_stable_and_simple_image(synthetic_l2_path, template_wcs):
    df = get_df_from_uri(template_wcs, f"file://{synthetic_l2_path}")
    # Two exposures: a single one gives a unique tmpl_ind index, which the
    # .spectral accessor rightly rejects (a stable needs duplicates).
    stable = merge_to_stable([df, df.copy()], tmpl_wcs=template_wcs)

    assert stable.index.name == "tmpl_ind"
    assert len(stable) == 2 * N_TMPL_PIX
    # sorted on write so parquet row-group statistics enable cutout reads
    assert stable.index.is_monotonic_increasing
    image = stable.spectral.make_simple_image(0.0, 10.0)
    assert image.shape == TMPL_SHAPE
    np.testing.assert_allclose(np.asarray(image), L2_SIGNAL, rtol=1e-5)
