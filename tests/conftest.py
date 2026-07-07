"""Shared fixtures: a small synthetic template and stable.

The synthetic stable is built so that ``image`` is an exact (noiseless)
linear combination of a line model and a continuum model with known,
per-pixel amplitudes, so the least-squares fitter must recover them to
numerical precision.
"""
import numpy as np
import pandas as pd
import pytest
import astropy.units as u

from ssiaat.wcs_helper import get_wcs_from_shape
from ssiaat.spherex_table import SsiaatConverter, promote_to_stable
from ssiaat.model import sed

TMPL_SHAPE = (16, 16)

# Pixels (flat template indices, y*16 + x) that get spectra in the
# synthetic stable.
PIXELS = [0, 1, 17, 100, 200, 255]

# Per-pixel true amplitudes. They differ from pixel to pixel on purpose:
# with identical amplitudes everywhere, a group-alignment bug in the
# fitter (coefficients attributed to the wrong pixel) would go unnoticed.
TRUE_LINE_AMPS = {pix: 2.0 + 0.01 * k for k, pix in enumerate(PIXELS)}
TRUE_CONT_AMPS = {pix: 0.5 + 0.005 * k for k, pix in enumerate(PIXELS)}

WVL = np.linspace(3.95, 4.15, 25)


@pytest.fixture(scope="session")
def line_model():
    return sed.hflattop(4.02, 4.08, a=8.0)


@pytest.fixture(scope="session")
def cont_model():
    return sed.const()


@pytest.fixture(scope="session")
def template_header():
    wcs = get_wcs_from_shape(150.0 * u.deg, 2.0 * u.deg, TMPL_SHAPE)
    header = wcs.to_header()
    # WCS.to_header() carries no NAXIS* cards, but SsiaatConverter needs
    # them (same injection as reproj.merge_to_stable).
    header["NAXIS1"] = TMPL_SHAPE[1]
    header["NAXIS2"] = TMPL_SHAPE[0]
    return header


@pytest.fixture(scope="session")
def converter(template_header):
    return SsiaatConverter(template_header)


@pytest.fixture
def synthetic_stable(template_header, line_model, cont_model):
    """A stable with PIXELS x len(WVL) rows and a noiseless image column.

    Each pixel MUST have >= 2 wavelength rows: check_index_stable requires
    an integer index WITH duplicates, so do not "simplify" this fixture to
    one row per pixel.
    """
    rows = []
    for pix in PIXELS:
        image = (TRUE_LINE_AMPS[pix] * line_model(WVL)
                 + TRUE_CONT_AMPS[pix] * cont_model(WVL))
        rows.append(pd.DataFrame({
            "tmpl_ind": pix,
            "wvl": WVL,
            "image": image,
        }))
    df = pd.concat(rows, ignore_index=True)
    # Deterministic, positive, non-constant variance (weights differ per
    # row; the fit is still exact because the image is noiseless).
    df["variance"] = 0.01 + 0.0005 * np.arange(len(df))
    return promote_to_stable(df, header=template_header)


@pytest.fixture
def stable_parquet(synthetic_stable, tmp_path):
    fn = tmp_path / "stable.parquet"
    synthetic_stable.to_parquet(fn)
    return fn
