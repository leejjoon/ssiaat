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
def template_wcs():
    return get_wcs_from_shape(150.0 * u.deg, 2.0 * u.deg, TMPL_SHAPE)


@pytest.fixture(scope="session")
def template_header(template_wcs):
    header = template_wcs.to_header()
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


# --- synthetic SPHEREx L2 exposure -----------------------------------------

L2_SHAPE = (2040, 2040)
L2_SIGNAL = 2.0   # constant sky signal after zodi subtraction
L2_ZODI = 1.0     # constant zodi level baked into IMAGE
L2_BAND = 1
L2_FILENAME = "level2_2025W24_1A_0405_1D1_spx_l2b-v19-2025-252.fits"


@pytest.fixture(scope="session")
def synthetic_l2_path(tmp_path_factory):
    """A minimal but structurally valid L2 file on disk.

    Layout required by ingest_hdul / SphxReprojector: IMAGE must be HDU
    index 1 (SphxReprojector reads hdul[1].header for EXPIDN/DETECTOR and
    hdul["IMAGE"] by name), with FLAGS/VARIANCE/ZODI extensions and the
    L2DQAFLG keyword. Same sky center as template_header, so the 16x16
    template falls entirely inside the detector footprint.
    """
    from astropy.io import fits

    wcs_in = get_wcs_from_shape(150.0 * u.deg, 2.0 * u.deg, L2_SHAPE)
    header = wcs_in.to_header()
    header["DETECTOR"] = L2_BAND
    header["EXPIDN"] = 12345
    header["L2DQAFLG"] = 0

    image = np.full(L2_SHAPE, L2_SIGNAL + L2_ZODI, dtype="float32")
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=image, header=header, name="IMAGE"),
        fits.ImageHDU(data=np.zeros(L2_SHAPE, dtype="int32"), name="FLAGS"),
        fits.ImageHDU(data=np.ones(L2_SHAPE, dtype="float32"), name="VARIANCE"),
        fits.ImageHDU(data=np.full(L2_SHAPE, L2_ZODI, dtype="float32"),
                      name="ZODI"),
    ])

    fn = tmp_path_factory.mktemp("l2") / L2_FILENAME
    hdul.writeto(fn)
    return fn
