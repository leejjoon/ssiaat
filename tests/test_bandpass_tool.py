"""BandpassTool smoke tests (the class had two fatal typos and had never run)."""
import numpy as np

from ssiaat.spherex_table import BandpassTool
from ssiaat.tabular_bandpass_lite import Tabular_Bandpass_Lite


class StubBandpass:
    """Minimal stand-in for the full spherex_tabular_bandpass model.

    Wavelength is a linear ramp in iy. For an array iy (the __init__
    calibration call) it returns per-pixel central wavelengths; for a
    scalar iy (the get_bp_at_wvl call) it returns a small tabulated
    bandpass curve around that wavelength.
    """

    def __call__(self, ix, iy, central_bandpass_only=False, array=None):
        iy = np.asarray(iy, dtype=float)
        if iy.ndim > 0:
            wvl = 4.0 + 0.001 * iy
            return wvl, np.ones_like(wvl)
        center = 4.0 + 0.001 * float(iy)
        w = center + np.linspace(-0.02, 0.02, 5)
        return w, np.ones_like(w)


def test_init_builds_wvl_to_iy():
    tool = BandpassTool(StubBandpass(), band=5)
    np.testing.assert_allclose(tool.wvl_to_iy(4.5), 500.0)


def test_get_bp_at_wvl():
    tool = BandpassTool(StubBandpass(), band=5)

    w1, t1 = tool.get_bp_at_wvl(4.5)
    np.testing.assert_allclose(w1.mean(), 4.5)
    np.testing.assert_array_equal(t1, np.ones_like(w1))

    knots = tool.get_bp_at_wvl(4.5, as_knots=True)
    assert callable(knots)
    np.testing.assert_allclose(knots(4.5), 1.0)


def test_init_with_tabular_bandpass_lite():
    # The Lite model supports the central_bandpass_only=True calibration
    # call used by __init__ (get_bp_at_wvl needs the full bandpass model).
    tool = BandpassTool(Tabular_Bandpass_Lite(), band=5)
    assert tool.band == 5
    center_wvl = float(tool.wvl_to_iy.x.mean())
    assert 0.5 < center_wvl < 6.0  # SPHEREx wavelength range, microns
