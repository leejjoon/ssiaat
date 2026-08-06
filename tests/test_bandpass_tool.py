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


class StubHflatlineBandpass:
    """A bandpass model whose true shape IS an sed.hflatline, so
    fit_hflatline has a known answer to recover exactly (up to fit
    tolerance) rather than just "some small residual"."""

    TRUE_PARAMS = dict(mu_center_left=4.05, dmu_left=0.02, a_left=1.2,
                       dmu_right=0.015, a_right=0.9)
    TRUE_AMPLITUDE = 0.08

    def __call__(self, ix, iy, central_bandpass_only=False, array=None):
        from ssiaat.model.sed import hflatline
        iy = np.asarray(iy, dtype=float)
        if iy.ndim > 0:
            # __init__'s calibration sweep: a linear central-wavelength
            # ramp is all that's needed to build wvl_to_iy.
            return 3.9 + 0.0001 * iy, np.ones_like(iy)
        center = 3.9 + 0.0001 * float(iy)
        shape = hflatline(self.TRUE_PARAMS["mu_center_left"],
                          self.TRUE_PARAMS["dmu_left"],
                          self.TRUE_PARAMS["a_left"],
                          dmu_right=self.TRUE_PARAMS["dmu_right"],
                          a_right=self.TRUE_PARAMS["a_right"])
        w = np.linspace(center - 0.1, center + 0.1, 400)
        return w, self.TRUE_AMPLITUDE * shape(w)


def test_fit_hflatline_recovers_known_shape():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    true = StubHflatlineBandpass.TRUE_PARAMS

    params = tool.fit_hflatline(true["mu_center_left"])

    assert set(params) == {"mu_center_left", "dmu_left", "a_left",
                           "dmu_right", "a_right", "amplitude"}
    # amplitude and the reconstructed curve are essentially exact (see
    # test_fit_hflatline_full_output_reports_small_rms).
    np.testing.assert_allclose(params["amplitude"],
                               StubHflatlineBandpass.TRUE_AMPLITUDE, atol=1e-6)
    # With asymmetric edges, mu_center_left trades off against
    # dmu_left/dmu_right (see fit_hflatline's docstring) -- only the edge
    # *positions* are tightly identified, not the individual parameters.
    true_left_edge = true["mu_center_left"] - true["dmu_left"]
    true_right_edge = true["mu_center_left"] + true["dmu_right"]
    fit_left_edge = params["mu_center_left"] - params["dmu_left"]
    fit_right_edge = params["mu_center_left"] + params["dmu_right"]
    np.testing.assert_allclose(fit_left_edge, true_left_edge, atol=1e-3)
    np.testing.assert_allclose(fit_right_edge, true_right_edge, atol=1e-3)


def test_fit_hflatline_symmetric_forces_matched_edges():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    params = tool.fit_hflatline(4.05, symmetric=True)
    assert params["dmu_left"] == params["dmu_right"]
    assert params["a_left"] == params["a_right"]


def test_fit_hflatline_full_output_reports_small_rms():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    params, diagnostics = tool.fit_hflatline(4.05, full_output=True)
    assert diagnostics["rms"] < 1e-6  # exact hflatline target, near-exact fit
    assert diagnostics["w"].shape == diagnostics["t"].shape
