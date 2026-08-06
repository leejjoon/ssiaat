"""BandpassTool smoke tests (the class had two fatal typos and had never run)."""
import numpy as np
import pytest

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
    assert diagnostics["resid"].shape == diagnostics["w"].shape


def test_fit_hflatline_fixed_mu_center_returned_verbatim():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    params = tool.fit_hflatline(4.05, mu_center=4.05)
    assert params["mu_center_left"] == 4.05  # exact, not read from popt


def test_fit_hflatline_fixed_mu_center_matches_free_fit_quality():
    """Fixing mu_center at a value that is NOT the stub's true symmetry
    center should still fit about as well as the free fit -- the
    degeneracy is in the FIT, not a real dependency on getting mu_center
    exactly right (see fit_hflatline's docstring: measured identical to 4
    sig figs on the real bandpass)."""
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    true = StubHflatlineBandpass.TRUE_PARAMS
    offset_center = true["mu_center_left"] + 0.003  # deliberately off-true

    _, free_diag = tool.fit_hflatline(true["mu_center_left"], full_output=True)
    fixed_params, fixed_diag = tool.fit_hflatline(
        true["mu_center_left"], mu_center=offset_center, full_output=True)

    assert fixed_diag["rms"] <= 1.5 * free_diag["rms"]
    assert fixed_diag["rms"] < 1e-5

    true_left_edge = true["mu_center_left"] - true["dmu_left"]
    true_right_edge = true["mu_center_left"] + true["dmu_right"]
    fixed_left_edge = fixed_params["mu_center_left"] - fixed_params["dmu_left"]
    fixed_right_edge = fixed_params["mu_center_left"] + fixed_params["dmu_right"]
    np.testing.assert_allclose(fixed_left_edge, true_left_edge, atol=1e-4)
    np.testing.assert_allclose(fixed_right_edge, true_right_edge, atol=1e-4)


class StubDriftingHflatlineBandpass:
    """A bandpass model whose TRUE shape parameters drift smoothly with
    iy, so fixing mu_center (at the analytic value) should recover each
    individual parameter accurately -- unlike the free-mu_center fit,
    which cannot (see test_fit_hflatline_recovers_known_shape's edge-only
    assertions). This is the "the degeneracy is actually resolved" check.
    """

    @staticmethod
    def true_params(iy):
        return dict(
            mu_center_left=3.9 + 1e-4 * iy,
            dmu_left=0.020 + 2e-6 * iy,
            a_left=1.20 - 5e-5 * iy,
            dmu_right=0.015 + 1e-6 * iy,
            a_right=0.90 + 3e-5 * iy,
        )

    TRUE_AMPLITUDE = 0.08

    def __call__(self, ix, iy, central_bandpass_only=False, array=None):
        from ssiaat.model.sed import hflatline
        iy = np.asarray(iy, dtype=float)
        if iy.ndim > 0:
            return 3.9 + 1e-4 * iy, np.ones_like(iy)

        p = self.true_params(float(iy))
        if central_bandpass_only:
            return p["mu_center_left"], 1.0

        shape = hflatline(p["mu_center_left"], p["dmu_left"], p["a_left"],
                          dmu_right=p["dmu_right"], a_right=p["a_right"])
        w = np.linspace(p["mu_center_left"] - 0.1, p["mu_center_left"] + 0.1, 400)
        return w, self.TRUE_AMPLITUDE * shape(w)


def test_fit_hflatline_at_iy_uses_analytic_center():
    tool = BandpassTool(StubDriftingHflatlineBandpass(), band=5)
    params = tool.fit_hflatline_at_iy(500)
    assert params["mu_center_left"] == 3.9 + 1e-4 * 500  # exact


def test_fit_hflatline_at_iy_recovers_true_shape_params_and_is_smooth():
    tool = BandpassTool(StubDriftingHflatlineBandpass(), band=5)
    iys = np.arange(0, 2000, 100)  # 20 rows across the drift range

    recovered = {k: [] for k in
                ("dmu_left", "a_left", "dmu_right", "a_right")}
    for iy in iys:
        params = tool.fit_hflatline_at_iy(int(iy))
        truth = StubDriftingHflatlineBandpass.true_params(float(iy))
        for k in recovered:
            np.testing.assert_allclose(params[k], truth[k], atol=1e-3, rtol=1e-2)
            recovered[k].append(params[k])

    # Truth is monotone in iy by construction; the fixed-mu_center fit
    # must recover that monotonicity (zero sign flips), which the
    # free-mu_center fit could not do reliably (degeneracy noise).
    for k, values in recovered.items():
        diffs = np.diff(values)
        assert np.all(diffs > 0) or np.all(diffs < 0), (
            f"{k} is not monotone in the fixed-mu_center fit: {values}")


def test_fit_hflatline_rejects_mu_center_outside_curve():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    with pytest.raises(ValueError, match="mu_center"):
        tool.fit_hflatline(4.05, mu_center=10.0)


def test_fit_hflatline_symmetric_with_fixed_center():
    tool = BandpassTool(StubHflatlineBandpass(), band=5)
    params = tool.fit_hflatline(4.05, mu_center=4.05, symmetric=True)
    assert params["mu_center_left"] == 4.05
    assert params["dmu_left"] == params["dmu_right"]
    assert params["a_left"] == params["a_right"]
