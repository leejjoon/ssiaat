"""BandpassShapeLite: dependency-free analytic bandpass shape.

The pinned values freeze the current Chebyshev evaluation (mirrors
test_bandpass_lite.py's pattern); the optional comparison against the
full spherex_tabular_bandpass package runs only where that package is
installed.
"""
import importlib.util

import numpy as np
import pytest

from ssiaat.bandpass_shape_lite import BandpassShapeLite
from ssiaat.model.sed import hflatline
from ssiaat.tabular_bandpass_lite import Tabular_Bandpass_Lite

# One probe wavelength per band -- the midpoint of that band's validated
# domain. Computed with the current model, 2026-08-06.
PROBE_WVL = {
    1: 0.9242384356581094,
    2: 1.3656157820417718,
    3: 1.9845806927643455,
    4: 3.1521171006926703,
    5: 4.102481709658159,
    6: 4.694818285104098,
}

PINNED_PARAMS = {
    1: {'amplitude': 0.08163518863081629, 'dmu_left': 0.011371798000999581, 'a_left': 1.5362322538120396, 'dmu_right': 0.011196905249904578, 'a_right': 1.2759576007704228, 'floor0': 0.00026441923517108877, 'floor1': -0.0007913389579594429},
    2: {'amplitude': 0.08884336243932159, 'dmu_left': 0.015176434854434519, 'a_left': 1.6562076223777544, 'dmu_right': 0.017053890960582043, 'a_right': 1.8776017831068328, 'floor0': 0.00011065203589348878, 'floor1': -0.0002491633766040981},
    3: {'amplitude': 0.08078837038614622, 'dmu_left': 0.02325339205588071, 'a_left': 1.5837855935384202, 'dmu_right': 0.024582095483764586, 'a_right': 1.4938992033546832, 'floor0': 5.415046847022808e-05, 'floor1': -4.3153198657918205e-05},
    4: {'amplitude': 0.041925539636672025, 'dmu_left': 0.04191824407621348, 'a_left': 1.4173941246490747, 'dmu_right': 0.04938835151698491, 'a_right': 1.4925581523705498, 'floor0': 3.89650736988196e-05, 'floor1': 5.034137304156346e-06},
    5: {'amplitude': 0.0877260941321212, 'dmu_left': 0.014752221033341709, 'a_left': 0.6443359246577178, 'dmu_right': 0.01913301400043055, 'a_right': 0.8488948118288563, 'floor0': -2.3751906661002203e-05, 'floor1': 2.036810131883099e-05},
    6: {'amplitude': 0.11975955610297165, 'dmu_left': 0.013525622533785177, 'a_left': 0.5295746558248474, 'dmu_right': 0.019386632542929175, 'a_right': 0.8539030253530284, 'floor0': -9.11584562569558e-06, 'floor1': 1.429817287711758e-05},
}


@pytest.fixture(scope="module")
def model():
    return BandpassShapeLite()


@pytest.mark.parametrize("band", range(1, 7))
def test_pinned_params(model, band):
    params = model.params(PROBE_WVL[band], array=band)
    assert params["mu_center_left"] == PROBE_WVL[band]
    for key, value in PINNED_PARAMS[band].items():
        np.testing.assert_allclose(params[key], value, rtol=1e-7)


@pytest.mark.parametrize("band", range(1, 7))
def test_pinned_native_grid(model, band):
    w, _ = model(PROBE_WVL[band], array=band)
    w_start, dw, n = model.NATIVE_GRID[band]
    assert w.shape == (n,)
    np.testing.assert_allclose(w[0], w_start)
    np.testing.assert_allclose(np.diff(w), dw)


@pytest.mark.parametrize("band", range(1, 7))
def test_params_are_hflatline_kwargs(model, band):
    params = model.params(PROBE_WVL[band], array=band)
    hflatline_kwargs = {k: params[k] for k in
                        ("dmu_left", "a_left", "dmu_right", "a_right")}
    shape = hflatline(params["mu_center_left"], **hflatline_kwargs)
    # Locate the peak on a fine grid, not the coarse native grid -- at
    # native spacing (dw ~ 2-4e-3 um), grid quantization alone is
    # comparable to the tolerance being tested.
    halfwidth = 3 * max(params["dmu_left"], params["dmu_right"])
    w_fine = np.linspace(params["mu_center_left"] - halfwidth,
                         params["mu_center_left"] + halfwidth, 20001)
    peak_wvl = w_fine[np.argmax(shape(w_fine))]
    # hflatline's peak isn't necessarily exactly at mu_center when
    # dmu_left != dmu_right (asymmetric edges shift it slightly) -- allow
    # a fraction of the half-width.
    tol = 0.25 * max(params["dmu_left"], params["dmu_right"])
    assert abs(peak_wvl - params["mu_center_left"]) <= tol
    assert params["mu_center_left"] == PROBE_WVL[band]


@pytest.mark.parametrize("band", range(1, 7))
def test_call_matches_params_reconstruction(model, band):
    """__call__'s formula, spelled out, must match its own output --
    guards the module docstring's reconstruction formula against drift."""
    center = PROBE_WVL[band]
    w, t = model(center, array=band)
    p = model.params(center, array=band)
    shape = hflatline(p["mu_center_left"], p["dmu_left"], p["a_left"],
                      dmu_right=p["dmu_right"], a_right=p["a_right"])
    expected = p["amplitude"] * shape(w) + p["floor0"] + p["floor1"] * (w - p["mu_center_left"])
    np.testing.assert_allclose(t, expected)


def test_local_grid_is_compact_and_covers_profile():
    model = BandpassShapeLite()
    center = PROBE_WVL[5]
    w_native, t_native = model(center, array=5, grid="native")
    w_local, t_local = model(center, array=5, grid="local")
    assert len(w_local) < len(w_native)
    peak = t_local.max()
    assert t_local[0] < 1e-3 * peak
    assert t_local[-1] < 1e-3 * peak


def test_caller_supplied_wvl_grid():
    model = BandpassShapeLite()
    wvl = np.array([4.0, 4.05, 4.1])
    w, t = model(4.05, array=5, wvl=wvl)
    np.testing.assert_array_equal(w, wvl)
    assert t.shape == wvl.shape


def test_out_of_domain_warns_but_still_returns_a_curve():
    model = BandpassShapeLite()
    lo, hi = model.SHAPE_CHEB_DOMAIN[3]
    with pytest.warns(UserWarning, match="outside band 3"):
        w, t = model(hi + 0.5, array=3)
    assert w.shape == t.shape
    assert np.all(np.isfinite(t))


def test_in_domain_does_not_warn(recwarn):
    model = BandpassShapeLite()
    model(PROBE_WVL[5], array=5)
    assert len(recwarn) == 0


def test_rejects_unsupported_band():
    model = BandpassShapeLite()
    with pytest.raises(ValueError, match="array"):
        model(4.0, array=9)
    with pytest.raises(ValueError, match="array"):
        model.params(4.0, array=9)


def test_invalid_grid_name_raises():
    model = BandpassShapeLite()
    with pytest.raises(ValueError, match="grid"):
        model(4.05, array=5, grid="bogus")


@pytest.mark.parametrize("band", range(1, 7))
def test_at_pixel_matches_lite_center(band):
    model = BandpassShapeLite()
    iy = 1024
    expected_center, _ = Tabular_Bandpass_Lite()(1024, iy, array=band,
                                                  central_bandpass_only=True)
    params = model.at_pixel(iy, array=band)
    np.testing.assert_allclose(params["mu_center_left"], expected_center)


@pytest.mark.skipif(
    importlib.util.find_spec("spherex_tabular_bandpass") is None,
    reason="spherex_tabular_bandpass not installed")
@pytest.mark.parametrize("band", range(1, 7))
def test_matches_full_bandpass_model(band):
    """Curve, integral, and centroid accuracy against the real package --
    thresholds set ~2x above the derivation script's own measured values
    (a regression guard, not a re-run of that validation)."""
    from spherex_tabular_bandpass import Tabular_Bandpass

    from ssiaat.spherex_table import BandpassTool

    model = BandpassShapeLite()
    bp = Tabular_Bandpass()
    tool = BandpassTool(bp, band=band)

    lo, hi = model.SHAPE_CHEB_DOMAIN[band]
    centers = np.linspace(lo, hi, 8)[1:-1]  # avoid the exact validated edges

    rms_list, integral_err_list, centroid_err_list = [], [], []
    for center in centers:
        w_real, t_real = tool.get_bp_at_wvl(float(center))
        w_real = np.asarray(w_real, dtype=float)
        t_real = np.asarray(t_real, dtype=float)

        _, t_model = model(float(center), array=band, wvl=w_real)

        rms_list.append(float(np.sqrt(np.mean((t_model - t_real) ** 2))))
        sum_real, sum_model = float(t_real.sum()), float(t_model.sum())
        integral_err_list.append(abs(sum_model - sum_real) / sum_real)
        centroid_real = float(np.sum(w_real * t_real) / sum_real)
        centroid_model = float(np.sum(w_real * t_model) / sum_model)
        centroid_err_list.append(abs(centroid_model - centroid_real))

    _, dw, _ = model.NATIVE_GRID[band]
    assert np.median(rms_list) <= 2 * model.TYPICAL_RMS[band]
    assert np.median(integral_err_list) <= 0.02  # 2x the 1% derivation target
    assert np.median(centroid_err_list) <= 0.4 * dw  # 2x the loosest 0.1x*2 case
