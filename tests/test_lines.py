"""Tests for the line-model registry (ssiaat.model.lines) and the sed facade."""
import warnings

import numpy as np
import pandas as pd
import pytest
import astropy.units as u

from ssiaat.model import lines as L
from ssiaat.model import sed, shapes
from ssiaat.bandpass_shape_lite import BandpassShapeLite
from ssiaat.wcs_helper import get_wcs_from_shape
from ssiaat.spherex_table import promote_to_stable


def test_registry_bandpass_centers_in_domain():
    bp = BandpassShapeLite()
    for name, lm in L.LINES.items():
        if lm.kind == "bandpass":
            lo, hi = bp.SHAPE_CHEB_DOMAIN[lm.band]
            assert lo <= lm.center <= hi, f"{name} center {lm.center} outside band {lm.band}"


def test_feature_components_resolve():
    for fname, feat in L.FEATURES.items():
        assert len(feat.components) >= 1
        for out_name, line_key in feat.components:
            assert line_key in L.LINES, f"{fname}: {line_key} not in LINES"
        w0, w1 = feat.window
        assert w0 < w1


def test_bandpass_line_unit_peak():
    # The profile is normalized to unit peak on a dense grid of half-width
    # `halfwidth` about the center; evaluated on that same grid the max is 1.
    for name, lm in L.LINES.items():
        if lm.kind != "bandpass":
            continue
        w = np.linspace(lm.center - lm.halfwidth, lm.center + lm.halfwidth, 2001)
        peak = float(L.get_line(name)(w).max())
        assert peak == pytest.approx(1.0, abs=1e-9), f"{name} peak {peak}"


def test_get_line_returns_callable_and_unknown_raises():
    f = L.get_line("feii_1644")
    assert callable(f)
    with pytest.raises(KeyError):
        L.get_line("no_such_line")


def _band1_stable():
    """A small noiseless band-1 stable: image = A*line + clo*cont_lo + chi*cont_hi
    with known per-pixel amplitudes, so a fit must recover them exactly."""
    shape = (16, 16)
    hdr = get_wcs_from_shape(150.0 * u.deg, 2.0 * u.deg, shape).to_header()
    hdr["NAXIS1"], hdr["NAXIS2"] = shape[1], shape[0]

    model = L.get_line_model(lines={"line": L.get_line("s3_9531")},
                             window=(0.85, 1.0))
    line = model.models[0]
    clo, chi = model.cont_models
    wvl = np.linspace(0.86, 0.99, 30)
    pixels = [0, 1, 17, 100, 200, 255]
    amps = {}
    rows = []
    for k, pix in enumerate(pixels):
        A, cl, ch = 2.0 + 0.1 * k, 0.5 + 0.01 * k, 0.3 + 0.02 * k
        amps[pix] = (A, cl, ch)
        image = A * line(wvl) + cl * clo(wvl) + ch * chi(wvl)
        rows.append(pd.DataFrame({"tmpl_ind": pix, "wvl": wvl, "image": image}))
    df = pd.concat(rows, ignore_index=True)
    df["variance"] = 0.01 + 0.0005 * np.arange(len(df))
    return promote_to_stable(df, header=hdr), model, amps


def test_get_line_model_recovers_amplitudes():
    stable, model, amps = _band1_stable()
    fitted = model.least_square_fit(stable)
    for pix, (A, cl, ch) in amps.items():
        assert fitted.coef("line")[pix] == pytest.approx(A, rel=1e-6)
        assert fitted.coef("cont_lo")[pix] == pytest.approx(cl, rel=1e-6)
        assert fitted.coef("cont_hi")[pix] == pytest.approx(ch, rel=1e-6)


def test_feature_spec_matches_get_line_model():
    spec = L.feature_spec("s3")
    assert list(spec["lines"]) == ["s3_9069", "s3_9531"]
    assert spec["window"] == L.FEATURES["s3"].window


def test_to_linespec_bandpass_only():
    ls = L.LINES["he_i"].to_linespec()
    assert ls.name == "he_i" and ls.rest_wavelength == pytest.approx(1.0830)
    with pytest.raises(ValueError):
        L.LINES["br_a"].to_linespec()   # shape line has no LineSpec


def test_manual_profile_kinds_build_from_params():
    # norm="none" reproduces the raw shape exactly; default peak-normalizes.
    from ssiaat.model.lines import LineModel
    from ssiaat.model import shapes
    w = np.linspace(3.0, 4.2, 400)
    ht = LineModel("t_ht", 4, 3.3, kind="hflattop",
                   params=dict(mu0=3.25, mu1=3.35, a=0.7))
    assert np.array_equal(ht.profile(norm="none")(w), shapes.hflattop(3.25, 3.35, a=0.7)(w))
    hl = LineModel("t_hl", 4, 3.07, kind="hflatline",
                   params=dict(mu_center_left=3.07, dmu_left=0.2, a_left=1.3, a_right=0.7))
    assert np.array_equal(hl.profile(norm="none")(w),
                          shapes.hflatline(3.07, 0.2, 1.3, a_right=0.7)(w))
    gz = LineModel("t_g", 5, 4.05, kind="gauss", params=dict(center=4.05, sigma=0.02))
    assert np.array_equal(gz.profile(norm="none")(w), shapes.gauss(4.05, 0.02)(w))


def test_all_lines_unit_peak_by_default():
    for name, lm in L.LINES.items():
        hs = max(getattr(lm, "halfwidth", 0.08), 0.2)
        w = np.linspace(lm.center - hs, lm.center + hs, 20001)
        assert float(np.nanmax(L.get_line(name)(w))) == pytest.approx(1.0, abs=1e-6), name


def test_area_normalization_unit_integral():
    # Manual shapes integrate to 1 under norm="area" (wide grid), gauss analytically.
    for name, lo, hi in [("pah_narrow", 2.6, 4.0), ("br_a", 3.6, 4.5),
                         ("h2o_ice", 1.5, 4.7)]:
        w = np.linspace(lo, hi, 60001)
        assert float(np.trapezoid(L.get_line(name, norm="area")(w), w)) == pytest.approx(1.0, abs=2e-4)
    from ssiaat.model.lines import LineModel
    from ssiaat.model.shapes import gauss
    lm = LineModel("g", 5, 4.05, kind="gauss", params=dict(center=4.05, sigma=0.02))
    assert gauss(4.05, 0.02).area == pytest.approx(0.02 * np.sqrt(2 * np.pi))
    w = np.linspace(3.9, 4.2, 80001)
    assert float(np.trapezoid(lm.profile(norm="area")(w), w)) == pytest.approx(1.0, abs=1e-5)


def test_analytic_areas_match_numeric():
    from ssiaat.model.shapes import hflattop, hflatline

    def num(fn, c, hs=6.0, n=4_000_001):
        w = np.linspace(c - hs, c + hs, n)
        return float(np.trapezoid(fn(w), w))

    # hflattop: (mu1-mu0)/(1-e^{-8a}), exact
    for mu0, mu1, a, c in [(4.0272, 4.0624, 1.55, 4.045), (3.2481, 3.3382, 0.7, 3.29)]:
        ht = hflattop(mu0, mu1, a)
        assert ht.area == pytest.approx((mu1 - mu0) / (1 - np.exp(-8 * a)))
        assert ht.area == pytest.approx(num(ht, c), rel=1e-5)
    # symmetric hflatline: same closed form
    hl = hflatline(3.07, 0.23, 1.0)
    assert hl.area == pytest.approx(num(hl, 3.07), rel=1e-6)
    # asymmetric hflatline: no closed form -> raises (caller falls back to numeric)
    with pytest.raises(NotImplementedError):
        hflatline(3.07, 0.23, 1.3, a_right=0.7).area
    # ...and area-normalization still integrates to 1 via the numeric fallback
    w = np.linspace(1.5, 4.7, 200001)
    assert float(np.trapezoid(L.get_line("h2o_ice", norm="area")(w), w)) == pytest.approx(1.0, abs=1e-3)


def test_bandpass_peak_path_is_division_bit_identical():
    # get_line default (peak) must equal the raw bandpass divided by its own max
    # -- i.e. the pre-normalization behavior, unchanged for bandpass lines.
    from ssiaat.bandpass_shape_lite import BandpassShapeLite
    bp = BandpassShapeLite()
    lm = L.LINES["he_i"]
    hw = lm.halfwidth
    wf = np.linspace(lm.center - hw, lm.center + hw, 2001)
    _, tf = bp(lm.center, array=lm.band, wvl=wf)
    peak = float(tf.max())
    w = np.linspace(0.95, 1.20, 300)
    _, t = bp(lm.center, array=lm.band, wvl=w)
    assert np.array_equal(L.get_line("he_i")(w), t / peak)


def test_unknown_kind_raises():
    from ssiaat.model.lines import LineModel
    with pytest.raises(ValueError):
        LineModel("bad", 4, 3.3, kind="wat").profile()


def test_gauss_shape_peak_and_width():
    from ssiaat.model import shapes
    g = shapes.gauss(4.05, 0.02)
    assert g(np.array([4.05]))[0] == pytest.approx(1.0)         # unit peak at center
    assert g(np.array([4.07]))[0] == pytest.approx(np.exp(-0.5))  # one sigma out
    hwhm = 1.1774100226 * 0.02                                  # sqrt(2 ln2) * sigma
    assert g(np.array([4.05 + hwhm]))[0] == pytest.approx(0.5, abs=1e-6)


def test_retrofit_legacy_lines_shape_preserved():
    """norm="none" must reproduce the old thunk expressions exactly (shape
    unchanged by the retrofit); the default just peak-normalizes on top."""
    from ssiaat.model.shapes import hflattop, hflatline
    d = 0.012 - 0.002
    a = 0.75 + 0.8
    old = {
        "br_a": hflattop(4.0372 - d, 4.0724 - d, a=a),
        "pa_b": hflattop(1.2693 - 0.004, 1.2988 - 0.004, a=a),
        "br_b": hflattop(2.605, 2.647, a=1.5),
        "pah_narrow": hflattop(3.2481, 3.3382, a=0.7),
        "pah_broad": hflattop(3.32, 3.52, a=0.9),
        "h2o_ice": hflatline(3.07, 0.23, 1.3, a_right=0.7),
        "h2_46957": hflattop(4.6947 - 0.017 - d, 4.6947 + 0.017 - d, a=a),
    }
    w = np.linspace(0.8, 5.0, 900)
    for name, fold in old.items():
        assert np.array_equal(fold(w), L.get_line(name, norm="none")(w)), name


def test_fit_hflatline_params_from_real_bandpass():
    pytest.importorskip("spherex_tabular_bandpass")
    from ssiaat.model.shapes import hflatline
    p = L.fit_hflatline_params(4.052, 5)
    assert "amplitude" not in p
    assert set(p) >= {"mu_center_left", "dmu_left", "a_left", "dmu_right", "a_right"}
    prof = hflatline(**p)                       # builds a usable profile
    assert np.all(np.isfinite(prof(np.linspace(3.95, 4.15, 50))))


def test_sed_shapes_still_importable():
    # Shape primitives re-exported from sed (used by conftest, examples, tests).
    assert sed.hflattop is shapes.hflattop
    assert sed.const is shapes.const
    assert sed.hflatline is shapes.hflatline
    assert sed.get_cont_model is shapes.get_cont_model


def test_sed_line_getters_deprecated_but_working():
    for getter, n_out in [("get_br_a", 1), ("get_pa_b", 1),
                          ("get_h2o_ice", 1), ("get_pah", 2)]:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = getattr(sed, getter)()
        assert any(issubclass(w.category, DeprecationWarning) for w in rec), getter
        callables = out if isinstance(out, tuple) else (out,)
        assert len(callables) == n_out
        assert all(callable(c) for c in callables)
