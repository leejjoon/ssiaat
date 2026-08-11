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
