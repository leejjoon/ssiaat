"""Tabular_Bandpass_Lite central wavelengths.

The pinned values freeze the current Chebyshev evaluation; the optional
comparison against the full spherex_tabular_bandpass package (the source
of the coefficients) runs only where that package is installed.
"""
import importlib.util

import numpy as np
import pytest

from ssiaat.tabular_bandpass_lite import Tabular_Bandpass_Lite

PIX = np.array([100, 1024, 1900])  # (ix == iy) probe positions

# Computed with the current Lite model, 2026-07-07.
PINNED_WVL = {
    1: [1.10263192, 0.90475643, 0.76443925],
    2: [1.63074591, 1.33613130, 1.12815803],
    3: [2.39828339, 1.98103880, 1.67816633],
    4: [3.77031449, 3.00608856, 2.48108694],
    5: [4.39817170, 4.09350042, 3.84623303],
    6: [4.98129774, 4.68545541, 4.44644065],
}


@pytest.mark.parametrize("band", range(1, 7))
def test_pinned_central_wavelengths(band):
    bp = Tabular_Bandpass_Lite()
    wvl, trans = bp(PIX, PIX, array=band, central_bandpass_only=True)
    np.testing.assert_allclose(wvl, PINNED_WVL[band], rtol=1e-7)
    np.testing.assert_array_equal(trans, np.ones_like(wvl))


def test_requires_central_bandpass_only():
    bp = Tabular_Bandpass_Lite()
    with pytest.raises(RuntimeError):
        bp(PIX, PIX, array=1)


@pytest.mark.skipif(
    importlib.util.find_spec("spherex_tabular_bandpass") is None,
    reason="spherex_tabular_bandpass not installed")
@pytest.mark.parametrize("band", range(1, 7))
def test_matches_full_bandpass_model(band):
    from spherex_tabular_bandpass import Tabular_Bandpass

    tb = Tabular_Bandpass()
    tblite = Tabular_Bandpass_Lite()

    # coarse grid instead of the full 2040x2040
    iy, ix = np.mgrid[0:2040:64, 0:2040:64]
    w_orig, _ = tb(ix, iy, array=band, central_bandpass_only=True)
    w_lite, _ = tblite(ix, iy, array=band, central_bandpass_only=True)
    np.testing.assert_allclose(w_orig, w_lite)
