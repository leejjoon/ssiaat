"""get_wcs: side-based and shape-based forms share one body."""
import astropy.units as u
import numpy as np
import pytest

from ssiaat.wcs_helper import get_wcs, get_wcs_from_shape

PIXEL_SCALE = 6.15 * u.arcsec


def test_side_and_shape_forms_agree():
    side = 30 * PIXEL_SCALE  # -> 31 pixels per get_wcs's side derivation
    by_side = get_wcs(150 * u.deg, 2 * u.deg, side)
    by_shape = get_wcs(150 * u.deg, 2 * u.deg, shape=by_side.array_shape)

    assert by_side.array_shape == by_shape.array_shape
    np.testing.assert_allclose(by_side.wcs.crpix, by_shape.wcs.crpix)
    np.testing.assert_allclose(by_side.wcs.cdelt, by_shape.wcs.cdelt)
    np.testing.assert_allclose(by_side.wcs.crval, by_shape.wcs.crval)


def test_get_wcs_from_shape_delegates():
    a = get_wcs_from_shape(150 * u.deg, 2 * u.deg, (16, 16))
    b = get_wcs(150 * u.deg, 2 * u.deg, shape=(16, 16))
    assert a.to_header() == b.to_header()
    assert a.array_shape == (16, 16)


def test_rectangular_side2():
    w = get_wcs(10 * u.deg, -5 * u.deg, 10 * PIXEL_SCALE, 20 * PIXEL_SCALE)
    ny, nx = w.array_shape
    assert ny == 21 and nx == 11


def test_exactly_one_of_side_or_shape():
    with pytest.raises(ValueError, match="exactly one"):
        get_wcs(0 * u.deg, 0 * u.deg)
    with pytest.raises(ValueError, match="exactly one"):
        get_wcs(0 * u.deg, 0 * u.deg, 1 * u.deg, shape=(4, 4))
    with pytest.raises(ValueError, match="side2"):
        get_wcs(0 * u.deg, 0 * u.deg, side2=1 * u.deg, shape=(4, 4))
