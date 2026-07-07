"""SsiaatConverter: template indexing and itable <-> image round trips."""
import numpy as np
import pandas as pd
import pytest

from ssiaat.spherex_table import Image

from conftest import TMPL_SHAPE

NY, NX = TMPL_SHAPE


def test_tmpl_ind_layout(converter):
    assert converter.tmpl_shape == TMPL_SHAPE
    assert converter.tmpl_ind.shape == TMPL_SHAPE
    assert converter.tmpl_ind_flat.shape == (NY * NX,)
    for y, x in [(0, 0), (0, 5), (3, 0), (7, 11), (NY - 1, NX - 1)]:
        assert converter.tmpl_ind[y, x] == y * NX + x


def test_image_itable_roundtrip(converter):
    rng = np.random.default_rng(0)
    image = rng.normal(size=TMPL_SHAPE)

    itable = converter.image_to_itable(image)
    assert isinstance(itable, pd.Series)
    assert len(itable) == NY * NX
    np.testing.assert_array_equal(itable.index.values, converter.tmpl_ind_flat)

    image2 = converter.itable_to_image(itable)
    assert isinstance(image2, Image)
    assert image2._ssiaat_converter is converter
    np.testing.assert_array_equal(np.asarray(image2), image)


def test_roundtrip_preserves_nan(converter):
    image = np.ones(TMPL_SHAPE)
    image[3, 4] = np.nan
    image2 = converter.itable_to_image(converter.image_to_itable(image))
    assert np.isnan(image2[3, 4])
    assert np.nansum(image2) == NY * NX - 1


def test_partial_itable_fills_nan(converter):
    # An itable covering a subset of pixels: the missing ones must come
    # back as NaN in the image (reindex semantics).
    itable = pd.Series([1.0, 2.0, 3.0], index=[0, 17, 255])
    image = converter.itable_to_image(itable)
    assert image[0, 0] == 1.0
    assert image[1, 1] == 2.0
    assert image[NY - 1, NX - 1] == 3.0
    assert np.isnan(image).sum() == NY * NX - 3


def test_image_to_itable_mask(converter):
    image = np.arange(NY * NX, dtype=float).reshape(TMPL_SHAPE)
    mask = np.zeros(TMPL_SHAPE, dtype=bool)
    mask[2, 3] = True
    mask[9, 0] = True

    itable = converter.image_to_itable(image, mask=mask)
    assert len(itable) == 2
    assert itable[2 * NX + 3] == image[2, 3]
    assert itable[9 * NX + 0] == image[9, 0]


def test_big_endian_input_converted_to_native(converter):
    # FITS data is big-endian; pandas needs native byte order.
    image = np.arange(NY * NX, dtype=">f8").reshape(TMPL_SHAPE)
    itable = converter.image_to_itable(image)
    assert itable.dtype.isnative
    np.testing.assert_array_equal(itable.values, np.ravel(image).astype(float))


def test_image_slice_keeps_converter(converter):
    image = converter.itable_to_image(pd.Series([1.0], index=[0]))
    sub = image[2:5, 1:4]
    assert isinstance(sub, Image)
    assert sub._ssiaat_converter is converter
