"""make_pixel_index / get_src_yx (shared by converter and reprojector)."""
import numpy as np

from ssiaat.indexing import make_pixel_index, get_src_yx, SRC_STRIDE


def test_template_index_layout():
    ind = make_pixel_index((4, 7))
    assert ind.dtype == np.int32
    assert ind.shape == (4, 7)
    for y, x in [(0, 0), (0, 6), (2, 3), (3, 6)]:
        assert ind[y, x] == y * 7 + x


def test_source_index_layout():
    ind = make_pixel_index((5, 5), stride=SRC_STRIDE, dtype="float32")
    assert ind.dtype == np.float32
    assert ind[3, 4] == 3 * 2048 + 4


def test_src_yx_roundtrip():
    y, x = np.mgrid[0:2040:97, 0:2040:101]
    packed = (y * SRC_STRIDE + x).astype("int32")
    src_y, src_x = get_src_yx(packed)
    np.testing.assert_array_equal(src_y, y)
    np.testing.assert_array_equal(src_x, x)
