"""Flat pixel-index images and their decoding.

Two kinds of packed pixel indices are used in ssiaat:

- **template index** (``tmpl_ind``): ``y * width + x`` over the template
  grid, with the stride equal to the template width. Integer, unique per
  template pixel; this is the stable/itable index.
- **source index** (``src_ind``): position on the 2048-wide SPHEREx
  detector, packed with a fixed stride of 2048 (``y * 2048 + x``) and
  stored as float32 so it survives reprojection. Decode with
  :func:`get_src_yx`, whose bit operations (``>> 11`` / ``& 2047``)
  assume exactly that stride.
"""
import numpy as np

SRC_STRIDE = 2048


def make_pixel_index(shape, stride=None, dtype="int32"):
    """Return a `shape`-shaped image of flat pixel indices ``y*stride + x``.

    ``stride`` defaults to ``shape[-1]`` (the template-index convention).
    For detector source indices use ``stride=SRC_STRIDE, dtype="float32"``.
    """
    if stride is None:
        stride = shape[-1]
    return np.sum(np.indices(shape) * np.array([stride, 1]).reshape((2, 1, 1)),
                  axis=0, dtype=dtype)


def get_src_yx(ind_array):
    """Decode a source index (stride 2048) back into (src_y, src_x)."""
    ind_array = np.asarray(ind_array)
    src_y = ind_array >> 11    # // 2048
    src_x = ind_array & 2047   # %  2048

    return src_y, src_x
