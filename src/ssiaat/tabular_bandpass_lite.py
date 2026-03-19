"""
This is a lite version of spherex Tabular_Bandpass class. It only support tvac4
centeral wavelength.
"""

import numpy as np
from numpy import array

from numpy.polynomial.chebyshev import chebval2d


class Tabular_Bandpass_Lite:
    center_cheb_coeff = {1: array([[ 1.12631682e+00, -2.26499509e-04,  9.71235213e-09],
                                   [-2.33855663e-05,  2.12292594e-09, -1.35000705e-14],
                                   [ 5.77435561e-09, -5.15529437e-13,  2.72351287e-18]]),
                         2: array([[ 1.66610096e+00, -3.37942152e-04,  1.47833261e-08],
                                   [-3.51782291e-05,  3.67641046e-09, -1.47584652e-13],
                                   [ 8.67717441e-09, -8.62156867e-13,  2.64899920e-17]]),
                         3: array([[ 2.44707264e+00, -4.69246482e-04,  1.74748618e-08],
                                   [-4.37181573e-05, -8.10259394e-09,  2.37997039e-12],
                                   [ 1.08829079e-08,  1.89370962e-12, -5.67543311e-16]]),
                         4: array([[ 3.86168061e+00, -8.79274112e-04,  4.04277885e-08],
                                   [-8.18569517e-05, -2.73199118e-08,  1.15031861e-11],
                                   [ 2.04993979e-08,  6.72488511e-12, -2.83609097e-15]]),
                         5: array([[ 4.43223103e+00, -3.24670180e-04,  4.43103918e-09],
                                   [-3.23899284e-05, -1.75047130e-09, -4.22289040e-14],
                                   [ 8.25840828e-09,  4.30861434e-13,  1.45186664e-17]]),
                         6: array([[ 5.01451176e+00, -3.15619000e-04,  4.71321904e-09],
                                   [-3.27718553e-05, -6.15931232e-10, -1.94212612e-13],
                                   [ 8.15619587e-09,  1.76940796e-13,  4.64533796e-17]])}

    def __init__(
        self,
    ):

        # The pixel offset to be subtracted from the input pixel index to
        # get the correct Chebyshev polynomial value. This is needed
        # because SSDC CENTER files are 2040x2040 (reference pixels
        # removed) and thus the Chebyshev fit is only for the central
        # 2040x2040 pixels.
        self.pixoffset = 4

    def __call__(
        self, ix, iy, array=2, sparse=False, central_bandpass_only=False, norm=True
    ):
        if not central_bandpass_only:
            raise RuntimeError("The lite version of Tabular_Bandpass only support center_bandpas_only mode.")

        ix = np.around(np.asarray(ix)).astype(int)
        iy = np.around(np.asarray(iy)).astype(int)
        assert ix.ndim == iy.ndim, "different dimension between `ix` and `iy`"
        assert ix.shape == iy.shape, "inconsistent shape between `ix` and `iy`"

        wavelength = chebval2d(
            ix - self.pixoffset,
            iy - self.pixoffset,
            self.center_cheb_coeff[array],
        )
        # around to get the value at the nearest pixel (pixel center).
        transmission = np.ones_like(wavelength)[()]

        return wavelength, transmission

def test():
    from spherex_tabular_bandpass import Tabular_Bandpass
    tb = Tabular_Bandpass()
    tblite = Tabular_Bandpass_Lite()
    
    iy, ix = np.indices((2040, 2040))

    for band in range(1, 7):
        w_orig, _ = tb(ix, iy, array=band, central_bandpass_only=True)
        w_lite, _ = tblite(ix, iy, array=band, central_bandpass_only=True)
        assert np.allclose(w_orig - w_lite, 0)
