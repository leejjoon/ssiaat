from collections import namedtuple
from importlib.resources import files

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .tabular_bandpass_lite import Tabular_Bandpass_Lite

# Directory holding the per-band zodi scaling tables, resolved as a package
# resource so it works regardless of the current working directory and even
# when the package is imported from a zip archive.
_DATA_DIR = files(__package__) / "ZodiScaling70Channels_0329"

# Detector array size (pixels). Used to build the per-pixel wavelength grid.
_DETECTOR_SHAPE = (2040, 2040)

# Conversion factor for ZodiOffset: the table is tabulated in nW/m^2/sr and is
# rescaled to the working units via (offset / _OFFSET_NORM * wavelength).
# See https://gemini.google.com/share/237c0b9c8fa1
_OFFSET_NORM = 2997.9


class ZodiCorrection:
    # ``scale`` and ``offset`` interp1d objects over wavelength_um.
    Intp = namedtuple("Intp", ["scale", "offset"])
    # Correction maps for a band: ``scale`` and ``offset`` are each a
    # detector-shaped array, evaluated at every pixel's central wavelength.
    Correction = namedtuple("Correction", ["wavelength", "scale", "offset"])

    def __init__(self):
        self.tblite = Tabular_Bandpass_Lite()
        self._zodi_correction_intp = {}
        self._zodi_correction_map = {}
        self._iy, self._ix = np.indices(_DETECTOR_SHAPE)

    def load_band_intp(self, band):
        csvfile = _DATA_DIR / f"ZodiScaling_AllSky_70Ch_Band{band}.csv"

        with csvfile.open() as f:
            df = pd.read_csv(f)

        z_scale = interp1d(
            df["wavelength_um"],
            df["ZodiScale"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        z_offset = interp1d(
            df["wavelength_um"],
            df["ZodiOffset_nWm2sr"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self._zodi_correction_intp[band] = self.Intp(z_scale, z_offset)


    def load_band_map(self, band):
        if band not in self._zodi_correction_intp:
            self.load_band_intp(band)

        # Central wavelength at every detector pixel.
        w, _ = self.tblite(self._ix, self._iy, array=band, central_bandpass_only=True)

        z = self._zodi_correction_intp[band]
        z_scale = z.scale(w)
        z_offset = z.offset(w) / _OFFSET_NORM * w

        self._zodi_correction_map[band] = self.Correction(w, z_scale, z_offset)

    def get_correction_intp(self, band):
        if band not in self._zodi_correction_intp:
            self.load_band_intp(band)

        return self._zodi_correction_intp[band]

    def get_correction_map(self, band):
        if band not in self._zodi_correction_map:
            self.load_band_map(band)

        return self._zodi_correction_map[band]

    def get_corrected_zodi(self, band, zodi_original):
        z_map = self.get_correction_map(band)

        return z_map.scale * zodi_original + z_map.offset
