import numpy as np
import hashlib

import astropy.units as u
from astropy import wcs
from astropy import coordinates as coord
from astropy.io import fits

def get_wcs(
    lon_center: u.deg,
    lat_center: u.deg,
    side: u.Quantity[u.deg] | None = None,
    side2: u.Quantity[u.deg] | None = None,
    *,
    shape: tuple | None = None,
    pixel_scale: u.deg = 6.15 * u.arcsec,
    frame: str = "icrs",
    projection: str = "TAN",
    return_frame: bool = False,
) -> wcs.WCS | tuple[wcs.WCS, coord.BaseCoordinateFrame]:
    """Build a celestial WCS centered on (lon_center, lat_center).

    Give exactly one of `side` (angular size; `side2` for a rectangular
    field) or `shape` (pixel array shape, (ny, nx)).
    """
    if (side is None) == (shape is None):
        raise ValueError("give exactly one of `side` or `shape`.")

    frame_obj = coord.sky_coordinate_parsers._get_frame_class(frame)()  # noqa: SLF001

    # Shape of array to be associated with the WCS
    if shape is None:
        side_pix = int((side / pixel_scale).to(1)) + 1
        side2_pix = side_pix if side2 is None else int((side2 / pixel_scale).to(1)) + 1
        array_shape = [side2_pix, side_pix]
    else:
        if side2 is not None:
            raise ValueError("`side2` cannot be combined with `shape`.")
        array_shape = shape

    # Build the output WCS
    out_wcs = wcs.utils.celestial_frame_to_wcs(frame_obj, projection=projection)
    out_wcs.array_shape = array_shape
    out_wcs.wcs.crpix = np.array(out_wcs.array_shape[::-1]) / 2.0 + 0.5
    pixel_scale_deg = pixel_scale.to_value(u.deg)
    out_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    out_wcs.wcs.crval = [lon_center.to_value(u.deg), lat_center.to_value(u.deg)]
    out_wcs.wcs.crota = [0.0, 0.0]

    if return_frame:
        return out_wcs, frame_obj
    return out_wcs

def get_wcs_from_shape(
    lon_center: u.deg,
    lat_center: u.deg,
    array_shape: tuple,
    **kwargs,
) -> wcs.WCS | tuple[wcs.WCS, coord.BaseCoordinateFrame]:
    """get_wcs with a pixel array shape instead of an angular size."""
    return get_wcs(lon_center, lat_center, shape=array_shape, **kwargs)


class TemplateHeaderCards:

    HASH_CONSTRUCTOR = hashlib.sha256

    def __init__(self, header_image_list, hash=None):
        self.header_image_list = header_image_list
        self.hash = hash if hash is not None else self.get_sha256(header_image_list)

    @classmethod
    def get_sha256(cls, header_image_list):
        hash = cls.HASH_CONSTRUCTOR()
        for image in header_image_list:
            hash.update(image.encode())

        return hash.hexdigest()

    def __eq__(self, other):
        if not isinstance(other, TemplateHeaderCards):
            return NotImplemented
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    @classmethod
    def from_header(cls, header):
        return cls([c.image for c in header.cards])

    @classmethod
    def from_dataframe(cls, df):
        # to_header(cls, obj):
        hash = df.attrs.get("ssiaat_template_header_hash", None)
        return cls(df.attrs.get("ssiaat_template_header"), hash=hash)

    def update_dataframe(self, df):
        # s = header.tostring()
        df.attrs["ssiaat_template_header"] = self.header_image_list
        df.attrs["ssiaat_template_header_hash"] = self.hash

    def to_header(self):
        header = fits.Header([fits.Card.fromstring(s) for s in self.header_image_list])
        return header

    @classmethod
    def retrieve_header_from_dataframe(cls, df):
        return cls.from_dataframe(df).to_header()
