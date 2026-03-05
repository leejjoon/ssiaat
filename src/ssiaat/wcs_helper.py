import numpy as np
import astropy.units as u
from astropy import wcs
from astropy import coordinates as coord

def get_wcs(
    lon_center: u.deg,
    lat_center: u.deg,
    side: u.deg,
    side2: u.Quantity[u.deg] | None = None,
    *,
    pixel_scale: u.deg = 6.15 * u.arcsec,
    frame: str = "icrs",
    projection: str = "TAN",
    return_frame: bool = False,
) -> wcs.WCS | tuple[wcs.WCS, coord.BaseCoordinateFrame]:

    frame_obj = coord.sky_coordinate_parsers._get_frame_class(frame)()  # noqa: SLF001

    # Shape of array to be associated with the WCS
    side_pix = int((side / pixel_scale).to(1)) + 1
    side2_pix = side_pix if side2 is None else int((side2 / pixel_scale).to(1)) + 1
    array_shape = [side2_pix, side_pix]

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

