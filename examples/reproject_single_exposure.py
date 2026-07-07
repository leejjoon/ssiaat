"""Reproject a single local L2 exposure onto a template and save a parquet.

Moved out of ssiaat.reproj; expects the L2 file and eso_244_template.fits
in the working directory.
"""
from astropy.io import fits
from astropy.wcs import WCS

from ssiaat.reproj import SphxReprojector, get_metadata_from_filename


def main():
    fn = "level2_2025W24_1A_0405_2D1_spx_l2b-v19-2025-252.fits"
    hdul = fits.open(fn)

    aux_metadata = get_metadata_from_filename(fn)

    reprojector = SphxReprojector(hdul, aux_metadata=aux_metadata)

    header = fits.open("eso_244_template.fits")[0].header
    output_wcs_tmpl = WCS(header)
    out_hdul = reprojector.process_single(output_wcs_tmpl)
    df = reprojector.hdul_to_pandas(out_hdul)
    df.to_parquet("a.parquet")
    print(df.columns)


if __name__ == '__main__':
    main()
