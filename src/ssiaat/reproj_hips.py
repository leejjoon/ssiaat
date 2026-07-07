import time
from pathlib import Path
from io import BytesIO

from collections.abc import Callable, Iterable, Mapping

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.nddata import NDData

import logging
import os

from reproject import reproject_adaptive, reproject_interp
from reproject.hips.utils import tile_header

from astropy.wcs import WCS
from astropy.coordinates import Galactic

from .reproj import ingest_hdul
from .flags import DEFAULT_FLAGS
from .indexing import make_pixel_index, SRC_STRIDE
# from spherex_tabular_bandpass import Tabular_Bandpass
from .tabular_bandpass_lite import Tabular_Bandpass_Lite

def reproject_to_hips_tile(array_in, wcs_in, header_out,
                           reproject_function=reproject_adaptive, **kwargs):

    """

    header_out: a Header instance of a tuple of headers. A tuple is allowed as `tile_header` can return a tuple of headers.
    """

    # reproject_function = reproject_adaptive
    # kwargs = dict()
    wcs_in_copy = wcs_in.deepcopy()

    if isinstance(header_out, tuple):

        array_out1, footprint1 = reproject_function(
            (array_in, wcs_in_copy), header_out[0], **kwargs
        )
        array_out2, footprint2 = reproject_function(
            (array_in, wcs_in_copy), header_out[1], **kwargs
        )
        with np.errstate(invalid="ignore"):
            array_out = (
                np.nan_to_num(array_out1) * footprint1 + np.nan_to_num(array_out2) * footprint2
            ) / (footprint1 + footprint2)
            footprint = (footprint1 + footprint2) / 2
        header_out = header_out[0]
    else:
        array_out, footprint = reproject_function((array_in, wcs_in_copy), header_out, **kwargs)

    return array_out, footprint


class SphxHpxProcess:
    def __init__(self, PROJNAME, plan, root, band, hdul, frame_name="galactic",
                 flags=None, bandpass_model=None):
        self.PROJNAME = PROJNAME
        self.plan = plan
        self.root = root
        self.band = int(band)

        # self.frame_name
        if frame_name == "galactic":
            self.frame = Galactic()
        else:
            raise ValueError()

        self.flags = DEFAULT_FLAGS if flags is None else flags
        nddata_list = ingest_hdul(hdul, flags=self.flags, process_variance=True)
        if isinstance(nddata_list, bool):
            raise RuntimeError()

        self.nddata = np.array([nd.data for nd in nddata_list])
        self.wcs_in = WCS(hdul["IMAGE"].header)

        self.ind = make_pixel_index(hdul["IMAGE"].data.shape,
                                    stride=SRC_STRIDE, dtype="float32")

        # self.bandpass_model = utils.mosaic_utils.PixelToCentralWavelengthUsingWCS()
        self.bandpass_model = (Tabular_Bandpass_Lite() if bandpass_model is None
                               else bandpass_model)

    def process_hid(self, level, tile_size, hid):
        header = tile_header(level=level, index=hid, frame=self.frame,
                             tile_dims=tile_size)

        shape_out = (tile_size, tile_size)
        # wcs_tmpl = WCS(header)

        # # mosaic = utils.SpectralChannelMosaic(
        # mosaic = utils.WavelengthRangeMosaic(
        #     wcs_tmpl,
        #     self.band,
        #     wavelength_range=(0, 10), # wide enough wavelength range
        #     flags=self.flags, # .split(","),
        #     # bandpass_model=bandpass,
        #     process_variance=True,
        #     bandpass_model=self.bandpass_model,
        # )

        # wcs_tmpl = mosaic.coadd_wcs
        # shape_out = mosaic.coadd_wcs.array_shape

        # nddata_list = mosaic.ingest_hdul(hdul)
        # if nddata_list is False:
        #     processed.append((hid, "empty", ""))
        #     continue

        array_out, footprint = reproject_to_hips_tile(self.nddata, self.wcs_in, header,
                                                      reproject_function=reproject_adaptive,
                                                      shape_out=shape_out,
                                                      bad_value_mode="ignore",
                                                      parallel=False,
                                                      )
        if np.all(footprint == 0):
            return None

        # tile_format IS fits
        array_out[footprint == 0] = np.nan

        # FIXME: having dtype of int32 fails when reproj with nan issue.
        # So we use float32 then convert it to int32 while saving

        # outputarray = np.zeros(array_out[0].data.shape, dtype="float32")

        ind_out, footprint = reproject_to_hips_tile(self.ind, self.wcs_in, header,
                                                    reproject_function=reproject_interp,
                                                    order=0,
                                                    # output_array=outputarray,
                                                    shape_out=shape_out,
                                                    # # shape_out=N,
                                                    # bad_value_mode="ignore",
                                                    parallel=False,
                                                    )
        ind_out[footprint == 0] = -1


        if isinstance(header, tuple):
            header1 = header[0]
        else:
            header1 = header

        out_hdul = fits.HDUList(
            [fits.PrimaryHDU(data=array_out[0], header=header1),
             fits.ImageHDU(data=array_out[1], header=header1),
             fits.ImageHDU(data=ind_out.astype("int32"), header=header1)]
        )


        return out_hdul


    def process_hid_with_status(self, level, tile_size, hid):
        try:
            hdul_out = self.process_hid(level, tile_size, hid)
        except Exception as e:
            # raise e
            return (e, None)

        if hdul_out is None: # no overlap
            return ("no overlap", None)
        else:
            return  ("success", hdul_out)

    def process_hid_list(self, level, tile_size, hid_list):
        processed = []
        for hid in hid_list:
            try:
                hdul_out = self.process_hid(level, tile_size, hid)
            except Exception as e:
                # raise e
                processed.append((hid, "error in processing", e))
                continue

            if hdul_out is None: # no overlap
                processed.append((hid, "no overlap", None))
                continue

            processed.append((hid, "success", hdul_out))

        return processed

    # def finalize_all(self, level, tile_size, processed):
    #     processed_new = []
    #     for hid, code, hdul_out in processed:
    #         if code != "success":
    #             processed_new.append((hid, code, []))
    #             continue
    #         try:
    #             fnout = self.finalize_output(level, tile_size, hid, hdul_out)
    #             processed_new.append((hid, code, fnout))
    #         except Exception as e:
    #             # raise e
    #             processed_new.append((hid, "error in finalizing", str(e)))
    #             continue

    #     return processed_new

    def finalize_all_local(self, level, tile_size, processed):
        processed_new = []
        for hid, code, hdul_out in processed:
            if code != "success":
                processed_new.append((hid, code, []))
                continue
            try:
                fnout = self._finalize_output_local_save(level, tile_size, hid, hdul_out)
                processed_new.append((hid, code, fnout))
            except Exception as e:
                raise e
                # processed_new.append((hid, "error in finalizing", str(e)))
                # continue

        return processed_new


    # def finalize_output(self, level, tile_size, hid, hdul_out):
    #     # fnout = self._finalize_output_local_save(level, tile_size, hid, hdul_out)
    #     fnout = self._finalize_output_s3(level, tile_size, hid, hdul_out)
    #     return fnout

    def get_outname(self, level, tile_size, hid):
        return f"hpx{level}_{hid}_{tile_size}_sph_{self.plan}_{self.root}.fits"

    # def _finalize_output_s3(self, level, tile_size, hid, hdul_output):

    #     outname = self.get_outname(level, tile_size, hid)

    #     fout = BytesIO()
    #     # tmpl_header = wcs_tmpl.to_header()
    #     hdul_output.writeto(fout)
    #     # bucket = "sphx-reproj-ouput"
    #     upload_data(fout.getvalue(), f"{self.PROJNAME}/{self.band}/{level}_{tile_size}/{hid}/{outname}")
    #     return outname

    def _finalize_output_local_save(self, level, tile_size, hid, hdul_output):
        outname = self.get_outname(level, tile_size, hid)

        # outdir = Path(f"{PROJNAME}/{self.band}/{level}_{tile_size}/{hid}")
        outdir = Path(f"{self.PROJNAME}/{self.band}/{level}_{tile_size}/{hid}")
        outdir.mkdir(parents=True, exist_ok=True)
        hdul_output.writeto(outdir / outname, overwrite=True)
        print("write", outdir / outname)
        return outname


def main():
    from astropy.io import fits

    fn = 'level2_2025W37_1A_0290_1D1_spx_l2b-v20-2025-272.fits'
    level = 6
    tile_size = 512

    hpx6 = 25557

    PROJNAME = ""
    band = 1
    plan = "2025W37_1A"
    root = "0290_1D1"

    hdul = fits.open(fn)

    processor = SphxHpxProcess(PROJNAME, plan, root, band, hdul, frame_name="galactic",
                               flags=None)
    # out_hdul = processor.process_hid(level, tile_size, hpx6)
    status, out_hdul = processor.process_hid_with_status(level, tile_size, hpx6)

    if status == "success":
        outname = processor.get_outname(level, tile_size, hpx6)

        out_hdul.writeto(outname, overwrite=True)
    else:
        print(status)



if __name__ == '__main__':
    main()
