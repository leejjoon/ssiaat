from collections.abc import Callable, Iterable, Mapping
import time
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.nddata import NDData

# from spherex_utils import utils

import logging
import os

from reproject import reproject_adaptive, reproject_interp
from reproject.hips.utils import tile_header

from astropy.wcs import WCS
from astropy.coordinates import Galactic

from spherex_tabular_bandpass import Tabular_Bandpass

#from spherex_utils.utils.mosaic_utils import get_flagval, DEFAULT_FLAGS
from .flags import get_flagval, DEFAULT_FLAGS
# DEFAULT_FLAGS: tuple[str, ...] = ("ALL", "-FULLSAMPLE", "-SOURCE")

def get_metadata_from_filename(fn):
    from pathlib import Path
    fn = Path(fn)
    pipe_run = fn.name.split(".")[0].split("_spx_")[-1]

    return dict(pipe_run=pipe_run)


def ingest_hdul(hdulist: fits.HDUList, *,
                flags: Iterable[str] = DEFAULT_FLAGS,
                process_variance: bool = True) -> bool | list:
    """Prepare given hdul and return ingested image data for mosaicking.

    Parameters
    ----------
    hdul: HDList
        fits HDUlist object.

    Returns
    -------
    list | bool
        list of nddata if successful. True is check_only is true and everything is okay. False for any issue.

    """

    # This is stripped down version of ingest_hdul. It does not check overwrap with the target template so that
    # it can be called without dependency on taget.

    ACTIVE_AREA_SHAPE = (2040, 2040)

    flag = get_flagval(*flags)

    # Check the argument and get image, WCS, and metadata from it
    input_image = hdulist.filename()

    # Get HDU of spectral image
    if hdulist[0].data is not None:
        image_hdu = hdulist[0]
    elif "IMAGE" in hdulist:
        image_hdu = hdulist["IMAGE"]
    else:
        msg = f"failed to find an HDU of image data from '{input_image}'"
        raise ValueError(msg)

    # Check presence of "L2DQAFLG" keyword in the image HDU
    # If not present, the FITS file is not of the Level-2 spectral image and
    # we stop and return False.
    image_hdr = image_hdu.header
    if "L2DQAFLG" not in image_hdr:
        msg = "Input image is not Level-2 spectral image"
        raise ValueError(msg)

    # Check presence of "FLAGS" extension
    if "FLAGS" not in hdulist:
        msg = f"failed to find 'FLAGS' extension from '{input_image}'"
        raise ValueError(msg)

    # Check presence of "VARIANCE" extension
    if process_variance and "VARIANCE" not in hdulist:
        msg = f"failed to find 'VARIANCE' extension from '{input_image}'"
        raise ValueError(msg)

    # Get values from image HDU header
    det_id = image_hdr["DETECTOR"]
    image_wcs = WCS(header=image_hdr)

    # Load data required
    image = image_hdu.data
    flag_image = hdulist["FLAGS"].data

    # Check image shapes
    if not (
        image.shape
        == flag_image.shape
        == image_wcs.array_shape
        == ACTIVE_AREA_SHAPE
    ):
        msg = "inconsistent shapes of input images"
        raise ValueError(msg)

    if process_variance:
        var_img = hdulist["VARIANCE"].data
        # Check image shapes including variance
        if var_img.shape != ACTIVE_AREA_SHAPE:
            msg = "inconsistent shape of variance image"
            raise ValueError(msg)

    spch_image_wcs = image_wcs # [self.image_slice]

    mask = flag_image & flag
    spch_image = np.ma.masked_array(
        data=image, mask=mask,
    ).filled(np.nan)

    nddata_list = [NDData(spch_image, wcs=spch_image_wcs)]

    if process_variance:
        spch_var_image = np.ma.masked_array(
            data=var_img, mask=mask,
        ).filled(np.nan)
        nddata_list.append(NDData(spch_var_image, wcs=spch_image_wcs))

    return nddata_list

def check_overwrapp():
    # FIXME this is copied from ingest_image function. It need to be fixed.
    # Check whether the spectral channel image overlaps the coadd WCS
    if not wcs_overlapped(
        self.coadd_wcs, spch_image_wcs, mask=self.mask_sliced,
    ):
        self._logger.warning(
            "Spectral channel of '%s' does not overlap the target region.",
            input_image,
        )
        return False



class SphxReprojector:
    DATASHAPE = (2048, 2048)

    @classmethod
    def get_ind_image(cls, input_image):
        """
        convert x,y indices of the input SPHEREx spectral image (which should be usually 2040x2040) to a single integer. For simplicity, we assume 2048x2048 shape.
        """
        (ny, nx) = input_image.shape
        assert (ny <= 2048) and (nx <= 2048)
        ind = np.sum(np.indices((ny, nx)) * np.array([2048, 1]).reshape((2, 1, 1)),
                          axis=0, dtype="float32")
        return ind

    @classmethod
    def get_src_yx(cls, ind_array):
        # src_y, src_x = np.divmod(ind_array, 2048)
        src_y = ind_array >> 11
        src_x = ind_array & 2047

        return src_y, src_x

    def __init__(self, input_hdul, *,
                 flags=None, aux_metadata=None):
        # self.PROJNAME = PROJNAME
        # self.plan = plan
        # self.root = root


        self.metadata = dict()

        header = input_hdul[1].header

        self.metadata["expidn"] = header["EXPIDN"]
        self.band = self.metadata["band"] = header["DETECTOR"]

        if aux_metadata is not None:
            for k in self.metadata:
                if k in aux_metadata:
                    raise ValueError(f"aux_metadata should not contain '{k}' key.")
            self.metadata.update(aux_metadata)

        self.flags = DEFAULT_FLAGS if flags is None else flags
        nddata_list = ingest_hdul(input_hdul, flags=self.flags, process_variance=True)
        if isinstance(nddata_list, bool):
            raise RuntimeError()

        self.nddata = np.array([nd.data for nd in nddata_list])
        self.wcs_in = WCS(input_hdul["IMAGE"].header)

        self.ind = self.get_ind_image(input_hdul["IMAGE"].data)

        # self.bandpass_model = utils.mosaic_utils.PixelToCentralWavelengthUsingWCS()
        self.bandpass_model = Tabular_Bandpass(mode="tvac4")

    def process_single(self, output_wcs_tmpl):

        # # mosaic = utils.SpectralChannelMosaic(
        # mosaic = utils.WavelengthRangeMosaic(
        #     output_wcs_tmpl,
        #     self.band,
        #     wavelength_range=(0, 10), # wide enough wavelength range
        #     flags=self.flags, # .split(","),
        #     # bandpass_model=bandpass,
        #     process_variance=True,
        #     bandpass_model=self.bandpass_model,
        # )

        # wcs_tmpl = mosaic.coadd_wcs
        # shape_out = mosaic.coadd_wcs.array_shape

        wcs_tmpl = output_wcs_tmpl
        shape_out = output_wcs_tmpl.array_shape

        # nddata_list = mosaic.ingest_hdul(hdul)
        # if nddata_list is False:
        #     processed.append((hid, "empty", ""))
        #     continue

        array_out, footprint = reproject_adaptive((self.nddata, self.wcs_in), wcs_tmpl,
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

        outputarray = np.zeros(array_out[0].data.shape, dtype="float32")

        ind_out, footprint = reproject_interp((self.ind, self.wcs_in), wcs_tmpl,
                                              order=0,
                                              output_array=outputarray,
                                              shape_out=shape_out,
                                              # # shape_out=N,
                                              # bad_value_mode="ignore",
                                              parallel=False,
                                              )
        ind_out[footprint == 0] = -1

        out_header = wcs_tmpl.to_header()
        out_hdul = fits.HDUList(
            [fits.PrimaryHDU(data=array_out[0], header=out_header),
             fits.ImageHDU(data=array_out[1], header=out_header),
             fits.ImageHDU(data=ind_out.astype("int32"), header=out_header)]
        )


        return out_hdul

    def hdul_to_pandas(self, hdul):
        """

        hdul : PrimaryHDU-Image, 2nd-Variance, 3rd-tmpl_ind
        """
        # band, channel = row[["band", "channel"]]
        # outdir = Path("intermediate/level2") / f"b{band}_c{channel:03d}"
        # fn = outdir / Path(row["filepath"]).with_suffix(".reproj.fits").name

        # hdul = fits.open(fn)

        # hdul[0].data.shape
        tmpl_shape = hdul[0].data.shape


        # ind = np.sum(np.indices((2048, 2048)) * np.array([2048, 1]).reshape((2, 1, 1)),
        #              axis=0, dtype="int32")
        # y, x = np.divmod(ind, 2048)


        tmpl_ind = np.sum(np.indices(tmpl_shape) * np.array([tmpl_shape[-1], 1]).reshape((2, 1, 1)),
                          axis=0, dtype="int32")

        msk = np.isfinite(hdul[0].data)
        # src_y, src_x = np.divmod(hdul[2].data[msk], 2048)
        src_y, src_x = self.get_src_yx(hdul[2].data[msk])
        wvl = self.bandpass_model(src_x, src_y,
                                  array=int(self.band), central_bandpass_only=True)[0]

        # iy, ix = np.indices((2048, 2048))
        # wvl = bp2(ix, iy, central_bandpass_only=True)

        df = pd.DataFrame(dict(
            tmpl_ind=tmpl_ind[msk].copy(),
            image=hdul[0].data.astype("float32")[msk],
            variance=hdul[1].data.astype("float32")[msk],
            #srcind=hdul[2].data[msk]
            src_x=src_x,
            src_y=src_y,
            wvl=wvl,
            **self.metadata
        ))

        return df


def main():
    fn = "level2_2025W24_1A_0405_2D1_spx_l2b-v19-2025-252.fits"
    hdul = fits.open(fn)

    aux_metadata = get_metadata_from_filename(fn)

    reprojector = SphxReprojector(hdul, aux_metadata=aux_metadata)

    header = fits.open("eso_244_template.fits")[0].header
    output_wcs_tmpl = WCS(header)
    out_hdul = reprojector.process_single(output_wcs_tmpl)
    df = reprojector.hdul_to_pandas(out_hdul)
    #out_hdul.writeto("a.fits", overwrite=True)
    df.to_parquet("a.parquet")
    print(df.columns)

if __name__ == '__main__':
    main()
