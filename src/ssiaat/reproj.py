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

from .wcs_helper import TemplateHeaderCards
from .tabular_bandpass_lite import Tabular_Bandpass_Lite
from .indexing import make_pixel_index, get_src_yx, SRC_STRIDE

#from spherex_utils.utils.mosaic_utils import get_flagval, DEFAULT_FLAGS
from .flags import get_flagval, DEFAULT_FLAGS
# DEFAULT_FLAGS: tuple[str, ...] = ("ALL", "-FULLSAMPLE", "-SOURCE")

logger = logging.getLogger(__name__)

def get_metadata_from_filename(fn):
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

    # This is stripped down version of ingest_hdul. It does not check overlap with the target template so that
    # it can be called without dependency on taget.

    ACTIVE_AREA_SHAPE = (2040, 2040)

    flag = get_flagval(*flags)

    # Check the argument and get image, WCS, and metadata from it
    input_image = hdulist.filename()

    # Get HDU of spectral image. Check the header, not `.data is not None`,
    # which would force a full lazy-load of the array just to test presence.
    if hdulist[0].header.get("NAXIS", 0) > 0:
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

def _convert_hdul_to_df(hdul, bandpass_model, band, metadata):
    tmpl_shape = hdul[0].data.shape


    # ind = np.sum(np.indices((2048, 2048)) * np.array([2048, 1]).reshape((2, 1, 1)),
    #              axis=0, dtype="int32")
    # y, x = np.divmod(ind, 2048)


    tmpl_ind = make_pixel_index(tmpl_shape)

    msk = np.isfinite(hdul[0].data)
    # src_y, src_x = np.divmod(hdul[2].data[msk], 2048)
    src_y, src_x = SphxReprojector.get_src_yx(hdul[2].data[msk])
    wvl = bandpass_model(src_x, src_y,
                         array=int(band), central_bandpass_only=True)[0]

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
        **metadata
    ))

    return df

class SphxReprojector:
    DATASHAPE = (2048, 2048)

    @classmethod
    def get_ind_image(cls, input_image):
        """
        convert x,y indices of the input SPHEREx spectral image (which should be usually 2040x2040) to a single integer. For simplicity, we assume 2048x2048 shape.
        """
        (ny, nx) = input_image.shape
        assert (ny <= SRC_STRIDE) and (nx <= SRC_STRIDE)
        # float32 so the packed index survives reprojection
        return make_pixel_index((ny, nx), stride=SRC_STRIDE, dtype="float32")

    @classmethod
    def get_src_yx(cls, ind_array):
        return get_src_yx(ind_array)

    def __init__(self, input_hdul, *,
                 flags=None, aux_metadata=None,
                 bandpass_model=None):
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
        
        self.bandpass_model = (Tabular_Bandpass_Lite() if bandpass_model is None
                               else bandpass_model)

    def process_single(self, output_wcs_tmpl, **reproject_kwargs):
        """Reproject this exposure onto output_wcs_tmpl.

        Extra keyword arguments are forwarded to reproject_adaptive
        (e.g. parallel=True for reproject's block-parallel mode on a
        single large template), overriding the defaults
        bad_value_mode="ignore", parallel=False.
        """

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

        adaptive_kwargs = dict(bad_value_mode="ignore", parallel=False)
        adaptive_kwargs.update(reproject_kwargs)
        array_out, footprint = reproject_adaptive((self.nddata, self.wcs_in), wcs_tmpl,
                                                  shape_out=shape_out,
                                                  **adaptive_kwargs,
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

        df = _convert_hdul_to_df(hdul, self.bandpass_model, self.band, self.metadata)
        cards = TemplateHeaderCards.from_header(hdul[1].header)
        cards.update_dataframe(df)

        return df


def get_df_from_hdul(hdul, wcs_tmpl, aux_metadata=None):
    if aux_metadata is None:
        aux_metadata = {}

    reprojector = SphxReprojector(hdul, aux_metadata=aux_metadata)

    out_hdul = reprojector.process_single(wcs_tmpl)

    if out_hdul is not None:

        df = reprojector.hdul_to_pandas(out_hdul)
    else:
        df = None

    return df

import fsspec

def read_uri(uri, **storage_options):
    # Get the filesystem instance
    protocol_name = uri.split('://')[0]
    fs = fsspec.filesystem(protocol_name, asynchronous=False, **storage_options)

    try:
        with fs.open(uri, "rb") as f:
            content = f.read()
            return content
    except AttributeError as e:
      raise e


def get_df_from_buffer(buffer, wcs_tmpl, *, aux_metadata=None,
                       zodi_corrector=None):
    """Zodi-correct and reproject an in-memory L2 file onto wcs_tmpl.

    The single shared implementation behind both the sync
    (get_df_from_uri) and async (reproj_s3_async.run_reproj_tasks) paths.
    """
    hdul = fits.open(BytesIO(buffer))

    zodi = hdul["ZODI"].data
    if zodi_corrector is not None:
        zodi = zodi_corrector(zodi)

    hdul["IMAGE"].data -= zodi

    return get_df_from_hdul(hdul, wcs_tmpl, aux_metadata=aux_metadata)


def get_df_from_uri(wcs_tmpl, uri, *, pbar=None, zodi_corrector=None):
    aux_metadata = get_metadata_from_filename(uri)

    buffer = read_uri(uri)
    _df = get_df_from_buffer(buffer, wcs_tmpl, aux_metadata=aux_metadata,
                             zodi_corrector=zodi_corrector)

    if pbar is not None:
        pbar.update()
    else:
        logger.info("processed %s", uri)

    return _df



def merge_to_stable(dflist, tmpl_wcs=None):
    # cards = TemplateHeaderCards.from_dataframe(dflist[0])
    if tmpl_wcs is None:
        cards_list = [TemplateHeaderCards.from_dataframe(df) for df in dflist]
        master_cards = cards_list[0]
        if any(master_cards.hash != cards.hash for cards in cards_list):
            raise RuntimeError("template header cards  are inconsistent.")

    else:
        header = tmpl_wcs.to_header()
        if "NAXIS1" not in header:
            header["NAXIS1"] = tmpl_wcs._naxis[1]
            header["NAXIS2"] = tmpl_wcs._naxis[0]
        master_cards = TemplateHeaderCards.from_header(header)

    # Sort by pixel index so parquet row-group min/max statistics make
    # later spatial-cutout reads cheap (predicate pushdown on tmpl_ind).
    df = (pd.concat(dflist, ignore_index=True)
          .set_index("tmpl_ind")
          .sort_index(kind="stable"))

    master_cards.update_dataframe(df)

    return df


