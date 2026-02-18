# COILED memory      8 GiB
# COILED region      us-east-1

import time
from pathlib import Path
from io import BytesIO

from collections.abc import Callable, Iterable, Mapping

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.nddata import NDData

from spherex_utils import utils

import boto3
from botocore.exceptions import ClientError
import logging
import os

from reproject import reproject_adaptive, reproject_interp
from reproject.hips.utils import tile_header

from astropy.wcs import WCS
from astropy.coordinates import Galactic
from spherex_utils.utils.mosaic_utils import get_flagval, DEFAULT_FLAGS

# for r2

# ENDPOINT_URL = "https://83df4672ce1bde3adf28577626654ab8.r2.cloudflarestorage.com"
# # endpoint_url = "https://83df4672ce1bde3adf28577626654ab8.r2.cloudflarestorage.com"
# ACCESS_KEY = "d824ca865eed0f4efa62f8ee270db6d4"
# SECRET_KEY = "3c395955a44415bf1ec28bf3b6ca2c5c15b87985ce8f0f8df4c44b77eed84666"
# BUCKET_NAME = "spherex-hips-staging"
# # ENDPOINT_URL = "https://83df4672ce1bde3adf28577626654ab8.r2.cloudflarestorage.com/spherex-hips-staging-l080-b0"

# For s3

BUCKET_NAME = "sphx-reproj-ouput"



# common
# PROJNAME = f"gal_l{l:03d}_b{b:02d}"



def upload_data_to_s3(data, object_name):

    bucket = BUCKET_NAME

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Bucket=bucket, Key=object_name, Body=data)
        logging.info(f"uploaded to {bucket}/{object_name}")
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_data_to_r2(data, object_name):
    """Upload a file to a Cloudflare R2 bucket.

    :param file_name: File to upload.
    :param bucket: Bucket to upload to.
    :param endpoint_url: The endpoint URL for R2 storage.
    :param object_name: R2 object name. If not specified then file_name is used.
    :return: True if file was uploaded, else False.
    """


    # Upload the file
    bucket = BUCKET_NAME

    s3_client = boto3.client('s3', endpoint_url=ENDPOINT_URL,
                             aws_access_key_id=ACCESS_KEY,
                             aws_secret_access_key=SECRET_KEY,
                             #config=Config(signature_version=UNSIGNED)
                             )
    try:
        s3_client.put_object(Bucket=bucket, Key=object_name, Body=data)
        # s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"uploaded to {bucket}/{object_name}")
    except ClientError as e:
        logging.error(e)
        return False
    return True


upload_data = upload_data_to_s3

del upload_data_to_s3
del upload_data_to_r2


# def upload_file_to_r2(file_name, bucket, object_name=None):
#     """Upload a file to a Cloudflare R2 bucket.

#     :param file_name: File to upload.
#     :param bucket: Bucket to upload to.
#     :param endpoint_url: The endpoint URL for R2 storage.
#     :param object_name: R2 object name. If not specified then file_name is used.
#     :return: True if file was uploaded, else False.
#     """

#     # If R2 object_name was not specified, use file_name
#     if object_name is None:
#         object_name = os.path.basename(file_name)

#     # Upload the file

#     s3_client = boto3.client('s3', endpoint_url=ENDPOINT_URL,
#                              aws_access_key_id=ACCESS_KEY,
#                              aws_secret_access_key=SECRET_KEY,
#                              #config=Config(signature_version=UNSIGNED)
#                              )
#     try:
#         s3_client.put_object(Bucket=bucket, Key=object_name, Body=data_to_upload)
#         # s3_client.upload_file(file_name, bucket, object_name)
#         logging.info(f"File {file_name} uploaded to {bucket}/{object_name}")
#     except ClientError as e:
#         logging.error(e)
#         return False
#     return True

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


class SphxHpxProcess:
    def __init__(self, PROJNAME, plan, root, band, hdul, frame_name="galactic",
                 flags=None):
        self.PROJNAME = PROJNAME
        self.plan = plan
        self.root = root
        self.band = int(band)

        # self.frame_name
        if frame_name == "galactic":
            self.frame = Galactic()
        else:
            raise ValueError()

        self.flags = utils.mosaic_utils.DEFAULT_FLAGS if flags is None else flags
        nddata_list = ingest_hdul(hdul, flags=self.flags, process_variance=True)
        if isinstance(nddata_list, bool):
            raise RuntimeError()

        self.nddata = np.array([nd.data for nd in nddata_list])
        self.wcs_in = WCS(hdul["IMAGE"].header)

        self.ind = np.sum(np.indices(hdul["IMAGE"].data.shape) * np.array([2048, 1]).reshape((2, 1, 1)),
                          axis=0, dtype="float32")

        self.bandpass_model = utils.mosaic_utils.PixelToCentralWavelengthUsingWCS()

    def process_hid(self, level, tile_size, hid):
        header = tile_header(level=level, index=hid, frame=self.frame,
                             tile_dims=tile_size)

        wcs_tmpl = WCS(header)

        # mosaic = utils.SpectralChannelMosaic(
        mosaic = utils.WavelengthRangeMosaic(
            wcs_tmpl,
            self.band,
            wavelength_range=(0, 10), # wide enough wavelength range
            flags=self.flags, # .split(","),
            # bandpass_model=bandpass,
            process_variance=True,
            bandpass_model=self.bandpass_model,
        )

        wcs_tmpl = mosaic.coadd_wcs
        shape_out = mosaic.coadd_wcs.array_shape

        # nddata_list = mosaic.ingest_hdul(hdul)
        # if nddata_list is False:
        #     processed.append((hid, "empty", ""))
        #     continue

        array_out, footprint = reproject_to_hips_tile(self.nddata, self.wcs_in, header,
                                                      reproject_function=reproject_adaptive,
                                                      shape_out=shape_out,
                                                      bad_value_mode="ignore",
                                                      parallel=True,
                                                      )
        if np.all(footprint == 0):
            return None

        # tile_format IS fits
        array_out[footprint == 0] = np.nan

        # FIXME: having dtype of int32 fails when reproj with nan issue.
        # So we use float32 then convert it to int32 while saving

        outputarray = np.zeros(array_out[0].data.shape, dtype="float32")

        ind_out, footprint = reproject_to_hips_tile(self.ind, self.wcs_in, header,
                                                    reproject_function=reproject_interp,
                                                    order=0,
                                                    output_array=outputarray,
                                                    shape_out=shape_out,
                                                    # # shape_out=N,
                                                    # bad_value_mode="ignore",
                                                    parallel=False,
                                                    )
        ind_out[footprint == 0] = -1


        out_hdul = fits.HDUList(
            [fits.PrimaryHDU(data=array_out[0], header=header),
             fits.ImageHDU(data=array_out[1], header=header),
             fits.ImageHDU(data=ind_out.astype("int32"), header=header)]
        )


        return out_hdul


    def process_hid_list(self, level, tile_size, hid_list):
        processed = []
        for hid in hid_list:
            try:
                hdul_out = self.process_hid(level, tile_size, hid)
            except Exception as e:
                # raise e
                processed.append((hid, "error in processing", e))
                continue

            if hdul_out is None: # no overwrap
                processed.append((hid, "no overwrapp", None))
                continue

            processed.append((hid, "success", hdul_out))

        return processed

    def finalize_all(self, level, tile_size, processed):
        processed_new = []
        for hid, code, hdul_out in processed:
            if code != "success":
                processed_new.append((hid, code, []))
                continue
            try:
                fnout = self.finalize_output(level, tile_size, hid, hdul_out)
                processed_new.append((hid, code, fnout))
            except Exception as e:
                # raise e
                processed_new.append((hid, "error in finalizing", str(e)))
                continue

        return processed_new

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


    def finalize_output(self, level, tile_size, hid, hdul_out):
        # fnout = self._finalize_output_local_save(level, tile_size, hid, hdul_out)
        fnout = self._finalize_output_s3(level, tile_size, hid, hdul_out)
        return fnout

    def get_outname(self, level, tile_size, hid):
        return f"hpx{level}_{hid}_{tile_size}_sph_{self.plan}_{self.root}.fits"

    def _finalize_output_s3(self, level, tile_size, hid, hdul_output):

        outname = self.get_outname(level, tile_size, hid)

        fout = BytesIO()
        # tmpl_header = wcs_tmpl.to_header()
        hdul_output.writeto(fout)
        # bucket = "sphx-reproj-ouput"
        upload_data(fout.getvalue(), f"{self.PROJNAME}/{self.band}/{level}_{tile_size}/{hid}/{outname}")
        return outname

    def _finalize_output_local_save(self, level, tile_size, hid, hdul_output):
        outname = self.get_outname(level, tile_size, hid)

        # outdir = Path(f"{PROJNAME}/{self.band}/{level}_{tile_size}/{hid}")
        outdir = Path(f"{self.PROJNAME}/{self.band}/{level}_{tile_size}/{hid}")
        outdir.mkdir(parents=True, exist_ok=True)
        hdul_output.writeto(outdir / outname, overwrite=True)
        print("write", outdir / outname)
        return outname


def process(plan, root, band, sphx_name, s3uri, level, tile_size, hid_list, frame_name="galactic"):
    mark1 = time.perf_counter()

    # hdul = fits.open(sphx_name)
    hdul = fits.open(s3uri, use_fsspec=True, fsspec_kwargs={"anon": True})

    mark2 = time.perf_counter()
    print("time spent 1:", mark2 - mark1)

    shp = SphxHpxProcess(plan, root, band, hdul, frame_name=frame_name)

    processed_intermediate = shp.process_hid_list(level, tile_size, hid_list)

    mark3 = time.perf_counter()

    print("time spent 2:", mark3 - mark2)

    processed = shp.finalize_all(level, tile_size, processed_intermediate)

    mark4 = time.perf_counter()
    print("time spent 3:", mark4 - mark3)

    time.sleep(1)

    return processed


def process_local(projname, plan, root, band, sphx_name, s3uri, level, tile_size, hid_list, frame_name="galactic"):
    mark1 = time.perf_counter()

    hdul = fits.open(s3uri)
    # hdul = fits.open(s3uri, use_fsspec=True, fsspec_kwargs={"anon": True})

    mark2 = time.perf_counter()
    print("time spent 1:", mark2 - mark1)

    shp = SphxHpxProcess(projname, plan, root, band, hdul, frame_name=frame_name)

    processed_intermediate = shp.process_hid_list(level, tile_size, hid_list)

    mark3 = time.perf_counter()

    print("time spent 2:", mark3 - mark2)

    processed = shp.finalize_all_local(level, tile_size, processed_intermediate)

    mark4 = time.perf_counter()
    print("time spent 3:", mark4 - mark3)

    # time.sleep(1)

    return processed


def get_test_data():
    hid = 22890

    fn = "hpix_gal_l080_b00_sphx_overwrapping.parquet"
    df0 = pd.read_parquet(fn)
    fn = "hpix_gal_l080_b00_sphx_overwrapping_s3_filtered.parquet"
    df_s3uri = pd.read_parquet(fn)

    if False:
        # simple code to figure out example filename
        df = df0.query(f"(hpx6 == {hid}) and (DETECTOR == 4) and (channel == 10)")

    dfg = df0.groupby("filename")

    sphx_name = "level2_2025W23_1C_0165_1D4_spx_l2b-v19-2025-251.fits"
    row = df_s3uri.set_index("filename").loc[sphx_name]
    s3uri = row["s3uri"]
    # band = row["DETECTOR"]
    plan = row["plan"]
    root = row["root"]
    band = row["band"]

    col = dfg.get_group(sphx_name)
    level = 6
    hpix6_list = col["hpx6"].unique().tolist()

    return plan, root, band, sphx_name, s3uri, level, hpix6_list


class SpectralImageDataFrame:
    def __init__(self, df_hips, df_filename):
        """
        df_hips : This df is used to get the hip list
        df_filename : used to get the band, root etc, unique to the file
        """
    #     # This file is .
    #     # This file is used to get the hip list.

        self.df_s3uri = df_filename
        self.df0 = df_hips
        self.level = 6

    # def __init__(self, root):
    #     # This file is used to get the band, root etc, unique to the file.
    #     fn = f"{root}_sphx_overwrapping_s3_filtered.parquet"
    #     self.df_s3uri = pd.read_parquet(fn)
    #     # self.df_s3uri = None

    #     # This file is used to get the hip list.
    #     fn = f"{root}_sphx_overwrapping.parquet"
    #     self.df0 = pd.read_parquet(fn)
    #     # self.df0 = None

    #     self.level = 6

    def get_band_plan_root(self, sphx_name, return_colname="s3uri"):
        row = self.df_s3uri.set_index("filename").loc[sphx_name]
        # band = row["DETECTOR"]
        band = row["band"]
        plan = row["plan"]
        root = row["root"]

        s3uri = row[return_colname]

        return band, plan, root, s3uri

    def get_hip_list(self, sphx_name):
        dfg = self.df0.groupby("filename")

        col = dfg.get_group(sphx_name)
        hpix6_list = col["hpx6"].unique().tolist()

        return self.level, hpix6_list

    def get_data(self, sphx_name, return_colname="s3uri"):

        band, plan, root, s3uri = self.get_band_plan_root(sphx_name,
                                                          return_colname=return_colname)
        level, hpix_list = self.get_hip_list(sphx_name)

        return plan, root, band, sphx_name, s3uri, level, hpix_list


def get_data(root, sphx_name, return_colname="s3uri"):
    """
    sphx_name : value for `filename` column in the parquet.
    return_colname : file uri to return. Default is s3uri.
    """
    fn = f"{root}_sphx_overwrapping_s3_filtered.parquet"
    df_s3uri = pd.read_parquet(fn)

    row = df_s3uri.set_index("filename").loc[sphx_name]
    s3uri = row[return_colname]
    # band = row["DETECTOR"]
    band = row["band"]
    plan = row["plan"]
    root = row["root"]

    fn = f"{root}_sphx_overwrapping.parquet"
    df0 = pd.read_parquet(fn)

    dfg = df0.groupby("filename")

    col = dfg.get_group(sphx_name)
    level = 6
    hpix6_list = col["hpx6"].unique().tolist()

    return plan, root, band, sphx_name, s3uri, level, hpix6_list



def coiled():
    from coiled import Cluster

    cluster = Cluster(
        n_workers=1,
        region="us-east-1",
        worker_memory="16 GiB",
        worker_options={"nthreads": 1}
    )

    client = cluster.get_client()
    # process(plan, root, band, sphx_name, s3uri, level, tile_size, hpix6_list, frame_name="galactic")

    tile_size = 512
    plan, root, band, sphx_name, s3uri, level, hpix6_list = get_test_data()
    r = client.submit(process, plan, root, band, sphx_name, s3uri, level, tile_size, hpix6_list, frame_name="galactic")

def run_test():

    plan, root, band, sphx_name, s3uri, level, hpix6_list = get_test_data()

    tile_size = 512

    processed = process_local(plan, root, band, sphx_name, s3uri, level, tile_size, hpix6_list[:4], frame_name="galactic")

    import json
    data = json.dumps(processed).encode("ascii")
    object_name = f"{PROJNAME}/{band}/processed_{sphx_name}.json"
    upload_data(data, object_name)

def main():
    import os
    input = os.environ["COILED_BATCH_TASK_INPUT"]


    _, sphx_name = [s.strip() for s in input.split(",")]

    l, b = 190, 0
    # l, b = 80, 0

    PROJNAME = f"gal_l{l:03d}_b{b:02d}"
    HPIX_PORJNAME = f"hpix_{PROJNAME}"

    plan, root, band, sphx_name, s3uri, level, hpix6_list = get_data(HPIX_PORJNAME, sphx_name)

    tile_size = 512

    processed = process(plan, root, band, sphx_name, s3uri, level, tile_size, hpix6_list,
                        frame_name="galactic")

    import json
    data = json.dumps(processed).encode("ascii")
    object_name = f"{PROJNAME}/{band}/processed_{sphx_name}.json"
    upload_data(data, object_name)


def do_local(sphx_name):
    # import os
    # input = os.environ["COILED_BATCH_TASK_INPUT"]
    # _, sphx_name = [s.strip() for s in input.split(",")]

    l, b = 190, 0
    # l, b = 80, 0

    PROJNAME = f"gal_l{l:03d}_b{b:02d}"
    HPIX_PROJNAME = f"hpix_{PROJNAME}"

    tile_size = 512

    region_root = "hpix_gal_l190_b00"

    # This file is used to get the hip list.
    fn = f"{region_root}_sphx_overwrapping.parquet"
    df_hips = pd.read_parquet(fn)

    # This file is used to get the band, root etc, unique to the file.
    # fn = f"{root}_sphx_overwrapping_s3_filtered.parquet"
    fn = f"{region_root}_sphx_overwrapping_s3_n_local_names.parquet"
    df_filename = pd.read_parquet(fn)

    si = SpectralImageDataFrame(df_hips, df_filename)
    plan, root, band, sphx_name, s3uri, level, hpix6_list = si.get_data(sphx_name,
                                                                        return_colname="local_name")

    print(f"processing... {s3uri}")
    # plan, root, band, sphx_name, s3uri, level, hpix6_list = get_data(HPIX_PROJNAME, sphx_name)

    processed = process_local(PROJNAME, plan, root, band, sphx_name,
                              s3uri,
                              level, tile_size, hpix6_list,
                              frame_name="galactic")

    # import json
    # data = json.dumps(processed).encode("ascii")
    # object_name = f"{PROJNAME}/{band}/processed_{sphx_name}.json"
    # upload_data(data, object_name)


def main_local():
    # sphx_name = "level2_2025W38_2C_0153_2D1_spx_l2b-v20-2025-269.fits"


    for i in range(len(df_to_process)):
        print(f"[{i}]")
        sphx_name = df_to_process["filename"].iloc[i]
        do_local(sphx_name)

def main_ray():
    import ray
    ray.init()

    df_to_process = pd.read_csv("local_files_to_process.csv")
    NUM_TASKS = len(df_to_process)
    r_df_to_rpocess = ray.put(df_to_process)

    @ray.remote(num_cpus=1)
    def doit(df_to_process, i):
        print(f"[{i}]")
        sphx_name = df_to_process["filename"].iloc[i]
        do_local(sphx_name)


    MAX_NUM_PENDING_TASKS = 8
    result_refs = []
    # for _ in range(NUM_TASKS):
    for i in range(32):
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            ray.get(ready_refs)

        result_refs.append(doit.remote(r_df_to_rpocess, i))

    ray.get(result_refs)
    print("Done")

if __name__ == '__main__':
    main_ray()
