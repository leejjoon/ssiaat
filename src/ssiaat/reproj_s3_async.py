"""
This runs reporj tasks in co-routines. It gain some spped ups from parallel s3 file fetching, but
not significant.
"""

import fsspec
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
from io import BytesIO
from contextlib import nullcontext
from astropy.io import fits
from .reproj import get_df_from_hdul, get_metadata_from_filename
# from .reproj import SphxReprojector


# from astropy.wcs import WCS

async def async_read_file(path, **storage_options):
    # Get the filesystem instance
    protocol_name = path.split('://')[0]
    fs = fsspec.filesystem(protocol_name, asynchronous=True, **storage_options)

    try:
        async with await fs.open_async(path, "rb") as f:
            content = await f.read()
            return content
    except AttributeError as e:
      raise e


class ProjectorRunner():
    def __init__(self, wcs_tmpl):
        self.dfl = []
        self.output_wcs_tmpl = wcs_tmpl
        self.queue = asyncio.Queue()

    async def init_queue(self, uri_list):
        # dfl = []
        for uri in uri_list:
            if uri is None:
                continue
            # uri = row["uri_decorated
            await self.queue.put((self.dfl, uri))

    async def get_df_from_uri(self, uri, *, pbar=None):
        aux_metadata = get_metadata_from_filename(uri)

        buffer = await async_read_file(uri, anon=True)
        f = BytesIO(buffer)

        hdul = fits.open(f)

        df = get_df_from_hdul(hdul, self.output_wcs_tmpl, aux_metadata=aux_metadata)

        if pbar is not None:
            pbar.update()
        else:
            print(uri)

        return df

    async def worker_s3(self, *, pbar=None):
        while True:
            dfl, uri = await self.queue.get()
            uri = f"s3://{uri}"
            df = await self.get_df_from_uri(uri, pbar=pbar)
            if df is not None:
                dfl.append(df)
            self.queue.task_done()


async def run_s3_repoj_tasks(uri_list, wcs_tmpl, num_tasks=4, progress=True):

    runner = ProjectorRunner(wcs_tmpl)
    await asyncio.create_task(runner.init_queue(uri_list)) # instead of waiting the task, we wait for the queue.

    pbar_ = tqdm_asyncio(total=len(uri_list)) if progress else nullcontext()
    with pbar_ as pbar:

        tasks = [asyncio.create_task(runner.worker_s3(pbar=pbar)) for _ in range(num_tasks)]

        await runner.queue.join()
        for t in tasks:
            t.cancel()

    return runner.dfl
