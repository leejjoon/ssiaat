import pandas as pd
import asyncio
import aiohttp
import re
import fsspec
from fsspec.core import url_to_fs
from ssiaat.finder import _get_table_from_filenames, _find_latest_single_async

async def test_single_file():
    # Testing the second unique filename (Row 3 in ECSV)
    filename = "level2_2025W24_1A_0405_2D1_spx_l2b-v19-2025-252.fits"
    df = _get_table_from_filenames([filename])
    row = df.iloc[0]
    
    print("Parsed row info:")
    print(row)
    
    root_uri = "http://100.103.128.7:3000"
    uri_option = dict()
    root_uri = "s3://nasa-irsa-spherex"
    uri_option = dict(anon=True)
    release = "qr2"
    
    fs, root_path = url_to_fs(root_uri, asynchronous=True, **uri_option)
    semaphore = asyncio.Semaphore(1)
    
    #async with aiohttp.ClientSession() as session:
    #    # Now run the actual logic
    result = await _find_latest_single_async(row, fs, root_path, release, semaphore)
    print(f"\nResult from _find_latest_single_async: {result}")

if __name__ == "__main__":
    asyncio.run(test_single_file())
