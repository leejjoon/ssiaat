# SPHEREx Spectral Image as a Table (ssiaat)


## Package requirement

"fsspec[http]<=2025.3.0" "s3fs<=2025.3.0" : this was set so that the package can be installed in google colab env.



## How to handle large tables

- `process_simeis147_4_5d.py` : create stable but in chunks.

- `merge_parquet.py` : merge or split the parquet files.

- `process_simeis147_fit.py` : attempt to fit in batch to reduce memory footprint.

