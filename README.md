# SPHEREx Spectral Image as a Table (ssiaat)

Tools for turning SPHEREx Level-2 spectral images into long-format pandas
tables, and for fitting spectral models per pixel.

## Glossary

- **stable** — *s*pectral *table*: a long-format DataFrame with one row per
  (pixel, wavelength) sample. Its index is the integer template pixel index
  `tmpl_ind`, **with duplicates** (many wavelengths per pixel).
- **itable** — *i*mage *table*: a Series with a **unique** integer pixel
  index; one value per template pixel (e.g. a fitted amplitude map).
- **template** — the output WCS grid that all exposures are reprojected onto.

`promote_to_stable(df, header=...)` attaches the template header metadata a
stable needs; it has nothing to do with stability.

## Install

```sh
pip install -e .          # plus `.[test]` for the test suite
```

The `fsspec[http]<=2025.3.0` / `s3fs<=2025.3.0` pins keep the package
installable in Google Colab (as of 2026-07).

## Quickstart

```python
import asyncio
import astropy.units as u
import ssiaat

# 1. Which exposures overlap the target?
df_query = ssiaat.query_overlapping(ra_deg=24.177, dec_deg=15.787, side_deg=0.9)
filenames = df_query["filename"].unique()

# 2. Find the latest pipeline version of each file on the archive.
uris = ssiaat.find_latest_uri(filenames, "s3://nasa-irsa-spherex")
# (inside Jupyter use:  await ssiaat.find_latest_uri_async(...))

# 3. Reproject everything onto a template WCS and build a stable.
wcs_tmpl = ssiaat.get_wcs(24.177 * u.deg, 15.787 * u.deg, 0.9 * u.deg)
dfl, failures = asyncio.run(ssiaat.run_reproj_tasks(uris, wcs_tmpl))
stable = ssiaat.merge_to_stable(dfl, tmpl_wcs=wcs_tmpl)
stable.to_parquet("target.parquet")

# 4. Later: read it back (template metadata travels inside the parquet).
stable = ssiaat.read_stable("target.parquet", wvl_range=(2.6, 4.2))

# 5. Quick looks via the .spectral accessor.
image = stable.spectral.make_simple_image(3.9, 4.1)      # 2d map
spec = stable.spectral.binned_spectrum(bins=100)         # median spectrum
spec.plot()

# 6. Per-pixel linear model fit with named components.
from ssiaat.model.sed import get_br_a, const
model = ssiaat.Model({"br_a": get_br_a()}, {"cont": const()})
fitted = model.least_square_fit(stable, return_error=True)
fitted.image("br_a")          # Br-alpha amplitude map on the template
fitted.to_frame()             # all coefficients as a named DataFrame
```

### Recipe: continuum-normalize rows with `.spectral.broadcast`

Per-pixel values (an itable or a 2d image) align onto the stable's
(pixel × wavelength) rows with `broadcast`, so no manual `join` dance is
needed:

```python
cont = stable.spectral.make_simple_itable(2.4, 3.0)      # per-pixel continuum
normalized = stable["image"] / stable.spectral.broadcast(cont)
```

## Notes

- Runnable end-to-end scripts live in `examples/`.
- Bulk runs never abort on a bad file: `run_reproj_tasks` returns
  `(dataframes, failures)` where `failures` is a list of
  `(uri, exception)` pairs to retry.
- HiPS-tile output (`ssiaat.reproj_hips`) exists but is second-class for
  now; import it explicitly if needed.
