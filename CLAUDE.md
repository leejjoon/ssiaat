# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ssiaat` turns SPHEREx Level-2 spectral FITS images into long-format pandas
tables ("stables") for per-pixel spectral analysis, and provides a linear
model fitter over them. Read `README.md` first — it has the glossary and a
full quickstart; this file only covers what the README doesn't.

## Commands

Use this repo's own venv at **`./.venv`** (Python 3.12.13, uv-managed) — not
the parent `/home/jjlee/spherex/.venv`, which is a separate environment for
the Coiled/dask HiPS pipeline. Activate it or prefix commands with
`.venv/bin/`:

```sh
uv venv --python 3.12 .venv         # one-time creation
uv pip install -e '.[test,lint]' --python .venv   # editable install + test + lint deps
.venv/bin/pytest                    # slow-marked tests are excluded by default (see pyproject addopts)
.venv/bin/pytest -m slow            # run only the slow tests
.venv/bin/pytest tests/test_stable.py::test_make_simple_image   # single test
.venv/bin/pytest -k make_simple     # by name substring
.venv/bin/ruff check src tests      # lint (see below)
```

CI (`.github/workflows/tests.yml`) does exactly `pip install -e .[test,lint]`,
`pytest -m "not slow"`, then `ruff check src tests` on Python 3.12 —
reproduce that when checking a change is CI-clean. The `ruff` step runs with
`continue-on-error: true`: findings are surfaced but never fail the build.

This repo's `pyproject.toml` pins `botocore<1.41.6` for the same
aiobotocore/s3fs reason documented in the parent workspace's `CLAUDE.md`;
a fresh `./.venv` resolves it correctly on its own.

`ruff` is configured (`[tool.ruff]` in `pyproject.toml`) with a deliberately
narrow rule set — pyflakes (`F`) plus a few pycodestyle error checks
(`E4`/`E7`/`E9`) — not ruff's much broader defaults (bugbear,
tryceratops, flake8-async, ...), which flag patterns this codebase uses on
purpose (see "never abort a bulk run" below). Widen `select` deliberately,
not by accident. A handful of pre-existing findings are left unfixed on
purpose: `E402` (module-level imports placed mid-file at `# %%`
notebook-style cell markers) and `F821`/`F841` inside `model/sed.py`'s
already-dead `channel_model_deprecated`/`test()` demo code and one
validating-but-unused `det_id` lookup in `reproj.py`'s `ingest_hdul` — check
`ruff check src tests` output before assuming a fresh finding is new.

## Architecture: the pipeline, in order

Data flows through five stages; each has one module and the public API
(`ssiaat/__init__.py`) exposes exactly one entry point per stage:

1. **Query** (`query.py`) — `query_overlapping(ra, dec, side_deg)` hits an
   external HTTP API to list exposures overlapping a sky region. Returns a
   DataFrame of filenames.
2. **Find** (`finder.py`) — SPHEREx reprocesses data under versioned
   `l2b-vNN-YYYY-DDD` pipeline directories; `find_latest_uri[_async]` parses
   filenames, lists candidate pipeline versions per plan, and returns the
   URI of the numerically latest one that exists (version strings must be
   compared numerically, never lexically — `v9` sorts after `v20` as a
   string). Works over any fsspec protocol (s3/http/file); `fs.py` registers
   a `webfsd://` filesystem (HTML directory-listing scraper) for one such
   backend. `find_latest_uri` is a sync wrapper that raises if called inside
   a running event loop (Jupyter) — use `find_latest_uri_async` there.
3. **Reproject** (`reproj.py`, `reproj_s3_async.py`) — `SphxReprojector`
   wraps one input HDUList: `ingest_hdul` validates it's a real L2 file
   (checks `L2DQAFLG`, `FLAGS`, `VARIANCE` extensions, masks flagged
   pixels), `process_single` reprojects image+variance (adaptive) and a
   packed per-pixel source-index array (nearest-neighbor, so it survives
   reprojection undistorted) onto the output template WCS, and
   `hdul_to_pandas`/`_convert_hdul_to_df` flattens the result into a
   DataFrame with one row per finite output pixel: `tmpl_ind`, `image`,
   `variance`, `src_x`/`src_y` (decoded detector position), `wvl` (looked up
   via the bandpass model), plus any `metadata` columns. `merge_to_stable`
   concatenates per-exposure DataFrames, sets `tmpl_ind` as the index, and
   sorts by it — sorted so parquet row-group min/max stats make later
   spatial-cutout reads cheap. `run_reproj_tasks` (in `reproj_s3_async.py`)
   is the bulk async driver: `num_fetchers` concurrent downloads feed a
   `num_workers`-process pool (CPU-bound `reproject_adaptive` scales across
   cores; `num_workers=0` runs inline for debugging/picklability issues). It
   never aborts on one bad file — see `AsyncCollector` below.
   `reproj_hips.py` (HiPS-tile output) is second-class / not part of the
   public API — import it explicitly if needed.
4. **Stable/Image bridge** (`spherex_table.py`) — the pandas-accessor layer.
   `promote_to_stable`/`read_stable` attach template WCS metadata into
   `df.attrs["ssiaat_template_header"]` (as the header's card images, hashed
   for equality checks — see `TemplateHeaderCards` in `wcs_helper.py`) so a
   stable is self-describing: no template file needs to travel alongside
   the parquet. `SsiaatConverter` reconstructs from that header and does the
   actual `tmpl_ind <-> 2D array` mapping (`itable_to_image`/
   `image_to_itable`); it's lazily rebuilt and cached (`_get_converter`) from
   `.attrs` whenever a bare DataFrame/Series shows up without one attached.
   Two accessors sit on top: `.spectral` (DataFrame; requires an integer
   index, duplicates optional — a single-exposure tile has one row per
   pixel) and `.itable` (Series; requires a *unique* integer index — one
   value per pixel, e.g. a fitted map). `Image` is an `np.ndarray` subclass
   that carries its converter through slicing/math so `.to_itable()` keeps
   working on a cropped image.
5. **Model fitting** (`model/`) — `sed.py` has the line/continuum-shape
   library (smooth top-hat-like functions built from `tanh_step`); `Model`
   bundles named line + continuum callables, `least_square_fit` factorizes
   the stable's (duplicated) pixel index into per-pixel groups and calls
   `vectorized_lstsq.py`'s numpy-vectorized batched least squares (one
   independent small linear solve per pixel, batched — not a Python loop per
   pixel). `FitResults` wraps the coefficients as pixel-indexed `pd.Series`
   (so `.itable.to_image()` works directly on any coefficient) and carries
   the template header through so a fitted map round-trips like a stable.

`zodi_correction.py` and `tabular_bandpass_lite.py` are inputs consumed by
stage 3 (zodiacal-light subtraction and detector-pixel → wavelength lookup,
respectively); `tabular_bandpass_lite.py` is a frozen from-coefficients
re-implementation of the full `spherex_tabular_bandpass` package (not a
dependency) supporting only `central_bandpass_only=True`.

## Cross-cutting conventions

- **Two packed pixel-index schemes coexist** (`indexing.py` has the full
  writeup): `tmpl_ind` (stride = template width, dtype int32, the
  stable/itable index) and the detector `src_ind` (fixed stride 2048, stored
  as **float32** so it survives interpolation during reprojection, decoded
  via bit ops in `get_src_yx` that assume that exact stride). Don't reuse
  `make_pixel_index`'s default stride for detector-space work.
- **Never let one bad item abort a bulk run.** Both `AsyncCollector`
  (`async_collector.py`, used by `run_reproj_tasks`) and the finder's
  per-file async lookups catch exceptions per item and record them
  (`.failures` / `None`) rather than propagating, specifically so a raising
  worker can't leave `queue.join()` waiting forever. Preserve this pattern
  when touching either bulk path.
- **A stable's index is meaningful, not incidental**: integer +
  duplicates = stable, integer + unique = itable. `check_index_stable` /
  `check_index_itable` (`spherex_table.py`) encode this and are the
  guardrails on `promote_to_stable` (bypass with `ignore_index_check=True`
  only when you're sure, e.g. a genuinely single-exposure tile).
- **Template metadata travels inside the data**, not as a side channel —
  any new code producing a stable/itable/fitted-map must propagate
  `df.attrs["ssiaat_template_header"]` (see how `FitResults.__init__` and
  `merge_to_stable` do it) or downstream `.spectral`/`.itable` calls will
  fail with an actionable `ValueError` rather than silently misbehave.
- **Process-pool-bound arguments must be picklable** (`reproj_s3_async.py`
  runs `_process_buffer` in a `ProcessPoolExecutor`): a `zodi_corrector`
  passed to `run_reproj_tasks` must be a real function (e.g.
  `functools.partial`), never a lambda or closure, when `num_workers > 0`.
- `spherex_tabular_bandpass/` at the repo root is an untracked, optional
  separate git checkout of the full (non-lite) bandpass package, used only
  so `tests/test_bandpass_lite.py::test_matches_full_bandpass_model` can
  cross-check `Tabular_Bandpass_Lite` against it; that test class
  self-skips when the package isn't installed.
- `build/` and stray root-level data files (`*.fits`, `*.parquet`, `*.csv`,
  etc.) are gitignored analysis/build artifacts, not tracked inputs.

## History and rationale

`ssiaat_improve_plan.md` is a large, mostly-completed design log (bug fixes,
robustness, UX, dedup, performance phases, each phase later marked done) —
check it before re-deriving *why* something is shaped the way it is (e.g.
why pipeline versions sort numerically, why the async collector never lets
an exception propagate, why indices were unified into `indexing.py`).
`table_size_possible_reduction.md` is a standalone, not-yet-implemented
analysis of where stable-parquet bytes go and what's cuttable.
