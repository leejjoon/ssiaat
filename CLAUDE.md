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
notebook-style cell markers) and one validating-but-unused `det_id` lookup
in `reproj.py`'s `ingest_hdul` (an `F841`) — check `ruff check src tests`
output before assuming a fresh finding is new. (The old `F821`/`F841` in
`model/sed.py`'s `channel_model_deprecated`/`test()` demo code are gone —
that dead code was removed when `sed.py` became a facade, see stage 5.)

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
5. **Model fitting** (`model/`) — `shapes.py` has the profile-shape
   primitives (smooth top-hat-like functions built from `tanh_step`:
   `hflatline`, `cont_left/right`, `get_cont_model`, ...). `lines.py` is the
   line-model *registry* — the single inventory of SPHEREx lines/features:
   `LINES` (name → `LineModel`) and `FEATURES` (name → `Feature`, carrying its
   continuum window), with one `bandpass_line()` factory over
   `BandpassShapeLite` and `get_line()`/`get_line_model()`/`feature_spec()`
   builders. Bandpass lines carry a `prior_offset` (the explicit replacement
   for the old magic `d_shift`); legacy analytic shapes (`br_a`, PAH, H2O-ice,
   ...) are registered `kind="shape"` reproducing the old getters verbatim.
   `sed.py` is now a thin back-compat facade: it re-exports the shapes and
   keeps `get_br_a`/`get_pa_b`/`get_pah`/`get_h2o_ice` as *deprecated* shims
   delegating to the registry. `Model`
   bundles named line + continuum callables, `least_square_fit` factorizes
   the stable's (duplicated) pixel index into per-pixel groups and calls
   `vectorized_lstsq.py`'s numpy-vectorized batched least squares (one
   independent small linear solve per pixel, batched — not a Python loop per
   pixel). `FitResults` wraps the coefficients as pixel-indexed `pd.Series`
   (so `.itable.to_image()` works directly on any coefficient) and carries
   the template header through so a fitted map round-trips like a stable.
   `BandpassTool` (`spherex_table.py`) is a separate, smaller piece: given
   a *central wavelength* instead of a detector pixel, it inverts the
   central-column wavelength map (built from the bandpass model's
   `central_bandpass_only=True` mode) to find the row, then fetches the
   real bandpass there (`get_bp_at_wvl`; needs the full
   `spherex_tabular_bandpass.Tabular_Bandpass`, not the Lite model, since
   Lite only implements `central_bandpass_only`). `fit_hflatline` goes one
   step further and fits `sed.hflatline`'s shape to that real bandpass by
   least squares (amplitude included as a free parameter, not read off
   the raw peak — see its docstring for why) — useful for getting a
   compact analytic stand-in for a tabulated instrumental response. Its
   docstring also covers a real parameter degeneracy: with
   `mu_center_left` left free and asymmetric edges, it trades off against
   `dmu_left`/`dmu_right` (measured `corr(dmu_left, a_left) = +1.000`
   within a 4-pixel window on the real bandpass) — pass `mu_center=`
   whenever it's already known (e.g. from `Tabular_Bandpass_Lite`) to
   remove the degeneracy at *no* fit-quality cost (measured identical to 4
   sig figs, fixed vs. free); prefer that over `symmetric=True`, which
   also removes the degeneracy but at a real accuracy cost.
   `fit_hflatline_at_iy` is the row-indexed variant used for bulk sweeps
   (e.g. `examples/derive_bandpass_shape_coeffs.py`): it fetches the curve
   and the analytic center from the same integer detector row directly,
   avoiding a wavelength→row inversion that can land one row off near
   template-quantization boundaries.

`model/multiband_lines.py`'s `MultiBandLineModel` is a different shape of
fitting problem `Model`/`vectorized_lstsq` can't express: ONE spectrum
split across bands that may not share a wavelength solution, where a
line's center should be ONE value shared across every band it appears
in while its amplitude is free per band (`Model`'s per-group solves are
independent by construction — nothing ties a parameter across them).
Solved as variable projection: `scipy.optimize.least_squares` searches
only the (few) tied centers/per-band offsets, with a fresh weighted
linear solve for every amplitude+baseline coefficient inside each
residual evaluation. The line-profile SHAPE is never an independent free
parameter — always read off `BandpassShapeLite.params(trial_center,
band)` at whatever the current trial center is, per `fit_hflatline`'s
degeneracy warning above. Two real, non-obvious things it protects
against, both discovered by prototyping this against a real stacked
spectrum: (1) correlated stack noise makes naive `1/error**2`-weighted
errors wrong by a large, band-dependent factor — `null_mad` recalibrates
via an empirical residual-scan null, not an assumed-covariance GLS
correction (tried, numerically unstable here); (2) a weak/absent line's
blind initial-position search can wander into a much stronger nearby
line's wing and lock onto that instead — both `prescan` and the final
fit bounds cap a line's search window at 0.4x its distance to the
nearest other candidate line in the same band. `blended=True` in the
results flags when two lines' fitted centers land within half a
resolution element of each other anyway (a real, not fixable,
identifiability limit — e.g. He I 1.083µm and Pa-γ 1.094µm in SPHEREx
band 1) — their individual amplitudes are then design-matrix-collinear
and shouldn't be trusted independently, only summed.

`zodi_correction.py` and `tabular_bandpass_lite.py` are inputs consumed by
stage 3 (zodiacal-light subtraction and detector-pixel → wavelength lookup,
respectively); `tabular_bandpass_lite.py` is a frozen from-coefficients
re-implementation of the full `spherex_tabular_bandpass` package (not a
dependency) supporting only `central_bandpass_only=True`.

`bandpass_shape_lite.py`'s `BandpassShapeLite` does the analogous
dependency-free trick for the *full bandpass curve*, not just the central
wavelength: `sed.hflatline`'s shape (via `fit_hflatline`, `mu_center`
fixed) plus a small additive `floor0 + floor1*(w - mu_center)` term —
needed because the real tabulated bandpass carries a genuine ~0.01–0.1%-
of-peak broadband transmission floor across the whole band that a
single-peak shape can't otherwise represent — each fit as a low-order
Chebyshev polynomial of wavelength, one set of coefficients per band,
central column only. Wavelength-first API (`model(center, array=band)`),
unlike `Tabular_Bandpass_Lite`'s pixel-first one, since with `mu_center`
fixed the shape parameters are polynomials in wavelength directly — no
`interp1d` inversion needed. Outside each band's validated domain
(`SHAPE_CHEB_DOMAIN`) it extrapolates and emits a `UserWarning` rather
than raising. Provenance (derivation script, date, trim rule, per-band
degree and error budget) is recorded in the module docstring; regenerate
via `examples/derive_bandpass_shape_coeffs.py`, which also documents the
floor-term finding and a noise-floor-aware validation gate (the real
tabulated data has its own row-to-row centroid noise — see
`measure_centroid_noise_floor` — that a smooth model can't beat and
shouldn't be held to).

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
