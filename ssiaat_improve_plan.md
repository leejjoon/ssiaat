# ssiaat Improvement Plan

Review date: 2026-07-07. Based on a full read of `src/ssiaat/`, the tracked
examples, and the untracked analysis scripts (`cygx_spec.py`, `fit_ice.py`,
`async_collector.py`, ...) that show how the package is used in practice.

The package is pre-1.0 and API compatibility is not a constraint. Focus areas:
correctness bugs first, then robustness, user experience, and performance.

---

## Phase 0 — Test suite (prerequisite for verifying every fix below)

There are currently no tests. Almost everything is testable with *synthetic*
data — no real SPHEREx files needed except for optional validation tests.

### 0.1 Three kinds of tests
1. **Known-answer tests** (highest value): build a tiny template (16×16 WCS
   via `get_wcs`), synthesize a stable in memory (`tmpl_ind` with duplicates,
   `wvl` grid, `image = 2.0*model_a(wvl) + 0.5*cont(wvl) + noise`) and assert
   `Model.least_square_fit` recovers (2.0, 0.5). Protects the whole Phase 5
   refactor chain. Same idea for `SsiaatConverter`:
   `itable_to_image`/`image_to_itable` round-trip identity (incl. NaN, masks,
   big-endian FITS input).
2. **Regression pins before refactoring**: run the *current*
   `vectorized_lstsq_numpy` on fixed synthetic input, assert the rewritten
   version matches (`assert_allclose`). The inline CSV in
   `vectorized_lstsq.example()` converts to pytest nearly for free.
3. **Bug-first failing tests** for Phase 1–2 (write before fixing):
   - version sort: fake tree with `l2b-v9/v19/v20` → assert v20 wins
     (fails today);
   - async deadlock: processor raising on one item, wrapped in
     `asyncio.wait_for(..., timeout=5)` → run completes and reports the
     failure instead of hanging (fails today);
   - `get_df_from_uri` without pbar (NameError today).

### 0.2 Testing the awkward parts without a network
- **finder/fsspec**: no S3 mocking — point `find_latest_uri` at a `tmp_path`
  local directory with the real `level2/{plan}/{pipe_ver}/{band}/{file}`
  layout built from empty files (fsspec `file://`; `memory://` for the pure
  async branch).
- **reprojection**: one `@pytest.mark.slow` integration test: synthetic L2
  HDUList (2040×2040 constant image, real WCS, zero FLAGS, unit VARIANCE,
  `L2DQAFLG` keyword) → `process_single` onto a small template → output ≈
  constant, `tmpl_ind` decodes via `get_src_yx`, `hdul_to_pandas` →
  `promote_to_stable` → `make_simple_image` reproduces the constant.
- **parquet metadata round-trip**: `to_parquet` → `read_stable` → assert
  `df.spectral.converter.tmpl_shape` reconstructed. The accessor UX depends
  entirely on `df.attrs` surviving parquet; this guards against pandas/pyarrow
  version changes.
- **tabular_bandpass_lite**: keep the comparison against the full
  `spherex_tabular_bandpass` as `skipif(not installed)`; add an unconditional
  test pinning a few hardcoded known wavelengths per band.

### 0.3 Layout
```
tests/
  conftest.py            # fixtures: template_wcs, template_header, synthetic_stable,
                         #           synthetic_l2_hdul, stable_parquet(tmp_path)
  test_flags.py          # get_flagval include/exclude/ALL logic
  test_sed.py            # hflattop ≈1 inside / ≈0 outside, continuity
  test_wcs_helper.py     # get_wcs shape/scale; TemplateHeaderCards header<->df<->hash
  test_finder.py         # filename parsing; version sort (bug-first); local-fs find
  test_spherex_table.py  # converter roundtrips; accessors; promote/read_stable;
                         #   parquet attrs roundtrip; broadcast; filter_with_image_mask
  test_lstsq.py          # known-answer fit; chunked==unchunked; weighted/unweighted
  test_model.py          # Model+FitResults end-to-end on synthetic stable
  test_async.py          # AsyncCollector: results, failures, no-deadlock (timeout)
  test_reproj.py         # @slow synthetic end-to-end reprojection
```
`pyproject.toml`: add `[project.optional-dependencies] test = ["pytest",
"pytest-asyncio"]` and a `slow` marker (`pytest -m "not slow"` as the fast
default, well under a minute).

### 0.4 Order of work
1. `conftest.py` fixtures + converter round-trip + lstsq known-answer +
   parquet attrs round-trip — the safety net required before Phase 5.
2. Bug-first tests for 1.1/1.2, 2.1, 1.5 — then fix each.
3. The rest opportunistically as each plan item is touched (1.4, 3.8, 5.1
   generate natural test material).

Note: `check_index_stable` requires a duplicated index, so synthetic stables
in fixtures need ≥2 wavelength rows per pixel — comment this in the fixture
so it doesn't get "simplified" away later.

---

## Phase 1 — Correctness bugs (small, high impact; do first)

### 1.1 Pipeline-version sorting is lexical, so `v9` beats `v20`
- **Where:** `src/ssiaat/finder.py:126` (`pipe_vers.sort(reverse=True)`).
- **Problem:** Versions are compared as strings; `'l2b-v9-...' > 'l2b-v20-...'`.
  As soon as single- and double-digit versions coexist in a plan directory,
  "find latest" silently returns the wrong (older) file.
- **Fix:** Add a `_parse_pipe_version(name) -> (int_version, date_tuple)`
  helper (regex `l2b-v(\d+)-(\d+)-(\d+)`) and sort with it as the key.
  Use the same helper everywhere versions are ordered.

### 1.2 `find_local_uri` returns the *oldest* version
- **Where:** `src/ssiaat/finder.py:84-93`.
- **Problem:** Candidates are sorted ascending and the first existing file is
  returned — the opposite of "find latest".
- **Fix:** Reuse the key from 1.1 with `reverse=True`. Add a small unit test
  with a fake directory tree covering v9/v19/v20.

### 1.3 `reproj_hips.py` cannot be imported at all (verified)
- **Where:** `src/ssiaat/reproj_hips.py:12,14,26,29`.
- **Problems:**
  - `from .tabular_bandpass_lite import Tabular_Bandpass_lite` — wrong case;
    the class is `Tabular_Bandpass_Lite`. Confirmed ImportError.
  - `from pandas.io.gbq import find_stack_level` — IDE auto-import artifact;
    `pandas.io.gbq` no longer exists in modern pandas.
  - Imports `spherex_utils` (`utils`, `mosaic_utils.get_flagval`), which is not
    a declared dependency; `reproj.py` already has local copies.
- **Fix:** Delete the stray imports; fix the class-name case; import
  `_ingest_hdul`, `get_flagval`, `DEFAULT_FLAGS` from `.reproj`/`.flags`
  instead of `spherex_utils`. Delete the ~130-line duplicated copy of
  `ingest_hdul` in this file (see 4.2).

### 1.4 `BandpassTool` has never run
- **Where:** `src/ssiaat/spherex_table.py:291-313`.
- **Problems:** `wwl` undefined (should be `wvl`) at line 300;
  `self.bandpaas_model` typo at line 308.
- **Fix:** Fix both typos and add a smoke test — or, if it is not used
  anywhere, remove the class (preferred until it is actually needed).

### 1.5 `get_df_from_uri` crashes when no progress bar is passed
- **Where:** `src/ssiaat/reproj.py:360` (`print(fn)`; variable is `uri`).
- **Fix:** `print(uri)` — or better, drop the print and use `logging.info`.

### 1.6 Dead code with undefined names
- **Where:**
  - `reproj.py:133` `check_overwrapp()` — references `self` at module level.
  - `finder.py:103` `get_local_path` — calls nonexistent `get_readpath`.
  - `query.py:25` `check_weeks` — undefined `rootdir`, `dff`.
  - `spherex_table.py:444` `get_test_model` — `from sed import ...` should be
    `from .model.sed import ...`.
- **Fix:** Delete `check_overwrapp`, `get_local_path`, `check_weeks` (keep the
  intent as a TODO note if desired). Move `get_test_model` + `test_save` /
  `test_load` / `main()` blocks into `examples/` with the import fixed
  (see 3.8).

---

## Phase 2 — Robustness of the network/async layer

### 2.1 A single bad file deadlocks the async reprojection run
- **Where:** `src/ssiaat/reproj_s3_async.py:62-69` (`ProjectorRunner.worker_s3`).
- **Problem:** If `get_df_from_uri` raises (corrupt FITS, missing ZODI
  extension, network error), the worker task dies before `task_done()`;
  `queue.join()` then waits forever. User sees a frozen progress bar, no error.
- **Fix plan:**
  1. Move the untracked `async_collector.py` (`AsyncCollector`) into the
     package as `src/ssiaat/async_collector.py` — it already handles
     `task_done()` in a `finally`.
  2. Rebuild `ProjectorRunner` on top of `AsyncCollector`.
  3. In the per-item processor, catch `Exception`, append `(uri, exc)` to a
     `failures` list, and return None so the run continues.
  4. At the end, log a summary ("N succeeded, M failed") and return/expose the
     failure list so the user can retry just those URIs.
  5. Remove the hardcoded `uri = f"s3://{uri}"` and `anon=True`; take full URIs
     and pass `storage_options` through.

### 2.2 Silent exception swallowing in the finder
- **Where:** `src/ssiaat/finder.py:142` (`except Exception: return None`).
- **Problem:** Credentials errors, bad root URIs, and genuinely missing files
  all collapse into "not found".
- **Fix:** `logging.warning("finder failed for %s: %r", row['filename'], e)`
  at minimum; preferably return a per-file status (`found` / `missing` /
  `error(exc)`) so `find_latest_uri` can report the three cases separately.

### 2.3 `find_latest_uri` sync wrapper breaks in Jupyter
- **Where:** `src/ssiaat/finder.py:170-180`.
- **Problem:** `loop.run_until_complete` raises
  `RuntimeError: This event loop is already running` inside notebooks — and
  the workflow is heavily notebook-based. `asyncio.get_event_loop()` is also
  deprecated.
- **Fix:** In the sync wrapper:
  - If no loop is running: `asyncio.run(...)`.
  - If a loop *is* running: raise a clear error telling the user to
    `await find_latest_uri_async(...)` (mention it in the docstring and
    README). Do not add `nest_asyncio` magic.

---

## Phase 3 — User experience

### 3.1 Make `import ssiaat` enough (biggest UX win)
- **Problem:** Users must know internal module paths
  (`from ssiaat.spherex_table import read_stable`), and the `.spectral` /
  `.itable` accessors are only registered if `ssiaat.spherex_table` happens to
  have been imported — otherwise `df.spectral` gives pandas' unhelpful generic
  AttributeError.
- **Fix:** Populate `src/ssiaat/__init__.py` with the public API:
  ```python
  from .spherex_table import (read_stable, promote_to_stable,
                              SsiaatConverter, Model, FitResults)  # registers accessors
  from .reproj import SphxReprojector, get_df_from_uri, merge_to_stable
  from .query import query_overlapping
  from .wcs_helper import get_wcs, TemplateHeaderCards
  from .finder import find_latest_uri, check_uri
  __all__ = [...]
  ```
  Keep `reproj_hips` out of the top level (heavier deps) until it is cleaned up.

### 3.2 Unify the two `read_stable`s and accept lists
- **Where:** `spherex_table.py:211` (module-level, `index_column=None`) vs
  `spherex_table.py:263` (`SsiaatConverter.read_stable`,
  `index_column="tmpl_ind"`).
- **Problems:** Two near-identical implementations with *different* defaults;
  and `cygx_spec.py:9` passes a list to the varargs API (works only because
  pyarrow tolerates list paths).
- **Fix:**
  - One implementation (module-level); the method delegates to it, adding
    `header=self.header` and caching the converter.
  - Normalize input: `read_stable("a.pq", "b.pq")` and
    `read_stable(["a.pq", "b.pq"])` both work.
  - Same `index_column="tmpl_ind"` default in both (set-index is skipped when
    the index is already `tmpl_ind`).
  - Add `columns=` and `wvl_range=` pass-through (see 5.3).

### 3.3 Add the accessor helpers that scripts keep rewriting
Evidence from the untracked analysis scripts:
- `cygx_spec.py:16-17` re-implements `make_simple_image` by hand because it
  needs the itable (groupby-mean Series), not the image.
- `cygx_spec.py:100-111` defines `median_spec` (wavelength-binned median
  spectrum) — needed in essentially every analysis.

**Fix:** extend `SpectralTable` (`df.spectral`):
- `make_simple_itable(w1, w2, column="image", agg="mean")` — the groupby
  aggregation without reprojection; `make_simple_image` becomes a thin wrapper
  (`agg` parameter added there too).
- `binned_spectrum(w1=None, w2=None, column="image", bins=50, agg="median")`
  → returns a Series indexed by bin-center wavelength (plot-ready).
- Document `spectral.broadcast` with a recipe replacing the
  `join(..., rsuffix="_cont")` dance in the README.

### 3.4 `converter` returning None causes confusing downstream crashes
- **Where:** `spherex_table.py:83-99,137-152`.
- **Problem:** When header metadata is missing, `.converter` silently returns
  None; the user then hits `AttributeError: 'NoneType' object has no attribute
  'itable_to_image'` inside `make_simple_image`.
- **Fix:** Raise immediately in the `converter` property with the actionable
  message already used by `promote_to_stable` ("no template header metadata in
  df.attrs; run promote_to_stable(df, header=...)").

### 3.5 `FitResults` / `Model` ergonomics
- **Problems:** Coefficients are positional (`fitted.C[0]`, `fitted.contC[1]`);
  model names are auto-generated (`model0`, `cmodel0`); `FitResults.__init__`
  defaults `model=None` then dereferences it unconditionally.
- **Fix:**
  - `Model` accepts `{"pah_narrow": hflattop(...), ...}` dicts (or a list of
    `(name, callable)`); keep list input working with auto-names.
  - `FitResults.coef(name)` / `FitResults.err(name)` lookup by name;
    `FitResults.to_frame()` returning a DataFrame with named columns
    (also gives trivial parquet serialization of fit results).
  - Make `model` a required positional argument.
  - Optional convenience: `FitResults.image(name)` → itable → Image using the
    stored template header.

### 3.6 Fix declared dependencies
- **Where:** `pyproject.toml`.
- **Missing:** `scipy` (used in `spherex_table.py`, `zodi_correction.py`),
  `pyarrow` (every parquet read/write), `numpy` (explicit).
- **Unused:** `boto3`, `botocore`, `aioboto3` (only broken `reproj_hips.py`
  touches boto3; s3 access goes through `s3fs`). Drop them.
- Re-check `fsspec`/`s3fs` `<=2025.3.0` pins (colab workaround) — keep but add
  a comment with the reason and a date to revisit.

### 3.7 README quickstart for the actual pipeline
- **Problem:** README currently documents three scripts that are not in the
  repo. The real end-to-end flow is undocumented:
  `query_overlapping` → `find_latest_uri` → `SphxReprojector` /
  `get_df_from_uri` → parquet "stable" → `read_stable` → `.spectral` →
  `Model.least_square_fit` → `.itable.to_image()`.
- **Fix:**
  - ~30-line quickstart in README walking one target through all steps
    (base it on a cleaned-up `examples/process_eso_244.py`).
  - Glossary paragraph defining the jargon: **stable** = *s*pectral *table*
    (long-format pixel × wavelength rows, integer `tmpl_ind` index with
    duplicates), **itable** = image table (unique pixel index), **template** =
    the WCS grid. `promote_to_stable` reads like English "stable" to a
    newcomer; a two-line definition removes the confusion.
  - Fold `MAKEHIPS.md` into the docs (or `docs/`) once `reproj_hips` imports.

### 3.8 Move demo/test code out of library modules
- **Where:** `spherex_table.py:443-558` (`get_test_model`, `test_save`,
  `test_load`, `main` — ~120 of 558 lines, hardcoded local filenames,
  `pyregion` import), `reproj.py:388-405` (`main`), `finder.py:219-248`
  (`test_s3`), `query.py:50-65` (`main`).
- **Fix:** Move runnable demos to `examples/`; convert assertion-style tests
  (`vectorized_lstsq.example`, `tabular_bandpass_lite.test`) into a real
  `tests/` directory runnable with pytest (they already contain usable inline
  data).

### 3.9 Small polish
- `ImageTable.to_image` docstring says "Converts the image back to its tabular
  form" — copy-paste; it is the reverse direction.
- "overwrap" → "overlap" everywhere (comments, messages, `check_overwrapp`).
- `spherex_table.py` module docstring example says
  `converter = SsiaatConverter("template.fits")` but the constructor takes a
  `fits.Header`; should be `SsiaatConverter.from_file(...)`.
- Split `spherex_table.py` (converter + accessors vs fitting): move `Model`,
  `FitResults` into `ssiaat/model/` next to `sed.py` / `vectorized_lstsq.py`.

---

## Phase 4 — Code de-duplication

### 4.1 Pixel-index computation exists in 4+ places
- **Where:** `SsiaatConverter.__init__`, `_convert_hdul_to_df`,
  `SphxReprojector.get_ind_image`, `reproj_hips.SphxHpxProcess.__init__`
  (and re-derived by hand in user scripts, e.g. `fit_ice.py:38`).
- **Fix:** One helper in `wcs_helper.py` (or new `indexing.py`):
  `make_pixel_index(shape, stride=None, dtype="int32")` where
  `stride=shape[-1]` for template indices and `stride=2048` for detector
  source indices. Keep `get_src_yx` (bit-shift decode) next to it, and name
  the two index kinds distinctly (`tmpl_ind` vs `src_ind`) in docstrings.

### 4.2 `ingest_hdul` duplicated verbatim
- **Where:** `reproj.py:_ingest_hdul` and `reproj_hips.py:ingest_hdul`
  (~130 identical lines).
- **Fix:** Keep one in `reproj.py` (make it public: `ingest_hdul`), import it
  from `reproj_hips.py`.

### 4.3 `get_wcs` / `get_wcs_from_shape` near-duplicates
- **Where:** `wcs_helper.py:9-71`.
- **Fix:** Single `get_wcs(lon, lat, side=None, side2=None, shape=None, ...)`
  where exactly one of `side` / `shape` must be given; shared body.

---

## Phase 5 — Performance

### 5.1 `vectorized_lstsq_numpy`: avoid giant float64 intermediates
- **Where:** `src/ssiaat/model/vectorized_lstsq.py:23-39`.
- **Problem:** For M total models it materializes M(M+1)/2 + M full-length
  product Series (upcast to float64), then a DataFrame, then a pandas
  groupby-sum. On a 100M-row stable with 4 models that is ~11 GB of
  intermediates — the reason `vectorized_lstsq_chunked` had to exist
  (README: "attempt to fit in batch to reduce memory footprint").
- **Fix:**
  1. Factorize the group labels once:
     `codes, uniques = pd.factorize(df.index, sort=True)` (or
     `df[group_column]`).
  2. For each product, compute a float64 numpy array and accumulate with
     `np.bincount(codes, weights=product, minlength=len(uniques))`; free it
     before the next one. Peak extra memory ≈ one row-length array.
  3. Fill ATA/ATy exactly as now; keep `pinv` (M is small; robustness beats
     the modest gain of `solve`).
  4. Keep `vectorized_lstsq_chunked` as a thin wrapper initially; deprecate it
     once the bincount version is validated. Validate against the existing
     inline example data (move to `tests/`).
- **Expected:** ~5–10× faster aggregation, near-constant memory; chunking no
  longer needed.

### 5.2 `Model._populate_table_with_model_eval` duplicates the largest columns
- **Where:** `spherex_table.py:393-412`.
- **Problem:** The non-inplace path copies `wvl`, `image`, `variance` into a
  second full-length DataFrame just so the fitter can find them by name.
- **Fix:** Change the fitting path to pass arrays: evaluate each model into a
  numpy array (dict `name -> ndarray`), and extend `vectorized_lstsq_numpy` to
  accept `(model_arrays, target_array, variance_array, codes)` directly.
  DataFrame round-trip disappears; combined with 5.1 the fit is one pure-numpy
  pass over the table.

### 5.3 Push filters down into parquet reads
- **Where:** `read_stable` + every analysis script's immediate
  `stable.query("(w1 < wvl) and (wvl < w2)")`.
- **Fix:** In the unified `read_stable` (3.2):
  - `columns=[...]` → `pd.read_parquet(fn, columns=...)` (most analyses never
    touch `src_x`/`src_y`).
  - `wvl_range=(w1, w2)` → `filters=[("wvl", ">", w1), ("wvl", "<", w2)]`
    (pyarrow predicate pushdown).
  - When writing stables (`merge_to_stable` → `to_parquet` in examples),
    sort by `tmpl_ind` first so row-group statistics make later spatial
    cutout reads cheap; document a recommended `row_group_size`.
- **Expected:** Large cuts in I/O and peak memory for band files that span
  wavelengths beyond the analysis window.

### 5.4 Parallelize reprojection on the right axis (CPU, not I/O)
- **Where:** `reproj_s3_async.py`; `reproj.py:262-266`.
- **Problem:** `reproject_adaptive` is CPU-bound; asyncio workers serialize on
  the GIL (the module docstring itself notes the speedup is "not
  significant"). Per-exposure work is embarrassingly parallel.
- **Fix:**
  1. Keep async (or threads) for the S3/HTTP fetch only.
  2. Run `SphxReprojector(...).process_single(...) + hdul_to_pandas(...)` in a
     `concurrent.futures.ProcessPoolExecutor` via
     `loop.run_in_executor(pool, ...)` from the fetch workers
     (producer/consumer: fetch → process pool → collect).
  3. Expose `num_fetchers` and `num_workers` in `run_s3_repoj_tasks` (rename
     `run_s3_reproj_tasks`); default `num_workers=os.cpu_count()-1`.
  4. Expose `**reproject_kwargs` from `process_single` instead of hardcoding
     `parallel=False` (reproject's own block-parallel mode then becomes
     available for the single-large-template case).
- **Expected:** near-linear scaling with cores for the bulk-reprojection step,
  which is the slowest part of the whole pipeline.

### 5.5 Minor
- `finder._get_table_from_filenames`: replace four `.apply(lambda)` passes
  with vectorized `.str` operations (only matters for very long file lists).
- `_ingest_hdul`: `hdulist[0].data is not None` forces a full data load just
  to check presence; use `image_hdr["NAXIS"] > 0` / `hdulist[0].header` first.
- `SsiaatConverter.itable_to_image`: for large templates, preallocate and
  assign (`out = np.full(n, nan); out[idx_positions] = values`) instead of
  `reindex` over the full template (several × faster at HiPS scales; keep
  `reindex` path for correctness reference).

---

## Suggested execution order

| Step | Items | Rationale |
|------|-------|-----------|
| 0 | 0.1–0.4 (core fixtures + safety-net tests) | Everything below is verified against these |
| 1 | 1.1–1.6 | Correctness; 1.1/1.2 silently select wrong data versions |
| 2 | 2.1–2.3 | The network step is the slowest and currently fails opaquely |
| 3 | 3.1, 3.2, 3.6 | One import, one `read_stable`, installable package |
| 4 | 3.3–3.5, 3.7–3.9, 4.1–4.3 | UX + structure cleanup (enables docs) |
| 5 | 5.1–5.3 | Memory is the binding constraint per README notes |
| 6 | 5.4, 5.5 | Throughput of bulk reprocessing |

Each step should land as its own commit (or small series), with the moved
inline examples turned into pytest tests as they are touched (1.4, 3.8, 5.1
all create natural test material). No API-compatibility shims needed — the
package is pre-1.0 and the only known consumers are the local scripts, which
can be updated in the same commits.

---

## Decision log (grilling session, 2026-07-07)

Decisions confirmed with the author; these amend or pin down the items above.

1. **Scope:** full plan, steps 0–6.
2. **CI (new item):** add a minimal GitHub Actions workflow in Phase 0 — one
   job, one Python version, `pip install -e .[test]` + `pytest -m "not slow"`
   on every push. No matrix, no coverage tooling.
3. **Test cadence:** §0.4 step 1 safety net lands as the first commit; after
   that, *every* plan item's commit must include its tests ("opportunistic"
   is upgraded to mandatory per commit).
4. **1.4 amended:** `BandpassTool` is **kept**, not deleted — fix the two
   typos (`wwl`→`wvl`, `bandpaas_model`) and add a smoke test.
5. **reproj_hips stays second-class:** do §1.3 + §4.2 (imports fixed,
   `ingest_hdul` deduped, import-smoke test); no top-level export, full
   cleanup deferred until the next HiPS production run.
6. **2.1 failure contract:** bulk runs never raise on per-item errors —
   complete the run, return results plus a `failures: list[(uri, exception)]`,
   log a loud end-of-run summary. No automatic retry.
7. **AsyncCollector contract:** exception-catching lives *inside*
   `AsyncCollector._worker` (append `(item, exc)` to `self.failures`, worker
   survives) so deadlock is structurally impossible for every consumer.
   AsyncCollector becomes public API, exported from the top level.
8. **2.3 as planned:** sync `find_latest_uri` raises with clear
   "use `await find_latest_uri_async(...)`" guidance inside a running loop;
   no nest_asyncio, no worker-thread magic.
9. **Naming stays:** stable/itable vocabulary and `promote_to_stable` kept;
   README glossary is the fix.
10. **3.5 fit API:** dict-based named models + `coef(name)`/`err(name)`/
    `to_frame()`, `model` required — and `.C`/`.contC` arrays remain public
    so existing scripts keep working.
11. **3.6 pins:** Colab is still a target — keep the fsspec/s3fs
    `<=2025.3.0` pins with a dated explanatory comment; drop boto3/botocore/
    aioboto3 (verified: only commented-out code in reproj_hips touches them);
    add scipy, pyarrow, numpy.
12. **5.1:** `vectorized_lstsq_chunked` becomes a thin wrapper and survives
    one real analysis cycle on full-scale data before deletion.
13. **5.4 as planned:** full fetch→ProcessPoolExecutor pipeline,
    `num_fetchers`/`num_workers` exposed, defaults (4, `cpu_count()-1`).
14. **Git workflow:** commits straight to `main`, one per plan item (or tight
    series), pushed at step boundaries so CI validates each step.
15. **Root hygiene (new item, Phase 0):** add `.gitignore` patterns for data
    artifacts (`*.fits`, `*.parquet`, `*.tgz`, `*.ecsv`, root `*.csv`, logs,
    `.ipynb_checkpoints/`); analysis scripts and notebooks stay untracked and
    untouched.

---

## Progress log

- **2026-07-07 — Phase 0 done** (5 commits): gitignore for root data
  artifacts; test extra + pytest config; tests/ safety net (23 tests:
  converter round-trips, attrs-through-parquet, lstsq known-answer +
  regression pin + chunked==unchunked); GitHub Actions CI (green).
- **2026-07-07 — Phase 1 done** (1.1–1.6 + 4.2, 5 commits):
  - 1.1/1.2: `_parse_pipe_version` numeric sort key in both finders,
    bug-first tests against a local `file://` tree; `find_local_uri`
    gained a `rootdir=` kwarg for testability.
  - 1.3/4.2: `reproj_hips` imports fixed (gbq/spherex_utils/boto3 gone,
    Tabular_Bandpass_Lite case fixed); duplicated `ingest_hdul` deleted,
    now imported from `reproj` (made public there).
  - 1.4 (amended: kept): `BandpassTool` typos fixed + smoke tests. Note:
    `get_bp_at_wvl` requires a full bandpass model; the Lite model only
    supports the `central_bandpass_only=True` path used by `__init__`.
  - 1.5 (downgraded): the `print(fn)` NameError had already been fixed in
    c2b0bcb; replaced `print(uri)` with `logging`. End-to-end coverage of
    `get_df_from_uri` lands with the Phase 2 synthetic-L2 fixture.
  - 1.6: `check_overwrapp`, `get_local_path`, `check_weeks` deleted (TODO
    note kept in query.py); demo blocks moved from spherex_table.py to
    `examples/fit_eso_244_demo.py` with the `sed` import fixed.
- **2026-07-07 — Phase 2 done** (2.1–2.3 + §0.2 fixture, 5 commits):
  - 2.1: `AsyncCollector` moved into the package with catch-in-worker
    failure handling (deadlock structurally impossible; `.failures` holds
    `(item, exc)` pairs). `reproj_s3_async` rebuilt on it:
    `run_reproj_tasks(uri_list, wcs_tmpl, *, num_tasks, progress,
    storage_options, zodi_corrector) -> (dfl, failures)`; full URIs, no
    hardcoded `s3://`/`anon=True` (s3 keeps anon as default only when no
    storage_options given); zodi correction now matches the sync path via
    the shared `reproj.get_df_from_buffer`. `ProjectorRunner` and
    `run_s3_repoj_tasks` deleted (no live callers; §5.4.3 rename done
    early). Top-level export of AsyncCollector lands with §3.1.
  - §0.2: synthetic 2040×2040 L2 fixture (IMAGE at HDU 1 + FLAGS/VARIANCE/
    ZODI, constant field) + end-to-end tests: `get_df_from_uri` over
    `file://` (zodi, columns, metadata, corrector callable) and
    `merge_to_stable` → `make_simple_image` reproducing the constant.
    Fast (~1.5 s), so not marked slow. Also closes the 1.5 coverage gap.
  - 2.2: finder logs a warning on unexpected errors (missing plan dir
    stays a quiet None) instead of swallowing everything.
  - 2.3: sync `find_latest_uri` uses `asyncio.run` when no loop runs and
    raises with "use `await find_latest_uri_async(...)`" inside Jupyter;
    deprecated `get_event_loop` gone.
