# Stable parquet size: possible reductions

Analysis date: 2026-07-27. Status: **findings only, not yet implemented.**

## Motivation

ssiaat stables are large — one (tile, band) parquet from the HiPS pipeline
is 30–280 MB, ~1.15 GB for all 6 bands of a single galactic-plane tile.
Downloading them from S3 (us-east-1) to a machine in Korea runs at only
~5–8 MB/s on a good window, so file size directly sets download time.
This doc records where the bytes go and how much we could cut.

## Does Parquet support lossy compression?

**No.** Every Parquet codec is lossless: `UNCOMPRESSED`, `SNAPPY`, `GZIP`,
`BROTLI`, `LZ4`/`LZ4_RAW`, `ZSTD`. There is no JPEG-style lossy codec.

Lossy size reduction is only possible by changing **what you store** before
writing (lower precision, quantization, dropping derivable columns); the
codec then losslessly packs the smaller data.

## Measured baseline

One tile-band, `hpx6_22933_512_b5.parquet`: **277.9 MB**, 13,617,364 rows,
codec **SNAPPY**. Per-column compressed size and compression ratio:

| column   | dtype        | comp MB | ratio | note |
|----------|--------------|--------:|------:|------|
| wvl      | float64      |   112.3 | 1.00  | **biggest; float64; derivable** |
| image    | float32      |    61.9 | 1.00  | flux |
| variance | float32      |    61.8 | 1.00  | |
| src_x    | int32        |    18.9 | 1.00  | range 1–2038 |
| src_y    | int32        |    18.7 | 1.01  | range 1–2038 |
| tmpl_ind | int32 (index)|     1.9 | 1.00  | |
| expidn   | int64        |     1.4 | 8.18  | repetitive → compresses |
| pipe_run | string       |     0.9 | 6.46  | repetitive → compresses |
| band     | int64        |     0.1 | 0.97  | |

**Key observation:** SNAPPY achieves ratio ≈ 1.0 on every large column —
float/int measurement data is high-entropy and does not compress. So the
file size is essentially raw column bytes, and the only levers are dtype
width, encoding, a stronger codec, or dropping redundant columns.

## Reduction levers

Estimates are for this 277.9 MB file; they scale to the whole campaign and
to download time.

### Lossless (no scientifically meaningful loss)

| change | saving | notes |
|--------|-------:|-------|
| `wvl` float64 → **float32** | −56 MB | wvl needs ~5 sig figs; float32 gives ~7. Effectively lossless. |
| `src_x`, `src_y` int32 → **int16** | −19 MB | values 1–2038 fit int16 (max 32767). |
| codec **ZSTD** + `byte_stream_split` on float cols | ~−20–40 MB | byte-split regroups mantissa/exponent bytes so ZSTD finds structure SNAPPY can't. Applies to image/variance/wvl. |

Lossless subtotal: **277.9 → ~180–200 MB (≈30–35% smaller)**, exact
round-trip for image/variance/ints, and wvl within float32 tolerance.

### Lossy (data-level precision reduction — the only way to get "lossy" in Parquet)

| change | saving | precision cost |
|--------|-------:|----------------|
| `image`, `variance` float32 → **float16** | −62 MB | round-trip relative error: **median 1.7e-4, 99th pct 4.3e-4** (~0.02%). Flux range −15…153 fits float16 (max 65504). Almost certainly below per-pixel noise. |
| **drop `wvl`**, recompute on read | −112 MB | none numerically, but couples reads to the bandpass model `(src_x, src_y, band) → wvl`. |

Lossless + float16 image/variance: **277.9 → ~140 MB (≈50% smaller)**.

## Where the changes live (write-side)

- **Column dtypes:** `ssiaat/src/ssiaat/reproj.py`, `_convert_hdul_to_df`.
  `image`/`variance` are already cast to `float32`; `wvl` comes out of the
  bandpass model as `float64`; `src_x`/`src_y` from `get_src_yx`. Cast wvl
  to float32 and src to int16 here.
- **Codec / encoding:** the `to_parquet` calls — `reproj.merge_to_stable`
  (or the pipeline's `stable.to_parquet`). Pass
  `compression="zstd"` and (via pyarrow) `use_byte_stream_split=[...]` for
  the float columns, or `use_dictionary=False` where it doesn't help.

## Recommendation

1. **Do the lossless set first** (wvl→float32, src→int16, ZSTD+byte_stream_split)
   — a safe ~30–35% cut, no scientific decision required. Add a test that the
   round-trip is exact for image/variance/ints and within float32 tolerance
   for wvl.
2. **Offer float16 image/variance as an opt-in flag** — whether ~0.02% flux
   error is acceptable is a scientific call, not a default.
3. Dropping `wvl` entirely is the largest single win (−112 MB) but trades
   storage for read-time coupling to the bandpass model; revisit only if
   size is still binding after (1).
