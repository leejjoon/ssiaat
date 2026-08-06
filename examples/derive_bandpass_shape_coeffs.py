#!/usr/bin/env python3
"""Derive a dependency-free polynomial parameterization of the SPHEREx
bandpass shape, central column (ix=1024) only.

For each band, this sweeps every detector row along the central column,
fits `ssiaat.model.sed.hflatline`'s shape with `mu_center_left` FIXED to
the already-known analytic central wavelength (see
`BandpassTool.fit_hflatline`'s docstring for why -- letting it float
creates a severe fit degeneracy; fixing it costs zero fit quality), then
fits low-order Chebyshev polynomials (in the central-wavelength domain) to
the remaining shape parameters and amplitude vs wavelength.

Requires the real `spherex_tabular_bandpass` package (NOT a declared
ssiaat dependency -- this script is provenance for hardcoded coefficients,
not something ssiaat's test suite or runtime imports). `--plot` additionally
needs matplotlib (also not declared).

The real tabulated bandpass carries a genuine, low-level (~0.01-0.1% of
peak) broadband transmission floor across the *entire* band -- not a
fitting artifact, confirmed by inspecting individual rows on a log scale.
hflatline (exactly zero outside its core) cannot represent it at any
polynomial degree, and it is small enough to barely affect curve RMS while
being large enough (because integral/centroid error are linear in the
residual and weighted by distance from center) to fail those two metrics
outright in several bands. The fix used here: fit a small additive
`floor0 + floor1*(w - mu_center)` term jointly with the hflatline shape
(2 extra free parameters per row, `floor0`/`floor1`, each getting their
own low-order polynomial vs wavelength just like the shape parameters) --
confirmed on the worst row (band 1) to drive both integral and centroid
error to ~0 while *improving* curve RMS. Disable with `--no-floor` to
reproduce the original (worse) hflatline-only numbers.

Usage
-----
    .venv/bin/python examples/derive_bandpass_shape_coeffs.py
    .venv/bin/python examples/derive_bandpass_shape_coeffs.py --bands 3 4 --plot
    .venv/bin/python examples/derive_bandpass_shape_coeffs.py --out sweep.npz
    .venv/bin/python examples/derive_bandpass_shape_coeffs.py --no-floor  # for comparison

Prints, per band: the trim window, the degree-selection table, the final
validation report (the Part C go/no-go gate), and correlation/ratio
diagnostics answering "can dmu_left/dmu_right and a_left/a_right be
merged into shared parameters?" (measured answer: no -- see the printed
ratios). Ends by printing a paste-ready coefficient dict for
`ssiaat/bandpass_shape_lite.py`.
"""
import argparse
import sys
import time

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import curve_fit

from spherex_tabular_bandpass import Tabular_Bandpass
from ssiaat.spherex_table import BandpassTool
from ssiaat.tabular_bandpass_lite import Tabular_Bandpass_Lite
from ssiaat.model.sed import hflatline

BANDS = (1, 2, 3, 4, 5, 6)
IX = 1024
N_IY = 2048
SHAPE_PARAM_NAMES = ("amplitude", "dmu_left", "a_left", "dmu_right", "a_right")
FLOOR_PARAM_NAMES = ("floor0", "floor1")
# Set for real in main() based on --floor/--no-floor; this default (floor on)
# lets the module-level functions below be written against one name.
PARAM_NAMES = SHAPE_PARAM_NAMES + FLOOR_PARAM_NAMES

# Degree-selection gate (see the plan doc / build report for rationale).
DEGREES = range(2, 9)
DEGREE_SLACK = 1.05        # E(deg) must be within 5% of the best degree tried
RMS_FLOOR = 6e-4           # ... and within max(1.5*D_band, RMS_FLOOR)
RMS_SLACK = 1.5
P95_TARGET = 1.5e-3
INTEGRAL_ERR_TARGET = 0.01
CENTROID_ERR_FACTOR = 0.1  # * dw

# Trim procedure constants.
TSUM_KEEP_FRAC = 0.95      # keep rows with tsum > this * median(tsum)
ERODE_MARGIN = 24          # rows eroded from each end of the kept run
MAD_TO_SIGMA = 1.4826
OUTLIER_MAD_MULT = 8


# --------------------------------------------------------------------------
# Per-row fit: hflatline (mu_center fixed), optionally + additive floor
# --------------------------------------------------------------------------

def fit_row(tool, band, iy, use_floor):
    """Fit one detector row directly (bypasses BandpassTool.fit_hflatline_at_iy
    when use_floor=True, since that method doesn't know about the floor term
    -- this is deliberately a standalone prototype in the derivation script,
    not yet promoted into production `fit_hflatline` code)."""
    w1, t1 = tool.bandpass_model(1024, int(iy), array=band)
    w1 = np.asarray(w1, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    wc, _ = tool.bandpass_model(1024, int(iy), array=band, central_bandpass_only=True)
    wc = float(np.asarray(wc))

    amplitude0 = float(t1.max())
    above_half = w1[t1 >= 0.5 * amplitude0]
    fwhm0 = ((above_half.max() - above_half.min())
            if len(above_half) >= 2 else (w1.max() - w1.min()) * 0.1)
    dmu0 = max(fwhm0 / 2, 1e-4)
    span = w1.max() - w1.min()

    if use_floor:
        def model_fn(mu, amplitude, dmu_left, a_left, dmu_right, a_right,
                     floor0, floor1):
            return (amplitude * hflatline(wc, dmu_left, a_left, dmu_right=dmu_right,
                                          a_right=a_right)(mu)
                    + floor0 + floor1 * (mu - wc))
        p0 = [amplitude0, dmu0, 1.0, dmu0, 1.0, 1e-4, 0.0]
        bounds = ([0, 1e-5, 1e-3, 1e-5, 1e-3, -1e-2, -1e-1],
                  [10 * amplitude0, span, 10.0, span, 10.0, 1e-2, 1e-1])
    else:
        def model_fn(mu, amplitude, dmu_left, a_left, dmu_right, a_right):
            return amplitude * hflatline(wc, dmu_left, a_left, dmu_right=dmu_right,
                                         a_right=a_right)(mu)
        p0 = [amplitude0, dmu0, 1.0, dmu0, 1.0]
        bounds = ([0, 1e-5, 1e-3, 1e-5, 1e-3],
                  [10 * amplitude0, span, 10.0, span, 10.0])

    popt, _ = curve_fit(model_fn, w1, t1, p0=p0, bounds=bounds, maxfev=20000)
    names = ("amplitude", "dmu_left", "a_left", "dmu_right", "a_right")
    if use_floor:
        names = names + FLOOR_PARAM_NAMES
    params = dict(zip(names, popt))
    params["mu_center_left"] = wc

    resid = t1 - model_fn(w1, *popt)
    diag = dict(w=w1, t=t1, resid=resid, rms=float(np.sqrt(np.mean(resid**2))))
    return params, diag


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def sweep_band(tool, band, use_floor):
    """Fit every row of the central column. Returns (iy_arr, out) where
    `out` maps each recorded quantity to a length-N_IY array (NaN where the
    fit failed -- never let one bad row abort the sweep)."""
    iy_arr = np.arange(N_IY)
    out = {k: np.full(N_IY, np.nan) for k in ("wc", "tsum", "peak", "rms", *PARAM_NAMES)}
    n_failed = 0
    for iy in iy_arr:
        try:
            params, diag = fit_row(tool, band, int(iy), use_floor)
        except Exception:
            n_failed += 1
            continue
        out["wc"][iy] = params["mu_center_left"]
        out["tsum"][iy] = float(diag["t"].sum())
        out["peak"][iy] = float(diag["t"].max())
        out["rms"][iy] = diag["rms"]
        for k in PARAM_NAMES:
            out[k][iy] = params[k]
    if n_failed:
        print(f"  ({n_failed}/{N_IY} rows failed to fit -- excluded)")
    return iy_arr, out


# --------------------------------------------------------------------------
# Trim
# --------------------------------------------------------------------------

def _largest_contiguous_run(mask):
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    if idx.size == 0:
        return out
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    out[best] = True
    return out


def _erode_run(mask, margin):
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    if idx.size == 0:
        return out
    lo, hi = idx.min() + margin, idx.max() - margin
    if lo <= hi:
        out[lo:hi + 1] = True
    return out


def trim_mask(out):
    """Step 3 of the plan: cut dichroic-contaminated / grid-truncated rows
    before any polynomial fit sees them."""
    tsum = out["tsum"]
    valid = np.isfinite(tsum)
    if not valid.any():
        return valid

    thresh = TSUM_KEEP_FRAC * np.nanmedian(tsum)
    keep = valid & (tsum > thresh)
    keep = _largest_contiguous_run(keep)
    keep = _erode_run(keep, ERODE_MARGIN)

    for k in PARAM_NAMES:
        v = out[k]
        med = np.nanmedian(v[keep])
        mad = np.nanmedian(np.abs(v[keep] - med)) or 1e-12
        keep &= np.isfinite(v) & (np.abs(v - med) <= OUTLIER_MAD_MULT * MAD_TO_SIGMA * mad)

    return keep


# --------------------------------------------------------------------------
# Polynomial fit with iterative sigma-clipping
# --------------------------------------------------------------------------

def cheb_fit_with_clipping(wc, y, deg, base_mask, passes=2, nsigma=3.0):
    domain = [float(wc[base_mask].min()), float(wc[base_mask].max())]
    mask = base_mask.copy()
    poly = Chebyshev.fit(wc[mask], y[mask], deg, domain=domain)
    for _ in range(passes):
        resid = y - poly(wc)
        sigma = float(np.std(resid[mask]))
        mask = base_mask & (np.abs(resid) <= nsigma * sigma)
        poly = Chebyshev.fit(wc[mask], y[mask], deg, domain=domain)
    return poly, mask


# --------------------------------------------------------------------------
# End-to-end curve reconstruction + validation
# --------------------------------------------------------------------------

def reconstruct_curve(polys, wc_row, w):
    pred = {k: float(polys[k](wc_row)) for k in PARAM_NAMES}
    shape = hflatline(wc_row, pred["dmu_left"], pred["a_left"],
                      dmu_right=pred["dmu_right"], a_right=pred["a_right"])
    floor = pred.get("floor0", 0.0) + pred.get("floor1", 0.0) * (w - wc_row)
    return pred["amplitude"] * shape(w) + floor


def validate_rows(tool, band, polys, wc_arr, iy_list):
    records = []
    for iy in iy_list:
        w1, t1 = tool.bandpass_model(1024, int(iy), array=band)
        w1 = np.asarray(w1, dtype=float)
        t1 = np.asarray(t1, dtype=float)
        wc_row = float(wc_arr[iy])

        model = reconstruct_curve(polys, wc_row, w1)
        rms = float(np.sqrt(np.mean((model - t1) ** 2)))

        sum_real, sum_model = float(t1.sum()), float(model.sum())
        integral_err = abs(sum_model - sum_real) / sum_real if sum_real else np.nan
        centroid_real = float(np.sum(w1 * t1) / sum_real) if sum_real else np.nan
        centroid_model = float(np.sum(w1 * model) / sum_model) if sum_model else np.nan
        peak_real = float(w1[np.argmax(t1)])
        peak_model = float(w1[np.argmax(model)])

        records.append(dict(iy=int(iy), rms=rms, integral_err=integral_err,
                            centroid_err=abs(centroid_model - centroid_real),
                            peak_err=abs(peak_model - peak_real)))
    return records


# --------------------------------------------------------------------------
# Per-band pipeline
# --------------------------------------------------------------------------

def process_band(band, plot_dir=None, use_floor=True):
    print(f"\n=== Band {band} (floor term {'ON' if use_floor else 'OFF'}) ===")
    t_start = time.time()

    bp = Tabular_Bandpass()
    lite = Tabular_Bandpass_Lite()
    tool = BandpassTool(bp, band=band)

    # Ground truth check: the coefficients we derive must be consistent
    # with the mu_center the production reconstruction will actually use
    # (Tabular_Bandpass_Lite's), which is a verbatim copy of the real
    # package's own coefficients. atol is loose, not exact-equality,
    # because a numba-accelerated chebval2d (if numba is installed) may
    # differ from the pure-numpy path at the ULP level.
    iy_check = np.arange(N_IY)
    ix_check = np.full(N_IY, IX)
    w_lite, _ = lite(ix_check, iy_check, array=band, central_bandpass_only=True)
    w_full, _ = bp(ix_check, iy_check, array=band, central_bandpass_only=True)
    np.testing.assert_allclose(w_lite, w_full, atol=1e-8, rtol=0,
                               err_msg=f"Lite vs. full central-wavelength mismatch, band {band}")

    iy_arr, out = sweep_band(tool, band, use_floor)
    keep = trim_mask(out)
    if keep.sum() < 32:
        print(f"  !!! FAIL: only {keep.sum()} rows survived trimming -- band {band} not gated")
        return dict(band=band, supported=False, reason="insufficient rows after trim")

    wc = out["wc"]
    kept_iy = iy_arr[keep]
    print(f"  kept {keep.sum()}/{N_IY} rows; iy range [{kept_iy.min()}, {kept_iy.max()}];"
          f" wc range [{wc[keep].min():.4f}, {wc[keep].max():.4f}] um")

    D_band = float(np.nanmedian(out["rms"][keep]))
    target = max(RMS_SLACK * D_band, RMS_FLOOR)

    train_mask = keep & (iy_arr % 4 != 0)
    val_mask = keep & (iy_arr % 4 == 0)
    val_iy = iy_arr[val_mask]

    # -- degree selection: end-to-end curve RMS on held-out rows, not
    # parameter-space residuals (which plateau uninformatively -- see the
    # investigation writeup in the build report).
    E = {}
    degree_polys = {}
    for deg in DEGREES:
        domain = [float(wc[train_mask].min()), float(wc[train_mask].max())]
        polys = {k: Chebyshev.fit(wc[train_mask], out[k][train_mask], deg, domain=domain)
                 for k in PARAM_NAMES}
        degree_polys[deg] = polys
        records = validate_rows(tool, band, polys, wc, val_iy)
        E[deg] = float(np.median([r["rms"] for r in records]))

    min_E = min(E.values())
    chosen_deg = next((d for d in DEGREES if E[d] <= DEGREE_SLACK * min_E and E[d] <= target),
                      None)
    print("  E(deg) [held-out median curve RMS]:",
          {d: f"{v:.3e}" for d, v in E.items()})
    print(f"  D_band (median direct-fit RMS) = {D_band:.3e}; target <= {target:.3e}")

    supported = chosen_deg is not None
    if not supported:
        chosen_deg = min(E, key=E.get)
        print(f"  !!! FAIL: no degree in {list(DEGREES)} meets the target for band {band}"
              f" (best E={min_E:.3e} at deg={chosen_deg}) -- NOT gated for Part C")
    else:
        print(f"  degree selected: {chosen_deg} (E={E[chosen_deg]:.3e})")

    # -- final validation report at the chosen degree (genuinely held-out:
    # these polys were trained on train_mask only, evaluated on val_mask).
    final_val_records = validate_rows(tool, band, degree_polys[chosen_deg], wc, val_iy)
    rms_vals = np.array([r["rms"] for r in final_val_records])
    direct_rms_vals = out["rms"][val_iy]
    report = dict(
        median_rms=float(np.median(rms_vals)),
        p95_rms=float(np.percentile(rms_vals, 95)),
        max_rms=float(np.max(rms_vals)),
        ratio_to_direct=float(np.median(rms_vals / direct_rms_vals)),
        median_integral_err=float(np.median([r["integral_err"] for r in final_val_records])),
        median_centroid_err=float(np.median([r["centroid_err"] for r in final_val_records])),
        median_peak_err=float(np.median([r["peak_err"] for r in final_val_records])),
    )
    native_wvl = np.asarray(bp.bandpass_wavelengths[band])
    dw = float(native_wvl[1] - native_wvl[0])
    print(f"  validation (held-out): median RMS={report['median_rms']:.3e}"
          f" (target<={target:.3e})  p95={report['p95_rms']:.3e} (target<={P95_TARGET:.1e})"
          f"  ratio-to-direct={report['ratio_to_direct']:.2f}x")
    print(f"  integral err (median)={report['median_integral_err']:.4f} (target<={INTEGRAL_ERR_TARGET})"
          f"  centroid err (median)={report['median_centroid_err']:.2e} um"
          f" (target<={CENTROID_ERR_FACTOR * dw:.2e})")

    gate_pass = (report["median_rms"] <= target and report["p95_rms"] <= P95_TARGET
                and report["median_integral_err"] <= INTEGRAL_ERR_TARGET
                and report["median_centroid_err"] <= CENTROID_ERR_FACTOR * dw)
    if supported and not gate_pass:
        print(f"  !!! Degree passed the degree-selection gate but the full validation report"
              f" gate failed for band {band} -- NOT gated for Part C")
    supported = supported and gate_pass

    # -- refit on ALL kept rows (not just train) for the production
    # coefficients, with iterative sigma-clipping.
    final_polys, final_masks = {}, {}
    for k in PARAM_NAMES:
        poly, mask = cheb_fit_with_clipping(wc, out[k], chosen_deg, keep)
        final_polys[k] = poly
        final_masks[k] = mask

    # -- correlation / ratio diagnostics (the parameter-reduction question).
    resid = {k: out[k][keep] - final_polys[k](wc[keep])
            for k in PARAM_NAMES if k != "amplitude"}
    corr_dmu_a_left = float(np.corrcoef(resid["dmu_left"], resid["a_left"])[0, 1])
    corr_dmu_a_right = float(np.corrcoef(resid["dmu_right"], resid["a_right"])[0, 1])
    corr_a_lr = float(np.corrcoef(resid["a_left"], resid["a_right"])[0, 1])
    corr_dmu_lr = float(np.corrcoef(resid["dmu_left"], resid["dmu_right"])[0, 1])
    ratio_a = out["a_left"][keep] / out["a_right"][keep]
    ratio_dmu = out["dmu_left"][keep] / out["dmu_right"][keep]
    print(f"  [reduction check] corr(dmu_L,a_L)={corr_dmu_a_left:+.3f}"
          f"  corr(dmu_R,a_R)={corr_dmu_a_right:+.3f}"
          f"  corr(a_L,a_R)={corr_a_lr:+.3f}  corr(dmu_L,dmu_R)={corr_dmu_lr:+.3f}")
    print(f"  [reduction check] a_L/a_R mean={ratio_a.mean():.3f} std={ratio_a.std():.3f}"
          f"   dmu_L/dmu_R mean={ratio_dmu.mean():.3f} std={ratio_dmu.std():.3f}")

    if use_floor:
        f0, f1 = out["floor0"][keep], out["floor1"][keep]
        amp = out["amplitude"][keep]
        print(f"  [floor] floor0/amplitude mean={np.mean(f0/amp):.4f}"
              f"  floor1/amplitude mean={np.mean(f1/amp):.4f}"
              f"  (both as a fraction of the peak, for scale)")

    print(f"  band {band} done in {time.time() - t_start:.1f}s")

    result = dict(
        band=band, supported=supported, degree=chosen_deg,
        D_band=D_band, target=target, E=E, report=report,
        polys=final_polys, domain=[float(wc[keep].min()), float(wc[keep].max())],
        native_grid=(float(native_wvl[0]), dw, int(native_wvl.size)),
        corr=dict(dmu_a_left=corr_dmu_a_left, dmu_a_right=corr_dmu_a_right,
                 a_lr=corr_a_lr, dmu_lr=corr_dmu_lr),
        ratios=dict(a_lr_mean=float(ratio_a.mean()), a_lr_std=float(ratio_a.std()),
                   dmu_lr_mean=float(ratio_dmu.mean()), dmu_lr_std=float(ratio_dmu.std())),
        iy_arr=iy_arr, out=out, keep=keep,
    )

    if plot_dir is not None:
        plot_band(band, result, plot_dir)

    return result


# --------------------------------------------------------------------------
# Diagnostics plots (--plot)
# --------------------------------------------------------------------------

def plot_band(band, result, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    iy_arr, out, keep = result["iy_arr"], result["out"], result["keep"]
    wc = out["wc"]
    order = np.argsort(wc[keep])
    wc_sorted = wc[keep][order]

    # 1. Parameters vs wc, with polynomial overlay and residuals.
    fig, axs = plt.subplots(len(PARAM_NAMES), 2, figsize=(11, 3 * len(PARAM_NAMES)),
                            gridspec_kw=dict(width_ratios=[3, 1.2]))
    for row, k in enumerate(PARAM_NAMES):
        ax, axr = axs[row, 0], axs[row, 1]
        y = out[k][keep][order]
        ax.plot(wc_sorted, y, ".", ms=3, color="0.5", label="fitted (per row)")
        wfine = np.linspace(wc_sorted.min(), wc_sorted.max(), 400)
        ax.plot(wfine, result["polys"][k](wfine), "-", color="C1", label=f"deg-{result['degree']} fit")
        ax.set_ylabel(k)
        ax.legend(fontsize=7)
        resid = y - result["polys"][k](wc_sorted)
        axr.plot(wc_sorted, resid, ".", ms=2, color="C1")
        axr.set_ylabel("residual")
        axr.axhline(0, color="0.7", lw=0.8)
    axs[-1, 0].set_xlabel("central wavelength [um]")
    axs[-1, 1].set_xlabel("central wavelength [um]")
    fig.suptitle(f"Band {band}: hflatline shape parameters vs. wavelength")
    fig.tight_layout()
    fig.savefig(plot_dir / f"band{band}_params.png", dpi=120)
    plt.close(fig)

    # 2. tsum / direct-fit RMS vs iy, with trim mask.
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(iy_arr, out["tsum"], ".", ms=2, color="0.6", label="tsum (all rows)")
    axs[0].plot(iy_arr[keep], out["tsum"][keep], ".", ms=2, color="C0", label="kept")
    axs[0].set_ylabel("sum(t)")
    axs[0].legend(fontsize=8)
    axs[1].plot(iy_arr, out["rms"], ".", ms=2, color="0.6")
    axs[1].plot(iy_arr[keep], out["rms"][keep], ".", ms=2, color="C0")
    axs[1].set_ylabel("direct-fit RMS")
    axs[1].set_xlabel("iy")
    fig.suptitle(f"Band {band}: trim diagnostics")
    fig.tight_layout()
    fig.savefig(plot_dir / f"band{band}_trim.png", dpi=120)
    plt.close(fig)

    # 3. Curve overlays at a few held-out rows (near both trimmed edges + mid).
    kept_iy = iy_arr[keep]
    sample_iy = sorted({int(kept_iy[0] + 5), int(kept_iy[len(kept_iy) // 2]),
                        int(kept_iy[-6])})
    fig, axs = plt.subplots(1, len(sample_iy), figsize=(4.2 * len(sample_iy), 3.6))
    if len(sample_iy) == 1:
        axs = [axs]
    tool = BandpassTool(Tabular_Bandpass(), band=band)
    for ax, iy in zip(axs, sample_iy):
        w1, t1 = tool.bandpass_model(1024, iy, array=band)
        w1 = np.asarray(w1, float)
        t1 = np.asarray(t1, float)
        model = reconstruct_curve(result["polys"], float(wc[iy]), w1)
        ax.plot(w1, t1, ".", ms=2, color="0.5", label="real")
        ax.plot(w1, model, "-", color="C1", label="poly reconstruction")
        ax.set_title(f"iy={iy}")
        ax.legend(fontsize=7)
    fig.suptitle(f"Band {band}: curve reconstruction at sample rows")
    fig.tight_layout()
    fig.savefig(plot_dir / f"band{band}_curves.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Coefficient dict emission
# --------------------------------------------------------------------------

def emit_coeff_dict(results):
    supported_bands = [b for b, r in results.items() if r.get("supported")]
    print("\n" + "=" * 70)
    print(f"# Supported bands: {supported_bands}"
         f"  (unsupported: {[b for b in results if b not in supported_bands]})")
    print("# ---- paste-ready coefficients for ssiaat/bandpass_shape_lite.py ----")
    print()
    print("SHAPE_CHEB_COEFF = {")
    for band in supported_bands:
        r = results[band]
        print(f"    {band}: {{")
        for k in PARAM_NAMES:
            coeffs = ", ".join(repr(float(c)) for c in r["polys"][k].coef)
            print(f"        {k!r}: np.array([{coeffs}]),")
        print("    },")
    print("}")
    print()
    print("SHAPE_CHEB_DOMAIN = {")
    for band in supported_bands:
        lo, hi = results[band]["domain"]
        print(f"    {band}: ({lo!r}, {hi!r}),")
    print("}")
    print()
    print("NATIVE_GRID = {  # (w_start, dw, n)")
    for band in supported_bands:
        print(f"    {band}: {results[band]['native_grid']!r},")
    print("}")
    print()
    print("FIT_DEGREE = {")
    for band in supported_bands:
        print(f"    {band}: {results[band]['degree']},")
    print("}")
    print()
    print("TYPICAL_RMS = {")
    for band in supported_bands:
        print(f"    {band}: {results[band]['D_band']!r},")
    print("}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", type=int, nargs="+", default=list(BANDS))
    ap.add_argument("--plot", action="store_true", help="write diagnostic PNGs")
    ap.add_argument("--plot-dir", default="bandpass_shape_diagnostics")
    ap.add_argument("--out", default=None, help="dump the raw per-band sweep to this .npz")
    ap.add_argument("--no-floor", dest="floor", action="store_false",
                    help="disable the additive floor term (reproduces the"
                         " original hflatline-only numbers, for comparison)")
    args = ap.parse_args()

    global PARAM_NAMES
    PARAM_NAMES = SHAPE_PARAM_NAMES + FLOOR_PARAM_NAMES if args.floor else SHAPE_PARAM_NAMES

    t0 = time.time()
    results = {}
    for band in args.bands:
        results[band] = process_band(band, plot_dir=args.plot_dir if args.plot else None,
                                     use_floor=args.floor)
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")

    emit_coeff_dict(results)

    if args.out:
        npz_payload = {}
        for band, r in results.items():
            for key in ("iy_arr", "keep"):
                npz_payload[f"b{band}_{key}"] = r[key]
            for key, arr in r["out"].items():
                npz_payload[f"b{band}_out_{key}"] = arr
        np.savez(args.out, **npz_payload)
        print(f"\nraw sweep dumped to {args.out}")

    n_supported = sum(1 for r in results.values() if r.get("supported"))
    if n_supported < len(results):
        print(f"\n{len(results) - n_supported}/{len(results)} band(s) did NOT clear the"
             " validation gate -- see FAIL lines above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
