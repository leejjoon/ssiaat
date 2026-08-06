#!/usr/bin/env python3
"""Identify and fit emission lines in the stacked SPHEREx PN spectrum.

Uses `ssiaat.model.multiband_lines.MultiBandLineModel`: each candidate
line's central wavelength is fit as ONE value shared across every band
it appears in, while its amplitude is free per band -- appropriate here
because `pn_master.ecsv`'s six bands may not share a wavelength solution
(SPHEREx's per-band calibration is known to be imperfect), so a per-band
independent line fit would give inconsistent, non-comparable centers.
The line profile itself is `ssiaat.bandpass_shape_lite.BandpassShapeLite`
evaluated at the trial center -- the physically correct shape for an
intrinsically unresolved line, not a generic Gaussian.

Data: `pn_master.ecsv`, a stacked spectrum from 308 SPHEREx-observed
planetary nebulae (method="A_pool", the pooled-GP combine -- higher line
contrast than "B_combine", per the existing convention in
`/home/jjlee/spherex/work_hips/pn_template.py`, a sibling, PN-specific
analysis workspace this script's conventions are chosen to match without
importing its code). `rel` is flux already divided by continuum
(baseline ~1); `rel_err` is its 1-sigma error.

Known caveats (read before trusting band 5/6 numbers):
- The stack's noise is strongly correlated (expected for a pooled-GP
  combine); this fit calibrates errors empirically per band via a
  residual-scan null (`MultiBandLineModel`'s `null_mad`) rather than
  trusting the formal `1/error**2`-weighted covariance.
- Bands 5 and 6's REAL instrumental profile is known (independent of
  this fit) to be more flat-topped than `BandpassShapeLite` predicts,
  beyond the wavelength shift alone. With only one candidate line each
  in band 5 (Br-alpha) and band 6 (Pf-beta), a free per-band width
  correction isn't identifiable (it would trade off exactly against
  that single line's amplitude), so this isn't fit for -- instead, this
  script reports the inner/outer residual split specifically for those
  two lines (see `resid_rms_inner`/`resid_rms_outer` in the results
  table and the dedicated zoom plots) as the diagnostic that would
  reveal this mismatch, rather than silently trusting their amplitudes.

Line list, extended from the original core H/He set: the first pass's
residuals showed two strong features not covered by the core list,
confirmed against the real data before adding them here (see this
session's analysis) --
- A complex at ~0.90-0.97um in band 1 (up to +17 sigma residual):
  matched to the [S III] 0.9069/0.9531um forbidden-line doublet, common
  and often bright in PNe. [S III] 0.9531 alone explains the dominant
  0.94-0.96um cluster (its expected FWHM at that wavelength, ~0.027um,
  spans almost exactly that range); [S III] 0.9069, shifted blueward by
  band 1's own ~0.003um offset, lines up with a distinct secondary bump
  measured at 0.9042um (S/N~5.8 in the pre-extension fit).
- The well-known PAH 3.29um complex in band 4 (up to +15 sigma). Unlike
  every other entry here, PAH is NOT modeled as a `LineSpec` -- its
  intrinsic width is comparable to or wider than band 4's instrumental
  resolution, so it is not "an unresolved line broadened by the
  instrument's bandpass" the way `LineSpec`'s profile assumes. It's
  attached instead as `BandSpectrum.extra_models`: `sed.py`'s existing
  `get_pah()` shapes (narrow aromatic + broad aliphatic, already tuned
  against real spectra), fixed in shape/position, with only their
  amplitude free. Reported separately in
  `MultiBandLineFitResults.extra_model_amplitudes`, not mixed into the
  tied-line `amplitudes` table -- it has no tied center to report
  against.

Known remaining unmodeled residual structure (out of scope for this
pass, not silently claimed as covered): band 4 bumps near 2.85-2.95um
and ~3.75um; band 3 near 2.15-2.18um; band 6 near 4.47um. Left as
candidates for a future extension.

Usage
-----
    .venv/bin/python examples/fit_pn_lines.py
    .venv/bin/python examples/fit_pn_lines.py --ecsv /path/to/pn_master.ecsv --plot-dir out/
"""
import argparse
import sys

import numpy as np
from astropy.table import Table

from ssiaat.model.multiband_lines import BandSpectrum, LineSpec, MultiBandLineModel
from ssiaat.model.sed import get_pah

PN_METHOD = "A_pool"  # pooled-GP stack; higher line contrast than "B_combine"

# Core H/He recombination line list, plus the [S III] doublet (see the
# module docstring for why they were added). Rest wavelengths, microns.
# [S III] prior_offset is band 1's own measured ~-0.003um shift (from the
# pre-extension fit's centers_mode="wavelength_solution" run) -- just an
# initial guess; the tied fit refines it like every other line here.
LINE_LIST = [
    LineSpec("He I 1.083", 1.08303),
    LineSpec("Pa-gamma", 1.09411),
    LineSpec("Pa-beta", 1.28181),
    LineSpec("Br-delta", 1.94509),
    LineSpec("Pa-alpha", 1.87510),
    LineSpec("Br-gamma", 2.16612),
    LineSpec("He I 2.059", 2.05869),
    LineSpec("Br-beta", 2.62587),
    LineSpec("Br-alpha", 4.05116),
    LineSpec("Pf-beta", 4.65378),
    LineSpec("[S III] 0.9069", 0.90686, prior_offset=-0.0025),
    LineSpec("[S III] 0.9531", 0.95306, prior_offset=-0.0025),
]

# Bands whose real profile is known to be flatter-topped than
# BandpassShapeLite predicts (see the module docstring) -- flagged in
# the report, not modeled.
FLAT_TOP_CAVEAT_BANDS = (5, 6)


def load_pn_spectra(path, method=PN_METHOD, baseline_degree=3):
    """Load `pn_master.ecsv` into one BandSpectrum per band, method
    `method` only, NaN rows dropped."""
    t = Table.read(path, format="ascii.ecsv")
    t = t[t["method"] == method]

    spectra = []
    for band in sorted(set(t["band"])):
        sub = t[t["band"] == band]
        good = np.isfinite(sub["rel"]) & np.isfinite(sub["rel_err"])
        order = np.argsort(sub["wavelength"][good])
        wavelength = np.asarray(sub["wavelength"][good], dtype=float)[order]
        value = np.asarray(sub["rel"][good], dtype=float)[order]
        error = np.asarray(sub["rel_err"][good], dtype=float)[order]
        spectra.append(BandSpectrum(int(band), wavelength, value, error,
                                    baseline_degree=baseline_degree))
    return spectra


def _pah_extra_models():
    """The PAH 3.29um complex as FIXED extra shape components (see the
    module docstring for why this isn't a LineSpec): sed.py's `get_pah`
    already has these tuned (narrow aromatic + broad aliphatic
    component) against real spectra."""
    narrow, broad = get_pah()
    return (("PAH_narrow_3.29", narrow), ("PAH_broad_3.42", broad))


def attach_extra_models(spectra, extra_models_by_band):
    """Return new BandSpectrum objects (the dataclass is frozen) with
    `extra_models` attached for the given bands; bands not in
    `extra_models_by_band` pass through unchanged."""
    out = []
    for spec in spectra:
        extra = extra_models_by_band.get(spec.band)
        if extra:
            spec = BandSpectrum(spec.band, spec.wavelength, spec.value, spec.error,
                                baseline_degree=spec.baseline_degree, extra_models=extra)
        out.append(spec)
    return out


def print_line_table(result, label):
    table = result.line_table()
    print(f"\n=== {label}: per-line summary ===")
    with_cols = table[["line", "rest_wavelength", "center", "n_bands",
                       "snr_comb", "classification", "chi2_consist", "upper_limit"]]
    print(with_cols.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\n=== {label}: per-(line,band) detail ===")
    amp = result.amplitudes[["line", "band", "center", "amplitude", "err_formal",
                             "err_cal", "snr_cal", "coverage", "shape_extrapolated",
                             "at_bound", "blended", "resid_rms_inner", "resid_rms_outer"]]
    print(amp.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if len(result.extra_model_amplitudes):
        print(f"\n=== {label}: fixed extra-shape components (e.g. PAH; no tied center) ===")
        print(result.extra_model_amplitudes.to_string(index=False,
                                                       float_format=lambda x: f"{x:.5f}"))

    print(f"\n=== {label}: per-band diagnostics ===")
    diag = result.null_mad.to_frame().join(result.chi2_red)
    print(diag.to_string(float_format=lambda x: f"{x:.3f}"))

    if result.deltas is not None:
        print(f"\n=== {label}: fitted per-band wavelength offsets (delta_b) ===")
        print(result.deltas.to_string(float_format=lambda x: f"{x:.5f}"))

    blended = result.amplitudes[result.amplitudes.blended]
    for band, group in blended.groupby("band"):
        names = ", ".join(group.line)
        print(f"\n  NOTE band {band}: {names} are fitted within half a resolution"
             " element of each other -- their individual amplitudes are"
             " near-degenerate (design-matrix collinearity, not a fitting mistake);"
             " trust only their COMBINED flux, not the per-line breakdown.")

    for band in FLAT_TOP_CAVEAT_BANDS:
        rows = result.amplitudes[result.amplitudes.band == band]
        if len(rows):
            print(f"\n  NOTE band {band}: known flat-top-vs-model mismatch (see module"
                 " docstring) -- check resid_rms_inner vs resid_rms_outer above for a"
                 " signed pattern before trusting this band's amplitude/S/N at face value.")


def make_plots(result_free, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    result = result_free
    model = result.model

    # Distinct, saturated colors per band -- data markers stay muted/gray
    # regardless of band, so only the MODEL lines carry color and are
    # never competing with same-toned data ink for attention.
    band_color = {b: c for b, c in zip(sorted(model.spectra),
                                       ["#d62728", "#1f77b4", "#2ca02c",
                                        "#9467bd", "#ff7f0e", "#17becf"])}

    # 1. Per-band overview: data + model + residual. Model drawn bold,
    # on top (high zorder), in a color that reads clearly against the
    # gray data -- the earlier orange-on-gray-at-lw-1.3 combination was
    # too low-contrast to see where the curve tracks through dense data.
    bands = sorted(model.spectra)
    fig, axs = plt.subplots(len(bands), 2, figsize=(12, 3 * len(bands)),
                            gridspec_kw=dict(width_ratios=[3, 1]))
    for row, band in enumerate(bands):
        spec = model.spectra[band]
        curve = result.model_curve(band)
        ax, axr = axs[row, 0], axs[row, 1]
        ax.errorbar(spec.wavelength, spec.value, yerr=spec.error, fmt=".",
                   ms=3, color="0.6", ecolor="0.82", zorder=1, label="data")
        ax.plot(spec.wavelength, curve, "-", color=band_color[band], lw=2.2,
               zorder=5, label="model")
        ax.set_title(f"band {band}"
                     + (" (flat-top caveat)" if band in FLAT_TOP_CAVEAT_BANDS else ""))
        ax.set_ylabel("rel")
        ax.legend(fontsize=7)
        resid = spec.value - curve
        axr.plot(spec.wavelength, resid / spec.error, ".", ms=3, color=band_color[band])
        axr.axhline(0, color="0.7", lw=0.8)
        axr.set_ylabel("resid / err")
    axs[-1, 0].set_xlabel("wavelength [um]")
    axs[-1, 1].set_xlabel("wavelength [um]")
    fig.suptitle("PN master spectrum: per-band fit (centers_mode='free')")
    fig.tight_layout()
    fig.savefig(plot_dir / "pn_lines_overview.png", dpi=130)
    plt.close(fig)

    # 2. Per-line zoom, all contributing bands overlaid. Three distinct
    # visual roles, ordered back-to-front: data (muted gray, zorder 1)
    # -> full-band model (dashed black, zorder 3 -- context: shows any
    # nearby line's leakage) -> this line's own isolated contribution
    # (bold, saturated, zorder 5 -- what "detected"/"blended" refers to).
    line_table = result.line_table().set_index("line")
    n_lines = len(model.lines)
    ncols = 3
    nrows = -(-n_lines // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
    for ax, (name, line) in zip(axs.ravel(), model.lines.items()):
        sub = result.amplitudes[result.amplitudes.line == name]
        for _, row in sub.iterrows():
            band = int(row.band)
            spec = model.spectra[band]
            fwhm = row.fwhm
            window = (spec.wavelength > row.center - 4 * fwhm) & \
                     (spec.wavelength < row.center + 4 * fwhm)
            ax.errorbar(spec.wavelength[window], spec.value[window],
                       yerr=spec.error[window], fmt=".", ms=4, color="0.65",
                       ecolor="0.82", alpha=0.9, zorder=1,
                       label=f"band {band} data")
            wfine = np.linspace(spec.wavelength[window].min(),
                               spec.wavelength[window].max(), 300)
            ax.plot(wfine, result.model_curve(band, wfine), "--", lw=1.8,
                   color="0.15", zorder=3, label=f"band {band} full model")
            ax.plot(wfine, result.model_curve(band, wfine, only_lines=[name]),
                   "-", lw=2.6, color=band_color[band], zorder=5,
                   label=f"band {band}: {name} only")
        ax.axvline(row.center if len(sub) else line.rest_wavelength,
                  ls=":", color="0.4", lw=1.0, zorder=2)
        classification = (line_table.loc[name, "classification"]
                          if name in line_table.index else "?")
        title = f"{name} ({classification})"
        if len(sub) and sub["blended"].any():
            title += " -- BLENDED"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)
    for ax in axs.ravel()[n_lines:]:
        ax.axis("off")
    fig.suptitle("Per-line zoom, all contributing bands (centers_mode='free')")
    fig.tight_layout()
    fig.savefig(plot_dir / "pn_lines_zoom.png", dpi=130)
    plt.close(fig)

    # 3. delta_b vs band (wavelength_solution mode -- see caller).
    print(f"wrote plots to {plot_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecsv", default="pn_master.ecsv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-dir", default="pn_lines_diagnostics")
    ap.add_argument("--out", default=None, help="write the merged results table here (.ecsv)")
    args = ap.parse_args()

    spectra = load_pn_spectra(args.ecsv)
    spectra = attach_extra_models(spectra, {4: _pah_extra_models()})
    print(f"loaded {sum(len(s.wavelength) for s in spectra)} points across"
         f" {len(spectra)} bands from {args.ecsv} (method={PN_METHOD!r})")

    model_free = MultiBandLineModel(list(LINE_LIST), spectra, centers_mode="free")
    print("\n--- band assignment ---")
    assignment = model_free.assignment()
    print(assignment[assignment.assigned][["line", "band", "coverage", "npts"]]
         .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    excluded = assignment[~assignment.assigned]
    if len(excluded):
        print("\n--- excluded (line, band) pairs ---")
        print(excluded[["line", "band", "coverage", "npts", "in_domain"]]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    result_free = model_free.fit()
    print_line_table(result_free, "centers_mode='free'")

    try:
        model_ws = MultiBandLineModel(list(LINE_LIST), spectra,
                                      centers_mode="wavelength_solution")
        result_ws = model_ws.fit()
        print_line_table(result_ws, "centers_mode='wavelength_solution'")
    except ValueError as e:
        result_ws = None
        print(f"\ncenters_mode='wavelength_solution' skipped: {e}")

    if args.plot:
        make_plots(result_free, args.plot_dir)

    if args.out:
        frame = result_free.to_frame()
        Table.from_pandas(frame).write(args.out, format="ascii.ecsv", overwrite=True)
        print(f"\nwrote results table to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
