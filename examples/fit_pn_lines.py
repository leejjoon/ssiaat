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

Literature verification and further extension (see
`examples/pn_line_list_literature.md` for the full sourced note, one
citation per claim -- not written from memory): a background research
pass against NIST ASD, the Rydberg formula (H I), and primary PN
near/mid-IR spectroscopy papers (Hora et al. 1999; Smith & McLean 2007;
Akras et al. 2024; Jones et al. 2023; and others cited in that file)
checked every rest wavelength above and searched for commonly-detected
PNe features not yet in this list. Results applied here:
- **Corrected** Br-alpha 4.05116 -> 4.05228um: the repo's old value was
  off from the literature/Rydberg-confirmed vacuum wavelength by
  0.00112um (~83 km/s equivalent) -- a real, non-trivial offset for
  line-center fitting, unlike every other H I line above 2um in the old
  list (Br-gamma/beta/Pf-beta), which all matched vacuum to <=2e-5um.
- **Resolved an internal air/vacuum inconsistency**: Pa-beta and Pa-alpha
  were air wavelengths (1.28181, 1.87510) while every other H I line in
  the list was vacuum. Corrected to vacuum (1.28217, 1.87563) for
  consistency with the rest -- both near-IR calibration atlases and the
  most recent JWST PN paper found in the literature search tabulate H I
  lab wavelengths as vacuum uniformly, not NIST's own <2um air rule.
- **Added**, in descending order of how well the literature confirms
  them as common in real PNe: the [C I] 0.8727/0.9824/0.9850um triplet
  (Akras et al. 2024, ~40 PNe including the same IC 5117 as the Rudy
  near-IR atlas); H2 1-0 S(1) 2.1218um (the single most commonly
  detected H2 line in PN near-IR spectra, and close to this list's own
  previously-noted unmodeled "band 3 near 2.15-2.18um" residual -- see
  below); Pf-gamma 3.74057um and Pf-epsilon 3.03921um (both real,
  Rydberg- and NOT/IAC-table-confirmed H I lines, added because
  Pf-gamma sits almost exactly on this list's previously-noted
  unmodeled "band 4 ~3.75um" residual); He I 1.7002um and He I
  2.1126-2.1138um (single-source confirmation each, lower confidence,
  added anyway since they're cited PNe lines and this model's
  `blended`/neighbor-distance-cap machinery already handles a close
  near-blend safely -- He I ~2.113 sits only ~0.009um from H2 2.1218,
  see the per-line note in the results); [Fe II] 1.257/1.644um (real
  but not universal -- shock/low-ionization-structure tracers present
  in only a subset of PNe, so a non-detection in this 308-PN stack is
  an expected, not alarming, outcome).
- **Not added**: coronal lines ([Si VI/VII/IX], confirmed real but
  rare, requiring an unusually hot central star -- not expected as a
  generic stack feature); Pfund-delta 3.29700um, despite being a real,
  literature-confirmed line sitting inside the PAH complex itself
  (Smith & McLean 2007 explicitly subtract it before fitting PAH in
  individual PNe) -- adding it here as a `LineSpec` would put its
  profile column nearly collinear with the fixed-shape PAH extra model
  in the exact same band-4 window, an identifiability risk with no
  clean fix in this pass; flagged here for whoever revisits band 4's
  PAH modeling next, not fit for now. [N I] (literature note found no
  source it could independently verify) and any ice/CO/CO2 band (no
  PN detection found anywhere inside 0.75-5.0um) were also not added.
- **Not changed**: `sed.py`'s `get_pah()` aromatic-component center.
  The literature's PN-specific value (Smith & McLean 2007: Class A3.3,
  center 3.288-3.290um) sits ~0.003-0.005um blueward of this repo's
  tuned center (~3.293um, closer to literature's H-II-region-typical
  Class B3.3) -- a real, if modest, discrepancy. Left as-is because the
  repo's value was tuned against this exact stacked spectrum, which is
  a more direct check for this dataset than a generic literature
  classification; noted here as an open item rather than silently
  ignored.

Confirmed against the actual fit: the previously-noted band 3
~2.15-2.18um residual is now resolved (worst |resid/err| in that window
dropped from a clear multi-sigma bump to <=0.65 -- explained by He I
~2.113/H2 2.1218), and the band 4 ~3.75um residual is likewise resolved
by Pf-gamma (peak residual there dropped from a clear bump to <=1.5
sigma ringing, consistent with a well-fit line, not a missing feature).

Known remaining unmodeled residual structure (out of scope for this
pass, not silently claimed as covered): band 4 bumps near 2.78-2.85um
and at the PAH complex's own blue edge (~3.256-3.262um -- plausibly
Pfund-delta 3.29700um, the literature-confirmed line flagged above as
sitting inside the PAH window and not modeled here).

Band 6 correction (important -- read before trusting any earlier
`null_mad`/significance number for this band): band 6 had only ONE
fitted line (Pf-beta), so `MultiBandLineModel`'s null-noise calibration
-- which masks the vicinity of every FITTED line before estimating the
band's true noise level -- had nothing else to mask, and ended up
measuring the amplitude of three additional real, unmodeled residual
bumps (at ~4.47, ~4.61, ~4.76um) as if they were noise. That inflated
band 6's null_mad to 4.65 (worst of any band), which in turn made those
same bumps (and Pf-beta itself) look far less significant than they
really are -- a self-contamination effect verified directly: manually
masking those three positions before recomputing null_mad drops it to
~1.8, at which point two of the three bumps become clear multi-sigma
detections instead of "noise". See
`MultiBandLineModel._compute_null_mad`'s docstring for the general
caveat (an automatic robust-statistics fix was tried and doesn't work
at this contamination fraction; the real fix is exactly what's done
below -- identify and fit the missing lines so the existing masking
logic can do its job).
- ~4.7635um: matches He II 8->7 (computed via the same Z=2 hydrogenic
  Rydberg-Ritz formula used for the H I list in
  `examples/pn_line_list_literature.md`) to 0.010um -- well within 1
  FWHM. He II is physically expected in a stack containing PNe with
  very hot central stars (see the coronal-line discussion in that
  file); a low-n member like 8->7 is far brighter than the only
  previously-found He II reference in the literature (a JWST paper's
  28->13 transition, essentially a continuum-level cascade line, not
  comparably bright). Added as a `LineSpec` -- comes out a clean
  ~6.3 sigma detection.
- ~4.4728um (the strongest of the three): no H I or He II hydrogenic
  transition lands within several FWHM (checked the same way). Added
  as a tentative, UNIDENTIFIED `LineSpec` (`Unid. 4.47um band6`) --
  comes out a robust ~15 sigma detection once band 6's null_mad is no
  longer self-contaminated, but its actual physical ID is an open
  question for a follow-up literature pass, not claimed here.
- ~4.6063um: tried as a fourth `LineSpec` and DROPPED. Its free center
  wandered to the very edge of its neighbor-distance-capped bounds
  (see `MultiBandLineModel.fit`'s neighbor-cap logic) right up against
  Pf-beta's own wing, and fitting it there cut Pf-beta's own amplitude
  roughly in half (2.21 -> 0.44 rel units) -- i.e. it isn't an
  independent feature, it's this band's own already-documented
  flat-top-vs-model mismatch (see `FLAT_TOP_CAVEAT_BANDS` above)
  showing up as excess wing residual next to Pf-beta, not a second
  line. Forcing a `LineSpec` there does real harm (destabilizing a
  previously solid detection) for no real gain, so it's left out;
  Pf-beta's own `resid_rms_inner`/`resid_rms_outer` numbers are the
  right place to track this, not a separate fitted line.

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

# Rest wavelengths, microns, all vacuum (H I lines: Rydberg formula,
# cross-checked against NOT/IAC + UKIRT calibration atlases and JWST
# Jones et al. 2023; see examples/pn_line_list_literature.md for the
# full sourced note -- every number below traces back to a citation
# there, not memory). [S III] prior_offset is band 1's own measured
# ~-0.003um shift (from the pre-extension fit's
# centers_mode="wavelength_solution" run) -- just an initial guess; the
# tied fit refines it like every other line here.
LINE_LIST = [
    # Core H I recombination series + He I, confirmed against the
    # literature (Br-alpha corrected, Pa-beta/Pa-alpha air->vacuum; see
    # module docstring).
    LineSpec("He I 1.083", 1.08303),
    LineSpec("Pa-gamma", 1.09412),
    LineSpec("Pa-beta", 1.28217),
    LineSpec("Br-delta", 1.94510),
    LineSpec("Pa-alpha", 1.87563),
    LineSpec("Br-gamma", 2.16613),
    LineSpec("He I 2.059", 2.05869),
    LineSpec("Br-beta", 2.62588),
    LineSpec("Br-alpha", 4.05228),
    LineSpec("Pf-beta", 4.65379),
    # [S III] doublet, added in the first residual-driven extension.
    LineSpec("[S III] 0.9069", 0.90686, prior_offset=-0.0025),
    LineSpec("[S III] 0.9531", 0.95306, prior_offset=-0.0025),
    # Literature-added candidates (see module docstring for confidence
    # notes per line and why each was/wasn't added).
    LineSpec("[C I] 0.8727", 0.8727),
    LineSpec("[C I] 0.9824", 0.9824),
    LineSpec("[C I] 0.9850", 0.9850),
    LineSpec("H2 1-0 S(1)", 2.1218),
    LineSpec("He I 2.113", 2.1132),
    LineSpec("He I 1.7002", 1.7002),
    LineSpec("[Fe II] 1.257", 1.257),
    LineSpec("[Fe II] 1.644", 1.644),
    LineSpec("Pf-gamma", 3.74057),
    LineSpec("Pf-epsilon", 3.03921),
    # Band-6 residual-driven additions (see module docstring's "Band 6
    # correction" section): band 6 had only Pf-beta fitted, so its
    # null-noise calibration was self-contaminated by these same real
    # bumps until they were identified/added here.
    LineSpec("He II 8-7", 4.7635),  # Z=2 hydrogenic Rydberg calc, Delta=0.010um
    LineSpec("Unid. 4.47um band6", 4.4728),  # no H I/He II match found; tentative
    # A third candidate at ~4.61um was tried and DROPPED -- see module
    # docstring's "Band 6 correction" section: it isn't an independent
    # feature, it's degenerate with (and destabilizes) Pf-beta itself.
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
    amp = result.amplitudes[["line", "band", "center", "amplitude", "amplitude_at_floor",
                             "err_formal", "err_cal", "snr_cal", "coverage",
                             "shape_extrapolated", "at_bound", "blended",
                             "resid_rms_inner", "resid_rms_outer"]]
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
