"""Test whether (sum, diff) or (sum, ratio) reparametrization of the
a_left/a_right and dmu_left/dmu_right pairs needs a lower polynomial
degree than fitting each independently, for the same end-to-end curve
accuracy.

Since Chebyshev fitting is linear, a degree-d fit to (a_left, a_right)
independently spans EXACTLY the same function space as a degree-d fit to
(a_sum, a_diff) -- sum/difference of two degree-d polynomials is still
degree-d, and vice versa. So this can only help if the noise/residual
structure differs between the two bases (e.g. correlated per-row fit
noise partially cancels in the difference), or if the TRUE underlying
sum and difference curves have genuinely different smoothness (e.g. a
near-constant geometric asymmetry riding on a slowly-varying common
trend) -- not guaranteed, worth measuring directly.

Uses the raw per-row sweep saved by derive_bandpass_shape_coeffs.py's
`--out` flag -- no need to re-run the expensive per-row curve_fit, just
re-fit cheap 1D polynomials to already-fitted parameters and re-evaluate
the end-to-end curve.

Measured result (see the build report, "parameter-reduction question"
section): sum+diff is numerically IDENTICAL to independent fitting (to
floating-point precision, confirming the linearity argument above) in
every band; sum+ratio never needs a lower degree either. No gain from
either reparametrization -- independent a_left/a_right/dmu_left/dmu_right
(what derive_bandpass_shape_coeffs.py already does) stays the simplest
working basis.

Usage
-----
    .venv/bin/python examples/derive_bandpass_shape_coeffs.py --out /tmp/sweep.npz
    .venv/bin/python examples/test_sum_diff_parameterization.py /tmp/sweep.npz
"""
import sys
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from spherex_tabular_bandpass import Tabular_Bandpass
from ssiaat.spherex_table import BandpassTool
from ssiaat.model.sed import hflatline

DEGREES = range(0, 9)


def load_band(d, band):
    keep = d[f"b{band}_keep"]
    iy_arr = d[f"b{band}_iy_arr"]
    out = {k: d[f"b{band}_out_{k}"] for k in
          ("wc", "tsum", "peak", "rms", "amplitude", "dmu_left", "a_left",
           "dmu_right", "a_right", "floor0", "floor1")}
    return iy_arr, keep, out


def fit_pair_independent(wc, y_left, y_right, deg, domain):
    return (Chebyshev.fit(wc, y_left, deg, domain=domain),
           Chebyshev.fit(wc, y_right, deg, domain=domain))


def reconstruct_independent(polys, wc_eval):
    left, right = polys
    return left(wc_eval), right(wc_eval)


def fit_pair_sumdiff(wc, y_left, y_right, deg, domain):
    s, dd = y_left + y_right, y_left - y_right
    return (Chebyshev.fit(wc, s, deg, domain=domain),
           Chebyshev.fit(wc, dd, deg, domain=domain))


def reconstruct_sumdiff(polys, wc_eval):
    s, dd = polys
    s_v, d_v = s(wc_eval), dd(wc_eval)
    return (s_v + d_v) / 2, (s_v - d_v) / 2


def fit_pair_sumratio(wc, y_left, y_right, deg, domain):
    s, r = y_left + y_right, y_left / y_right
    return (Chebyshev.fit(wc, s, deg, domain=domain),
           Chebyshev.fit(wc, r, deg, domain=domain))


def reconstruct_sumratio(polys, wc_eval):
    s, r = polys
    s_v, r_v = s(wc_eval), r(wc_eval)
    return s_v * r_v / (1 + r_v), s_v / (1 + r_v)


PARAMETERIZATIONS = {
    "independent": (fit_pair_independent, reconstruct_independent),
    "sum+diff": (fit_pair_sumdiff, reconstruct_sumdiff),
    "sum+ratio": (fit_pair_sumratio, reconstruct_sumratio),
}


def end_to_end_curve_rms(tool, band, iy_list, wc_arr, out,
                         a_left_hat, a_right_hat, dmu_left_hat, dmu_right_hat):
    """Reconstruct the FULL curve using the candidate a_left/a_right (or
    dmu_left/dmu_right) at each held-out row, holding every other
    quantity at its own best per-row fitted value (isolates the effect of
    just the pair being tested)."""
    rms_list = []
    for j, iy in enumerate(iy_list):
        w1, t1 = tool.bandpass_model(1024, int(iy), array=band)
        w1 = np.asarray(w1, dtype=float)
        t1 = np.asarray(t1, dtype=float)
        wc = float(wc_arr[iy])
        shape = hflatline(wc, dmu_left_hat[j], a_left_hat[j],
                          dmu_right=dmu_right_hat[j], a_right=a_right_hat[j])
        model = (out["amplitude"][iy] * shape(w1)
                + out["floor0"][iy] + out["floor1"][iy] * (w1 - wc))
        rms_list.append(float(np.sqrt(np.mean((model - t1) ** 2))))
    return float(np.median(rms_list))


def test_pair(tool, band, iy_arr, keep, out, wc, pair_name, left_key, right_key):
    train_mask = keep & (iy_arr % 4 != 0)
    val_mask = keep & (iy_arr % 4 == 0)
    val_iy = iy_arr[val_mask]
    domain = [float(wc[train_mask].min()), float(wc[train_mask].max())]

    y_left, y_right = out[left_key], out[right_key]
    D_band = float(np.nanmedian(out["rms"][keep]))  # direct per-row fit RMS, as reference

    print(f"  --- pair: {pair_name} (band {band}, D_band={D_band:.3e}) ---")
    results = {}
    for pname, (fit_fn, recon_fn) in PARAMETERIZATIONS.items():
        rms_by_deg = {}
        for deg in DEGREES:
            polys = fit_fn(wc[train_mask], y_left[train_mask], y_right[train_mask],
                          deg, domain)
            left_hat, right_hat = recon_fn(polys, wc[val_iy])
            # hold the OTHER pair (dmu if testing a, or a if testing dmu) at its
            # own true per-row fitted value -- isolates this pair's effect.
            if left_key == "a_left":
                dmu_l, dmu_r = out["dmu_left"][val_iy], out["dmu_right"][val_iy]
                a_l, a_r = left_hat, right_hat
            else:
                a_l, a_r = out["a_left"][val_iy], out["a_right"][val_iy]
                dmu_l, dmu_r = left_hat, right_hat
            rms_by_deg[deg] = end_to_end_curve_rms(
                tool, band, val_iy, wc, out, a_l, a_r, dmu_l, dmu_r)
        results[pname] = rms_by_deg
        degs_str = "  ".join(f"d{d}={rms_by_deg[d]:.3e}" for d in DEGREES)
        print(f"    {pname:12s}: {degs_str}")

    # For each parameterization, the smallest degree reaching within 10%
    # of its own best achieved value (a fair per-parameterization "how low
    # can this go" summary).
    print("    -> smallest degree within 10% of that basis's own best:")
    for pname, rms_by_deg in results.items():
        best = min(rms_by_deg.values())
        d_min = next(d for d in DEGREES if rms_by_deg[d] <= 1.10 * best)
        print(f"       {pname:12s}: degree {d_min} (rms={rms_by_deg[d_min]:.3e}, best={best:.3e})")
    return results


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <sweep.npz produced by"
                 " derive_bandpass_shape_coeffs.py --out>")
    d = np.load(sys.argv[1])
    bands = sorted(int(k[1:k.index("_")]) for k in d.files if k.endswith("_keep"))
    bp = Tabular_Bandpass()
    for band in bands:
        print(f"\n=== Band {band} ===")
        iy_arr, keep, out = load_band(d, band)
        wc = out["wc"]
        tool = BandpassTool(bp, band=band)
        test_pair(tool, band, iy_arr, keep, out, wc, "a_left/a_right", "a_left", "a_right")
        test_pair(tool, band, iy_arr, keep, out, wc, "dmu_left/dmu_right", "dmu_left", "dmu_right")


if __name__ == "__main__":
    main()
