"""Multi-band emission-line fitting with a tied central wavelength.

`Model`/`vectorized_lstsq` (this package's other fitting module) are
strictly linear: every model is evaluated once as a fixed function of
wavelength, then one independent linear solve per group (e.g. per pixel).
There is no way to tie a parameter across otherwise-independent groups.

This module is for a different shape of problem: ONE spectrum split
across several bands that may not share a wavelength solution, where an
emission line's central wavelength should be a single value shared
across every band it appears in, while its amplitude is free per band.
That needs a genuinely nonlinear parameter (the shared center), so the
fit is done as separable/variable-projection nonlinear least squares:
for any fixed set of trial centers the model is linear in every
amplitude and baseline coefficient, so those are solved exactly by
weighted least squares inside every residual evaluation, and
`scipy.optimize.least_squares` only has to search the (few) tied
centers. This is fast (a handful of centers, not centers-times-bands)
and better conditioned than fitting everything jointly, since the
amplitude/center coupling that would otherwise plague a fully-joint
nonlinear fit is eliminated exactly at each trial center rather than
searched.

The line-profile SHAPE at a trial center is never a free parameter of
its own -- it is always read off `ssiaat.bandpass_shape_lite.BandpassShapeLite`
at that center. Letting a profile's own width/asymmetry
(`dmu_left`/`a_left`/`dmu_right`/`a_right`) float independently of its
center reintroduces the degeneracy documented on
`BandpassTool.fit_hflatline` (`corr(dmu_left, a_left) = +1.000` when
both float) -- see CLAUDE.md. An intrinsically unresolved emission
line's *observed* shape at a given wavelength literally is the
instrument's per-wavelength bandpass response, so this is not an
approximation of convenience, it's the physically correct profile.

The stack this was built for (a pooled-GP combine) has strongly
correlated noise, so naive `1/error**2`-weighted amplitude errors are
wrong by a large, band-dependent factor. `MultiBandLineModel.fit` always
recalibrates errors via an empirical residual-scan null (see
`_null_mad_for_band`) rather than trusting the formal covariance --
an assumed-covariance GLS correction was tried during this module's
design and is numerically unstable for this data (a near-singular
inverse amplifies high-frequency noise), so don't reach for that instead.

Every band's per-line/extra-model amplitude is solved as a NON-NEGATIVE
weighted least squares by default (`nonneg_lines=True`, see
`MultiBandLineModel`) -- an emission line's flux is not physically
negative, so letting a weak/blended line's coefficient swing negative
(typically to partially cancel a stronger neighbor's imperfectly-modeled
wing) would be a fitting artifact reported as if it were real. Only the
point estimate is constrained; the formal error used for calibrated S/N
and upper limits is still the unconstrained covariance, so a line
pinned at zero still gets a meaningful upper limit rather than a
degenerate one.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares, lsq_linear

from ..bandpass_shape_lite import BandpassShapeLite
from .sed import hflatline

_HFLATLINE_SHAPE_KEYS = ("dmu_left", "a_left", "dmu_right", "a_right")


@dataclass(frozen=True)
class LineSpec:
    """One candidate emission line.

    `rest_wavelength` is the laboratory value (microns); the fit's
    initial guess for this line's center is `rest_wavelength +
    prior_offset`. Set `fixed=True` for a line whose center should never
    move (e.g. a known strong blend used only to soak up flux, or an
    anchor line in `centers_mode="wavelength_solution"`) -- it still gets
    a free per-band amplitude, just no free center/no `delta_b` shift.
    """
    name: str
    rest_wavelength: float
    prior_offset: float = 0.0
    fixed: bool = False

    @property
    def initial_center(self):
        return self.rest_wavelength + self.prior_offset


@dataclass(frozen=True)
class BandSpectrum:
    """One band's data. `wavelength` must be 1D, finite, and monotonic
    increasing within the band (NOT necessarily across bands).

    `extra_models` is for FIXED-shape features that are not appropriate
    to fit as a tied `LineSpec` -- most importantly a genuinely broad
    astrophysical feature (e.g. a PAH complex) whose own intrinsic width
    is comparable to or wider than the instrumental resolution, so it is
    not "an unresolved line broadened by the instrument's bandpass" the
    way `LineSpec` assumes. Each entry is `(name, callable)`, `callable`
    a fixed function of wavelength (e.g. `ssiaat.model.sed.hflattop`);
    unlike a `LineSpec`'s profile, its shape and position never move
    during the fit -- only its per-band linear amplitude is solved for,
    reported separately from `amplitudes` in `MultiBandLineFitResults`
    (see `extra_model_amplitudes`) since it has no tied center to report
    against."""
    band: int
    wavelength: np.ndarray
    value: np.ndarray
    error: np.ndarray
    baseline_degree: int = 3
    extra_models: tuple = ()

    def __post_init__(self):
        w = np.asarray(self.wavelength, dtype=float)
        v = np.asarray(self.value, dtype=float)
        e = np.asarray(self.error, dtype=float)
        if not (w.shape == v.shape == e.shape):
            raise ValueError(f"band {self.band}: wavelength/value/error shape"
                             f" mismatch {w.shape}/{v.shape}/{e.shape}")
        if w.ndim != 1:
            raise ValueError(f"band {self.band}: expected 1D arrays")
        if not np.all(np.isfinite(w)) or not np.all(np.diff(w) > 0):
            raise ValueError(f"band {self.band}: wavelength must be finite"
                             " and strictly increasing")
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(e))):
            raise ValueError(f"band {self.band}: value/error must be finite"
                             " -- drop NaN rows before constructing this")
        object.__setattr__(self, "wavelength", w)
        object.__setattr__(self, "value", v)
        object.__setattr__(self, "error", e)

    @property
    def weight(self):
        return 1.0 / self.error ** 2


def _profile_shape(shape_model, center, band, *, allow_extrapolated):
    """hflatline shape kwargs + FWHM at `center`, band `band`. Returns
    (shape_kwargs, fwhm, extrapolated). Suppresses BandpassShapeLite's
    per-call out-of-domain UserWarning -- callers get `extrapolated`
    back as a flag instead of a warning storm."""
    lo, hi = shape_model.SHAPE_CHEB_DOMAIN[band]
    extrapolated = not (lo <= center <= hi)
    eval_center = min(max(center, lo), hi) if not allow_extrapolated else center
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p = shape_model.params(eval_center, array=band)
    shape_kwargs = {k: p[k] for k in _HFLATLINE_SHAPE_KEYS}
    fwhm = _numeric_fwhm(center, **shape_kwargs)
    return shape_kwargs, fwhm, extrapolated


def _numeric_fwhm(center, dmu_left, a_left, dmu_right, a_right, n=2001):
    halfwidth = 6 * max(dmu_left, dmu_right)
    w = np.linspace(center - halfwidth, center + halfwidth, n)
    y = hflatline(center, dmu_left, a_left, dmu_right=dmu_right, a_right=a_right)(w)
    peak = y.max()
    above = w[y >= 0.5 * peak]
    if above.size < 2:
        return 2 * max(dmu_left, dmu_right)
    return float(above.max() - above.min())


def _normalized_profile(center, dmu_left, a_left, dmu_right, a_right):
    """hflatline shape at `center`, rescaled to peak exactly 1 -- so a
    fitted amplitude means "peak contrast", comparable across bands
    despite hflatline's own peak varying ~0.88-0.99 with dmu/a."""
    shape = hflatline(center, dmu_left, a_left, dmu_right=dmu_right, a_right=a_right)
    halfwidth = 6 * max(dmu_left, dmu_right)
    probe = np.linspace(center - halfwidth, center + halfwidth, 2001)
    peak = shape(probe).max()
    peak = peak if peak > 0 else 1.0

    def profile(w):
        return shape(w) / peak

    return profile


def _weighted_lstsq(X, y, weight, *, lower=None):
    """Weighted least squares via row scaling (numerically nicer than
    forming normal equations directly). Returns (coeffs, cov, resid),
    with `cov = pinv(X^T W X)` UNRESCALED by chi2/dof -- same convention
    as `vectorized_lstsq_arrays` (weights are assumed to already be
    1/variance); singular systems are handled gracefully via `pinv`
    rather than raising, also matching that module.

    `lower`, if given, is a per-column lower-bound array (`-inf` for an
    unconstrained column); used to enforce e.g. amplitude >= 0 for
    emission-line/extra-model columns (see `MultiBandLineModel`'s
    `nonneg_lines`) while leaving the baseline free. When any bound is
    finite, solved via `scipy.optimize.lsq_linear(..., method="bvls")`
    -- an exact active-set method, appropriate here since these design
    matrices are always small and dense -- instead of plain `lstsq`.
    `cov` is STILL the unconstrained `pinv(X^T W X)` regardless of
    `lower`: "how uncertain would this coefficient be without the
    physical constraint," so downstream error/upper-limit reporting
    isn't distorted by which coefficients happened to land on their
    bound -- only the point estimate itself is constrained."""
    sqrt_w = np.sqrt(weight)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    if lower is not None and np.any(np.isfinite(lower)):
        ub = np.full(X.shape[1], np.inf)
        coeffs = lsq_linear(Xw, yw, bounds=(lower, ub), method="bvls").x
    else:
        coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ coeffs
    cov = np.linalg.pinv(Xw.T @ Xw)
    return coeffs, cov, resid


def _baseline_vander(wavelength, wavelength_span, degree):
    lo, hi = wavelength_span
    x = 2 * (wavelength - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(wavelength)
    return chebvander(x, degree)


class MultiBandLineModel:
    """Fit `lines` against `spectra` with tied centers, free per-band
    amplitudes, via variable projection. See the module docstring.

    Parameters
    ----------
    lines : sequence of LineSpec
        Candidate lines. Names must be unique.
    spectra : sequence of BandSpectrum
        One entry per band. Band numbers must be unique.
    shape : BandpassShapeLite, optional
        Reused if given (construction is cheap, but this lets tests
        inject a stub). Built fresh if not given.
    centers_mode : {"free", "wavelength_solution"}
        "free" (default): each non-fixed line gets one free center
        shared across its assigned bands -- what "tied center" means by
        default in this module. "wavelength_solution": every non-fixed
        line's center is pinned to `rest_wavelength + prior_offset`;
        instead each band with >=1 assigned non-fixed line gets one free
        offset `delta_b`, added to every line assigned to that band.
        These are alternate framings of the same fit, not combinable --
        a free center plus a free per-band offset is exactly degenerate
        whenever a band's lines are otherwise unique to it (see the
        design plan/CLAUDE.md for the identifiability argument).
    min_coverage, min_pts, profile_halfwidth_fwhm : float, float, float
        Band-assignment thresholds (see `assignment`).
    allow_extrapolated_shape : bool
        If False (default), a line whose center lies outside a band's
        `BandpassShapeLite.SHAPE_CHEB_DOMAIN` is excluded from that
        band's design (flagged `shape_extrapolated`) rather than using
        an unvalidated extrapolated profile there.
    center_bound_fwhm, min_center_bound : float, float
        Free centers (or deltas) are bounded to
        `initial +/- max(center_bound_fwhm * FWHM, min_center_bound)`.
    normalize_profiles : bool
        Peak-normalize each line profile before fitting (default True;
        see `_normalized_profile`).
    nonneg_lines : bool
        If True (default), constrain every tied-line and extra-model
        amplitude to be >= 0 in each band's linear solve -- an emission
        line's flux is not physically negative, so an unconstrained fit
        letting one go negative (typically a weak/blended line's
        coefficient swinging negative to partially cancel a stronger
        neighbor's imperfectly-modeled wing) is a fitting artifact, not
        a real absorption feature. The baseline is NOT constrained.
        Coefficients pinned at the amplitude=0 floor are flagged
        `amplitude_at_floor=True` in the results; their FORMAL error
        (and hence calibrated S/N / upper limit) is still computed from
        the unconstrained covariance, so a floored line still gets a
        meaningful upper limit rather than a degenerate zero error.
        Set False to recover the old unconstrained behavior (e.g. to
        compare/debug).
    """

    def __init__(self, lines, spectra, *, shape=None, centers_mode="free",
                min_coverage=0.5, min_pts=10, profile_halfwidth_fwhm=3.0,
                allow_extrapolated_shape=False, center_bound_fwhm=0.5,
                min_center_bound=0.005, normalize_profiles=True,
                nonneg_lines=True):
        if centers_mode not in ("free", "wavelength_solution"):
            raise ValueError(f"centers_mode={centers_mode!r}; must be"
                             " 'free' or 'wavelength_solution'")

        self.lines = {}
        for line in lines:
            if line.name in self.lines:
                raise ValueError(f"duplicate line name: {line.name!r}")
            self.lines[line.name] = line

        self.spectra = {}
        for spec in spectra:
            if spec.band in self.spectra:
                raise ValueError(f"duplicate band: {spec.band!r}")
            self.spectra[spec.band] = spec

        self.shape = shape if shape is not None else BandpassShapeLite()
        self.centers_mode = centers_mode
        self.min_coverage = min_coverage
        self.min_pts = min_pts
        self.profile_halfwidth_fwhm = profile_halfwidth_fwhm
        self.allow_extrapolated_shape = allow_extrapolated_shape
        self.center_bound_fwhm = center_bound_fwhm
        self.min_center_bound = min_center_bound
        self.normalize_profiles = normalize_profiles
        self.nonneg_lines = nonneg_lines

        self._assignment = self._compute_assignment()

    # -- Stage: which (line, band) pairs get a design column -------------

    def _compute_assignment(self):
        rows = []
        for line in self.lines.values():
            c0 = line.initial_center
            for band, spec in self.spectra.items():
                if band not in self.shape.SHAPE_CHEB_DOMAIN:
                    continue
                shape_kwargs, fwhm, extrapolated = _profile_shape(
                    self.shape, c0, band, allow_extrapolated=True)
                halfwidth = self.profile_halfwidth_fwhm * fwhm
                lo, hi = c0 - halfwidth, c0 + halfwidth
                in_win = (spec.wavelength >= lo) & (spec.wavelength <= hi)
                npts = int(in_win.sum())

                total_flux, band_flux = self._coverage_flux(
                    c0, shape_kwargs, spec.wavelength, lo, hi)
                coverage = band_flux / total_flux if total_flux > 0 else 0.0

                in_domain = not extrapolated
                assigned = (coverage >= self.min_coverage and npts >= self.min_pts
                           and (in_domain or self.allow_extrapolated_shape))
                rows.append(dict(line=line.name, band=band, coverage=coverage,
                                 npts=npts, fwhm=fwhm, in_domain=in_domain,
                                 shape_extrapolated=extrapolated, assigned=assigned))
        return pd.DataFrame(rows)

    @staticmethod
    def _coverage_flux(center, shape_kwargs, wavelength, lo, hi):
        halfwidth = max(hi - center, center - lo)
        fine = np.linspace(center - halfwidth, center + halfwidth, 4001)
        y = hflatline(center, **shape_kwargs)(fine)
        total_flux = float(np.trapezoid(y, fine))
        data_lo, data_hi = wavelength.min(), wavelength.max()
        in_band = (fine >= max(lo, data_lo)) & (fine <= min(hi, data_hi))
        band_flux = float(np.trapezoid(np.where(in_band, y, 0.0), fine))
        return total_flux, band_flux

    def assignment(self):
        """DataFrame, one row per (line, band) pair evaluated: `coverage`,
        `npts`, `fwhm`, `in_domain`, `shape_extrapolated`, `assigned`.
        Includes pairs that did NOT get assigned, with the reason
        visible in the other columns."""
        return self._assignment.copy()

    def _assigned_bands(self, line_name):
        df = self._assignment
        return sorted(df.loc[(df.line == line_name) & df.assigned, "band"].tolist())

    # -- Stage: initial centers -------------------------------------------

    def _nearest_neighbor_distance(self, line_name, band):
        """Distance (microns) from `line_name`'s initial center to the
        closest OTHER line also assigned to `band`, by their initial
        centers. `inf` if it's the only line assigned there."""
        c0 = self.lines[line_name].initial_center
        others = self._assignment.loc[
            (self._assignment.band == band) & self._assignment.assigned
            & (self._assignment.line != line_name), "line"]
        if len(others) == 0:
            return np.inf
        other_centers = [self.lines[n].initial_center for n in others]
        return float(min(abs(c0 - oc) for oc in other_centers))

    def prescan(self, *, n_grid=200):
        """Slide a single-line-plus-baseline fit across each assigned
        (line, band)'s search window; returns a DataFrame with the
        argmax-formal-S/N trial center and S/N per pair. Used by `fit`
        to build `x0` when not given explicitly.

        The search window is `profile_halfwidth_fwhm * fwhm`, but capped
        at 0.4x the distance to the nearest OTHER candidate line assigned
        to the same band -- otherwise a weak or absent line's blind scan
        can wander into a much stronger neighbor's wing and lock onto
        that instead (measured on the real PN spectrum: He I 2.059's
        default 3-FWHM window reaches to within ~4 FWHM of Pa-alpha,
        close enough for its wing to dominate the scan)."""
        rows = []
        for _, row in self._assignment[self._assignment.assigned].iterrows():
            line = self.lines[row.line]
            spec = self.spectra[row.band]
            c0 = line.initial_center
            neighbor_dist = self._nearest_neighbor_distance(row.line, row.band)
            halfwidth = min(self.profile_halfwidth_fwhm * row.fwhm, 0.4 * neighbor_dist)
            trial_centers = np.linspace(c0 - halfwidth, c0 + halfwidth, n_grid)
            best_snr, best_c = -np.inf, c0
            for c in trial_centers:
                shape_kwargs, _, _ = _profile_shape(
                    self.shape, c, row.band, allow_extrapolated=True)
                profile = (_normalized_profile(c, **shape_kwargs) if self.normalize_profiles
                          else hflatline(c, **shape_kwargs))
                g = profile(spec.wavelength)
                w = spec.weight
                denom = float(np.sum(w * g ** 2))
                if denom <= 0:
                    continue
                amp = float(np.sum(w * spec.value * g)) / denom
                snr = amp * np.sqrt(denom)
                if snr > best_snr:
                    best_snr, best_c = snr, c
            rows.append(dict(line=row.line, band=row.band, center=best_c, snr=best_snr))
        return pd.DataFrame(rows)

    def _initial_centers(self, prescan_df):
        """One initial guess per non-fixed line: the position found by
        its highest-S/N assigned band (a good global guess since the
        true center is shared across bands)."""
        centers = {}
        for name, line in self.lines.items():
            if line.fixed:
                continue
            sub = prescan_df[prescan_df.line == name]
            if len(sub) == 0 or not np.isfinite(sub.snr).any():
                centers[name] = line.initial_center
            else:
                centers[name] = float(sub.loc[sub.snr.idxmax(), "center"])
        return centers

    # -- Design matrix + linear solve for one band, given trial centers --

    def _effective_center(self, line_name, theta_centers, theta_deltas, band):
        line = self.lines[line_name]
        if line.fixed:
            base = line.initial_center
            return base if self.centers_mode == "free" else base
        if self.centers_mode == "free":
            return theta_centers[line_name]
        return line.initial_center + theta_deltas.get(band, 0.0)

    def _band_design(self, band, theta_centers, theta_deltas):
        """Returns (X, line_names, extra_names, extrap_flags). Column
        order is always [tied lines..., extra fixed-shape models...,
        baseline...] -- `_build_results` slices `coeffs` by these three
        counts rather than re-deriving them, so keep this order in sync
        with any caller that indexes into `coeffs` positionally."""
        spec = self.spectra[band]
        lines_here = self._assignment.loc[
            (self._assignment.band == band) & self._assignment.assigned, "line"].tolist()

        cols, line_names, extrap_flags = [], [], {}
        for name in lines_here:
            c = self._effective_center(name, theta_centers, theta_deltas, band)
            shape_kwargs, fwhm, extrapolated = _profile_shape(
                self.shape, c, band, allow_extrapolated=self.allow_extrapolated_shape)
            profile = (_normalized_profile(c, **shape_kwargs) if self.normalize_profiles
                      else hflatline(c, **shape_kwargs))
            cols.append(profile(spec.wavelength))
            line_names.append(name)
            extrap_flags[name] = extrapolated

        extra_names = []
        for name, fn in spec.extra_models:
            cols.append(np.asarray(fn(spec.wavelength), dtype=float))
            extra_names.append(name)

        wavelength_span = (spec.wavelength.min(), spec.wavelength.max())
        baseline = _baseline_vander(spec.wavelength, wavelength_span, spec.baseline_degree)
        for d in range(spec.baseline_degree + 1):
            cols.append(baseline[:, d])

        X = np.column_stack(cols) if cols else np.zeros((len(spec.wavelength), 0))
        return X, line_names, extra_names, extrap_flags

    def _solve_band(self, band, theta_centers, theta_deltas):
        spec = self.spectra[band]
        X, line_names, extra_names, extrap_flags = self._band_design(
            band, theta_centers, theta_deltas)
        if X.shape[1] == 0:
            return dict(coeffs=np.array([]), cov=np.zeros((0, 0)),
                       resid=spec.value.copy(), line_names=line_names,
                       extra_names=extra_names, extrap_flags=extrap_flags)
        lower = None
        if self.nonneg_lines and (line_names or extra_names):
            lower = np.full(X.shape[1], -np.inf)
            lower[:len(line_names) + len(extra_names)] = 0.0
        coeffs, cov, resid = _weighted_lstsq(X, spec.value, spec.weight, lower=lower)
        return dict(coeffs=coeffs, cov=cov, resid=resid, line_names=line_names,
                   extra_names=extra_names, extrap_flags=extrap_flags)

    # -- Nonlinear residual over ALL bands --------------------------------

    def _unpack_theta(self, theta, free_line_names, free_bands):
        if self.centers_mode == "free":
            centers = dict(zip(free_line_names, theta))
            return centers, {}
        deltas = dict(zip(free_bands, theta))
        return {}, deltas

    def _residual(self, theta, free_line_names, free_bands):
        theta_centers, theta_deltas = self._unpack_theta(theta, free_line_names, free_bands)
        chunks = []
        for band in self.spectra:
            solved = self._solve_band(band, theta_centers, theta_deltas)
            chunks.append(solved["resid"] * np.sqrt(self.spectra[band].weight))
        return np.concatenate(chunks) if chunks else np.zeros(0)

    # -- Public: fit -------------------------------------------------------

    def fit(self, *, x0=None, **ls_kwargs):
        free_line_names = [n for n, line in self.lines.items() if not line.fixed]
        free_bands = sorted({b for b in self.spectra
                             if any((self._assignment.band == b) & self._assignment.assigned
                                   & self._assignment.line.isin(free_line_names))})

        if self.centers_mode == "wavelength_solution":
            for band in self.spectra:
                has_line = bool(((self._assignment.band == band)
                                 & self._assignment.assigned).any())
                if not has_line:
                    raise ValueError(
                        f"centers_mode='wavelength_solution' but band {band} has no"
                        " assigned line -- its wavelength offset would be"
                        " unconstrained. Add a line expected in that band, or drop it.")

        prescan_df = self.prescan()
        auto_centers = self._initial_centers(prescan_df)

        if self.centers_mode == "free":
            if x0 is None:
                x0_arr = np.array([auto_centers[n] for n in free_line_names])
            else:
                x0_arr = np.array([x0[n] for n in free_line_names])
            fwhms = {n: self._assignment.loc[
                (self._assignment.line == n) & self._assignment.assigned, "fwhm"
            ].mean() for n in free_line_names}
            # Cap each line's bound by the nearest OTHER line's initial
            # center in any band it's assigned to (mirrors prescan's own
            # cap) -- otherwise the FINAL bounded refinement, not just
            # the prescan, can still let two close lines swap/blend
            # flux with each other (measured: He I 1.083 and Pa-gamma
            # 1.094, 0.011um apart in rest wavelength -- closer than one
            # band-1 FWHM -- do exactly this without the cap).
            neighbor_caps = {
                n: min((self._nearest_neighbor_distance(n, b) for b in
                       self._assignment.loc[(self._assignment.line == n)
                                           & self._assignment.assigned, "band"]),
                      default=np.inf)
                for n in free_line_names}
            bounds_half = np.array([
                max(min(self.center_bound_fwhm * fwhms[n], 0.4 * neighbor_caps[n]),
                   self.min_center_bound)
                for n in free_line_names])
        else:
            if x0 is None:
                x0_arr = np.zeros(len(free_bands))
            else:
                x0_arr = np.array([x0.get(b, 0.0) for b in free_bands])
            fwhms = self._assignment[self._assignment.assigned].groupby("band")["fwhm"].mean()
            bounds_half = np.array([
                max(self.center_bound_fwhm * fwhms.get(b, 0.01), self.min_center_bound)
                for b in free_bands])

        lo = x0_arr - bounds_half
        hi = x0_arr + bounds_half

        ls_defaults = dict(x_scale=1e-3, diff_step=1e-3, method="trf",
                          xtol=1e-12, ftol=1e-12)
        ls_defaults.update(ls_kwargs)
        result = least_squares(self._residual, x0_arr, bounds=(lo, hi),
                               args=(free_line_names, free_bands), **ls_defaults)

        theta_centers, theta_deltas = self._unpack_theta(
            result.x, free_line_names, free_bands)
        at_bound = {}
        for i, (name_or_band) in enumerate(
                free_line_names if self.centers_mode == "free" else free_bands):
            rel = min(abs(result.x[i] - lo[i]), abs(result.x[i] - hi[i]))
            at_bound[name_or_band] = bool(rel <= 0.01 * bounds_half[i])

        return self._build_results(theta_centers, theta_deltas, at_bound, prescan_df)

    # -- Assembling MultiBandLineFitResults --------------------------------

    def _build_results(self, theta_centers, theta_deltas, at_bound, prescan_df):
        amp_rows = []
        extra_rows = []
        band_solutions = {}
        for band in self.spectra:
            solved = self._solve_band(band, theta_centers, theta_deltas)
            band_solutions[band] = solved
            spec = self.spectra[band]
            coeffs, cov, resid = solved["coeffs"], solved["cov"], solved["resid"]
            n_lines = len(solved["line_names"])
            for j, extra_name in enumerate(solved["extra_names"]):
                i = n_lines + j
                amp = float(coeffs[i])
                err_formal = float(np.sqrt(max(cov[i, i], 0.0)))
                extra_rows.append(dict(
                    name=extra_name, band=band, amplitude=amp, err_formal=err_formal,
                    snr_formal=(amp / err_formal if err_formal > 0 else np.nan),
                    amplitude_at_floor=bool(self.nonneg_lines and amp <= 1e-9)))
            for i in range(n_lines):
                line_name = solved["line_names"][i]
                c = self._effective_center(line_name, theta_centers, theta_deltas, band)
                amp = float(coeffs[i])
                err_formal = float(np.sqrt(max(cov[i, i], 0.0)))
                shape_kwargs, fwhm, extrapolated = _profile_shape(
                    self.shape, c, band, allow_extrapolated=self.allow_extrapolated_shape)
                inner_rms, outer_rms = self._inner_outer_resid(
                    spec, resid, c, fwhm)
                amp_rows.append(dict(
                    line=line_name, band=band, center=c, amplitude=amp,
                    err_formal=err_formal, snr_formal=(amp / err_formal
                                                       if err_formal > 0 else np.nan),
                    coverage=float(self._assignment.loc[
                        (self._assignment.line == line_name)
                        & (self._assignment.band == band), "coverage"].iloc[0]),
                    fwhm=fwhm, shape_extrapolated=extrapolated,
                    at_bound=at_bound.get(
                        line_name if self.centers_mode == "free" else band, False),
                    amplitude_at_floor=bool(self.nonneg_lines and amp <= 1e-9),
                    resid_rms_inner=inner_rms, resid_rms_outer=outer_rms,
                ))
        amplitudes = pd.DataFrame(amp_rows)
        amplitudes["blended"] = False
        for band, group in amplitudes.groupby("band"):
            centers = group["center"].to_numpy()
            fwhms = group["fwhm"].to_numpy()
            idx = group.index.to_numpy()
            blended = np.zeros(len(idx), dtype=bool)
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if abs(centers[i] - centers[j]) < 0.5 * max(fwhms[i], fwhms[j]):
                        blended[i] = blended[j] = True
            amplitudes.loc[idx, "blended"] = blended

        null_mad = self._compute_null_mad(theta_centers, theta_deltas, band_solutions)
        amplitudes = amplitudes.merge(
            null_mad.rename("null_mad"), left_on="band", right_index=True, how="left")
        amplitudes["err_cal"] = amplitudes["err_formal"] * amplitudes["null_mad"]
        amplitudes["snr_cal"] = np.where(
            amplitudes["err_cal"] > 0, amplitudes["amplitude"] / amplitudes["err_cal"], np.nan)

        chi2_red = {}
        for band, solved in band_solutions.items():
            spec = self.spectra[band]
            dof = max(len(spec.wavelength) - len(solved["coeffs"]), 1)
            chi2_red[band] = float(np.sum(solved["resid"] ** 2 * spec.weight) / dof)

        centers = pd.Series(theta_centers, name="center") if theta_centers else None
        deltas = pd.Series(theta_deltas, name="delta") if theta_deltas else None
        extra_model_amplitudes = pd.DataFrame(
            extra_rows, columns=["name", "band", "amplitude", "err_formal", "snr_formal",
                                 "amplitude_at_floor"])

        return MultiBandLineFitResults(
            model=self, centers=centers, deltas=deltas, amplitudes=amplitudes,
            extra_model_amplitudes=extra_model_amplitudes,
            null_mad=null_mad, chi2_red=pd.Series(chi2_red, name="chi2_red"),
            band_solutions=band_solutions, prescan=prescan_df,
        )

    @staticmethod
    def _inner_outer_resid(spec, resid, center, fwhm):
        dist = np.abs(spec.wavelength - center)
        inner = dist <= fwhm
        outer = (dist > fwhm) & (dist <= 3 * fwhm)
        inner_rms = float(np.sqrt(np.mean(resid[inner] ** 2))) if inner.any() else np.nan
        outer_rms = float(np.sqrt(np.mean(resid[outer] ** 2))) if outer.any() else np.nan
        return inner_rms, outer_rms

    def _compute_null_mad(self, theta_centers, theta_deltas, band_solutions,
                          n_grid=200, mask_fwhm=2.0):
        """Empirical error-calibration factor per band: mask the
        vicinity of every fitted line, slide a matched-filter probe
        line across what's left of the residual spectrum, and take the
        MAD of the resulting formal-S/N distribution (scaled to a
        Gaussian-equivalent sigma). This measures the ACTUAL noise
        properties of the (correlated, GP-stacked) data, which the
        formal `1/error**2`-weighted covariance does not capture.

        CAVEAT (found on the real PN spectrum, band 6 specifically, not
        hypothetical): "the vicinity of every fitted line" only masks
        lines that are IN the model -- a real, unmodeled feature sits
        unmasked in this scan and inflates the very calibration factor
        meant to judge whether it, and everything else in that band, is
        significant. With only one fitted line (Pf-beta), band 6's
        naive null_mad (4.65, worst of any band) turned out to be
        dominated by three additional real residual bumps, not genuine
        noise: manually masking those too dropped it to ~1.8, at which
        point two of the three bumps became clear multi-sigma
        detections. An automatic robust-statistics fix (iterative
        sigma-clipping) was tried and DOESN'T work here -- at ~40% of
        the probe grid affected, the contamination is too close to a
        typical robust estimator's ~50% breakdown point for a single
        pass to self-correct. The real fix is the one already built
        into this class: identify the missing line(s), add them to the
        model, and refit -- once genuinely assigned, the existing mask
        above excludes them from this scan correctly, no special-casing
        needed. This is a standing risk for any band with few
        already-known lines relative to how many real features it
        actually has -- treat that band's null_mad/significance as
        provisional until residual structure there has been checked
        for missed lines, not trusted at face value."""
        out = {}
        for band, solved in band_solutions.items():
            spec = self.spectra[band]
            resid = solved["resid"]
            rows = self._assignment[(self._assignment.band == band)
                                    & self._assignment.assigned]

            mask = np.ones_like(spec.wavelength, dtype=bool)
            for _, row in rows.iterrows():
                c = self._effective_center(row.line, theta_centers, theta_deltas, band)
                mask &= np.abs(spec.wavelength - c) > mask_fwhm * row.fwhm

            w = spec.weight
            masked_wavelength = spec.wavelength[mask] if mask.any() else spec.wavelength
            lo, hi = masked_wavelength.min(), masked_wavelength.max()
            trial_centers = np.linspace(lo, hi, n_grid)
            snrs = []
            for c in trial_centers:
                shape_kwargs, _, _ = _profile_shape(self.shape, c, band,
                                                     allow_extrapolated=True)
                profile = (_normalized_profile(c, **shape_kwargs) if self.normalize_profiles
                          else hflatline(c, **shape_kwargs))
                g = np.where(mask, profile(spec.wavelength), 0.0)
                denom = float(np.sum(w * g ** 2))
                if denom <= 0:
                    continue
                amp = float(np.sum(w * resid * g)) / denom
                snrs.append(amp * np.sqrt(denom))
            if len(snrs) < 5:
                out[band] = 1.0
                continue
            snrs = np.asarray(snrs)
            mad = np.median(np.abs(snrs - np.median(snrs))) * 1.4826
            out[band] = float(max(mad, 1e-6))
        return pd.Series(out, name="null_mad")


@dataclass
class MultiBandLineFitResults:
    """Result of `MultiBandLineModel.fit`.

    `centers` (mode "free") or `deltas` (mode "wavelength_solution") is
    populated depending on which mode produced this result; the other is
    None. `amplitudes` is the full per-(line,band) table: `amplitude`,
    `err_formal`, `err_cal`, `snr_formal`, `snr_cal`, `coverage`, `fwhm`,
    `shape_extrapolated`, `at_bound`, `resid_rms_inner`, `resid_rms_outer`,
    `blended` (the inner/outer split matters specifically for bands whose
    real profile is known to be flatter-topped than `BandpassShapeLite`
    predicts -- e.g. bands 5/6 for SPHEREx -- where a real shape mismatch
    shows up as a *signed* inner-vs-outer pattern, not just elevated
    combined RMS; `blended=True` means this line's fitted center in this
    band landed within half a FWHM of another line's -- their individual
    amplitudes are then near-degenerate, from design-matrix collinearity
    rather than any fitting mistake, and shouldn't be trusted
    independently, only in combination). `extra_model_amplitudes` is the
    separate table for each `BandSpectrum.extra_models` entry (band,
    name, amplitude, err_formal, snr_formal) -- these have no tied
    center, so they're not mixed into `amplitudes`."""
    model: MultiBandLineModel
    centers: pd.Series
    deltas: pd.Series
    amplitudes: pd.DataFrame
    extra_model_amplitudes: pd.DataFrame
    null_mad: pd.Series
    chi2_red: pd.Series
    band_solutions: dict = field(repr=False)
    prescan: pd.DataFrame = field(repr=False)

    def line_table(self, *, detect_snr=5.0, marginal_snr=3.0):
        """One row per line: combined calibrated S/N (quadrature over
        bands with positive amplitude), cross-band consistency chi2,
        classification, and a 3-sigma calibrated upper limit for
        non-detections."""
        rows = []
        for name, line in self.model.lines.items():
            sub = self.amplitudes[self.amplitudes.line == name]
            if len(sub) == 0:
                rows.append(dict(line=name, rest_wavelength=line.rest_wavelength,
                                 center=np.nan, n_bands=0, snr_comb=np.nan,
                                 chi2_consist=np.nan, classification="not_assigned",
                                 upper_limit=np.nan))
                continue
            center = (float(self.centers[name]) if self.centers is not None
                      and name in self.centers.index
                      else float(sub["center"].iloc[0]))
            positive = sub[sub.amplitude > 0]
            snr_comb = float(np.sqrt(np.sum(positive.snr_cal ** 2))) if len(positive) else 0.0
            if len(sub) > 1:
                weights = 1.0 / sub.err_cal ** 2
                mean_amp = float(np.sum(sub.amplitude * weights) / np.sum(weights))
                chi2_consist = float(np.sum((sub.amplitude - mean_amp) ** 2
                                            / sub.err_cal ** 2))
            else:
                chi2_consist = np.nan
            if snr_comb >= detect_snr:
                classification = "detected"
            elif snr_comb >= marginal_snr:
                classification = "marginal"
            else:
                classification = "non_detection"
            upper_limit = float(max(sub.amplitude.max(), 0.0)
                                + 3 * sub.err_cal.min()) if classification != "detected" else np.nan
            rows.append(dict(line=name, rest_wavelength=line.rest_wavelength,
                             center=center, n_bands=len(sub), snr_comb=snr_comb,
                             chi2_consist=chi2_consist, classification=classification,
                             upper_limit=upper_limit))
        return pd.DataFrame(rows)

    def model_curve(self, band, wavelength=None, *, only_lines=None, include_baseline=True):
        """The fitted model for `band`, on its own data grid by default or
        a caller-supplied `wavelength`. By default (`only_lines=None`):
        baseline + every line assigned to this band. Pass `only_lines`
        (an iterable of line names) to isolate just those lines' own
        contribution -- e.g. for a per-line zoom plot that shouldn't
        show a nearby stronger line's peak leaking into the same band's
        full curve. `include_baseline=False` drops the baseline term
        too (useful combined with `only_lines` for a pure line-shape
        overlay)."""
        m = self.model
        spec = m.spectra[band]
        w = spec.wavelength if wavelength is None else np.asarray(wavelength, dtype=float)
        theta_centers = ({name: float(self.centers[name]) for name in self.centers.index}
                        if self.centers is not None else {})
        theta_deltas = ({band_: float(self.deltas[band_]) for band_ in self.deltas.index}
                       if self.deltas is not None else {})

        solved = self.band_solutions[band]
        coeffs = solved["coeffs"]
        line_names = solved["line_names"]
        n_lines, n_extra = len(line_names), len(solved["extra_names"])

        total = np.zeros_like(w)
        for i, name in enumerate(line_names):
            if only_lines is not None and name not in only_lines:
                continue
            c = m._effective_center(name, theta_centers, theta_deltas, band)
            shape_kwargs, _, _ = _profile_shape(m.shape, c, band,
                                                allow_extrapolated=m.allow_extrapolated_shape)
            profile = (_normalized_profile(c, **shape_kwargs) if m.normalize_profiles
                      else hflatline(c, **shape_kwargs))
            total = total + coeffs[i] * profile(w)

        if only_lines is None:
            # extra_models (e.g. a fixed-shape PAH complex) are part of
            # the band's full model, but excluded when isolating one
            # line's own contribution via `only_lines`.
            for j, (_, fn) in enumerate(spec.extra_models):
                total = total + coeffs[n_lines + j] * np.asarray(fn(w), dtype=float)

        if include_baseline:
            span = (spec.wavelength.min(), spec.wavelength.max())
            baseline = _baseline_vander(w, span, spec.baseline_degree)
            for d in range(spec.baseline_degree + 1):
                total = total + coeffs[n_lines + n_extra + d] * baseline[:, d]
        return total

    def to_frame(self):
        """The per-(line,band) amplitude table, merged with each line's
        combined classification -- directly serializable (ECSV/parquet)."""
        return self.amplitudes.merge(
            self.line_table()[["line", "snr_comb", "classification"]],
            on="line", how="left")


def fit_multiband_lines(lines, spectra, **kwargs):
    """Convenience wrapper: `MultiBandLineModel(lines, spectra, **kwargs).fit()`."""
    return MultiBandLineModel(lines, spectra, **kwargs).fit()
