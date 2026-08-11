"""Line-model registry: the single inventory of SPHEREx line/feature models.

A line's profile is one of several kinds, selected by ``LineModel.kind``:

* ``"bandpass"`` -- an *unresolved* emission line whose observed profile IS the
  SPHEREx instrument bandpass at its wavelength. Built by :func:`bandpass_line`
  from :class:`ssiaat.bandpass_shape_lite.BandpassShapeLite` (band-/center-
  specific), unit-peak normalized so the fitted coefficient is the line's peak
  amplitude. Supersedes the four near-identical ``_bandpass_line`` copies that
  used to live in ``work_hips/model_*.py``.

* ``"hflattop"`` / ``"hflatline"`` / ``"gauss"`` -- a **manually-specified**
  profile whose parameters are given as data in ``LineModel.params``. Use these
  when the bandpass model is not appropriate: a spectrally *resolved* feature
  (PAH), or a line in a band-5/6 range where ``BandpassShapeLite``'s polynomial
  fit is imprecise. Params: hflattop ``{mu0, mu1, a}``, hflatline
  ``{mu_center_left, dmu_left, a_left[, dmu_right, a_right]}``, gauss
  ``{center, sigma}``. For accurate band-5/6 hflatline params fit to the *real*
  tabulated bandpass, see :func:`fit_hflatline_params`. Adding a manual line is a
  one-row registry edit, e.g.::

      LineModel("pah_narrow", 4, 3.29, kind="hflattop",
                params=dict(mu0=3.2481, mu1=3.3382, a=0.7))

* ``"callable"`` -- an escape hatch: an arbitrary ``shape_factory()`` returning a
  ``wvl -> ndarray`` callable, for anything the declarative kinds can't express.

The legacy analytic lines (``br_a``, ``pa_b``, ``br_b``, PAH, ``h2o_ice``,
``h2_46957``) are ``"hflattop"``/``"hflatline"`` carrying their old hand-tuned
params (the ad-hoc ``d_shift`` preserved verbatim in the params).

Every profile is normalized to **unit peak** by default (``LineModel.norm``, also
overridable per call: ``get_line(name, norm="area")``). ``"area"`` gives unit
integral using each shape's analytic integral -- ``gauss`` (``sigma*sqrt(2*pi)``),
``hflattop`` and symmetric ``hflatline`` (``(mu1-mu0)/(1-e^{-8a})``), with a
numeric fallback only for the asymmetric ``hflatline`` -- and ``"none"`` leaves
it raw. The bandpass lines were already
unit-peak (unchanged); normalizing the manual top-hats rescales their amplitude
by a per-line constant (their raw peak was < 1), so manual-line maps differ from
pre-normalization ones by exactly that factor.

``LINES`` maps a line name to a :class:`LineModel`; ``FEATURES`` groups lines into
fittable features carrying their default continuum window. :func:`get_line` returns
a bare ``wvl -> ndarray`` callable (what :class:`ssiaat.model.fitting.Model` wants);
:func:`get_line_model` assembles a whole feature's lines + continuum ramps into a
``Model``.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..bandpass_shape_lite import BandpassShapeLite
from .fitting import Model
from .shapes import get_cont_model, hflattop, hflatline, gauss

# One shared bandpass reconstruction (central column, all bands).
_BP = BandpassShapeLite()


def _norm_divisor(y, w, mode):
    """Value to DIVIDE a profile by for the given normalization (dividing, not
    multiplying by a reciprocal, keeps the peak path bit-identical to the old
    ``t / peak``): ``"peak"`` -> max, ``"area"`` -> integral, ``"none"``/None -> 1."""
    if mode in (None, "none"):
        return 1.0
    if mode == "peak":
        return float(np.nanmax(y))
    if mode == "area":
        return float(np.trapezoid(y, w))
    raise ValueError(f"norm must be 'peak', 'area' or 'none'; got {mode!r}")


def bandpass_line(rest_wavelength, band, *, prior_offset=0.0, halfwidth=0.08,
                  norm="peak"):
    """Callable line profile = the SPHEREx instrument bandpass centred at
    ``rest_wavelength + prior_offset``, normalised over a dense local grid of
    half-width ``halfwidth`` (um): ``norm="peak"`` (default; fitted coefficient
    is the line's peak amplitude), ``"area"`` (unit integral), or ``"none"``.

    The scale is fixed once from that grid, so the returned profile is
    independent of the caller's ``wvl`` sampling. This is the single factory the
    per-feature ``work_hips`` models used to each re-implement (only ``halfwidth``
    differed: 0.06 for the band-1 lines, 0.08 for band 2, 0.10 for band 3).
    """
    center = rest_wavelength + prior_offset
    wfine = np.linspace(center - halfwidth, center + halfwidth, 2001)
    _, tf = _BP(center, array=band, wvl=wfine)
    s = _norm_divisor(tf, wfine, norm)

    def line(wvl):
        _, t = _BP(center, array=band, wvl=np.asarray(wvl, dtype=float))
        return t / s

    line.__name__ = f"bp_line_b{band}_{rest_wavelength}"
    return line


_MANUAL_SHAPES = {"hflattop": hflattop, "hflatline": hflatline, "gauss": gauss}


def _manual_window(kind, params):
    """(center, half_span) for the normalization grid of a manual shape -- wide
    enough to capture the whole profile (peak in the middle, edges/tails out to
    ``half_span``)."""
    if kind == "hflattop":
        mu0, mu1 = params["mu0"], params["mu1"]
        return 0.5 * (mu0 + mu1), 3.0 * abs(mu1 - mu0) + 0.1
    if kind == "hflatline":
        dl = params.get("dmu_left", 0.05)
        dr = params.get("dmu_right", dl)
        return params["mu_center_left"], 5.0 * max(dl, dr) + 0.1
    if kind == "gauss":
        return params["center"], 10.0 * params["sigma"]
    raise ValueError(f"no normalization window for kind {kind!r}")


def _normalize_manual(raw, mode, kind, params):
    """Scale a manual-shape callable to unit peak / unit area (or leave as-is).

    Area normalization uses each shape's analytic integral where available
    (``gauss``, ``hflattop``, symmetric ``hflatline`` -- via their ``.area``
    property) and falls back to numeric integration only for the asymmetric
    ``hflatline`` (no clean elementary closed form)."""
    if mode in (None, "none"):
        return raw
    if mode == "area":
        try:
            s = raw.area                                # analytic (exact)
        except (AttributeError, NotImplementedError):
            center, hs = _manual_window(kind, params)
            w = np.linspace(center - hs, center + hs, 8001)
            s = _norm_divisor(raw(w), w, "area")
    else:                                               # peak
        center, hs = _manual_window(kind, params)
        w = np.linspace(center - hs, center + hs, 8001)  # odd: samples center
        s = _norm_divisor(raw(w), w, "peak")

    def scaled(wvl):
        return raw(np.asarray(wvl, dtype=float)) / s

    scaled.__name__ = f"{kind}_{mode}norm"
    return scaled


@dataclass(frozen=True)
class LineModel:
    """One registered line/feature model.

    ``rest_wavelength`` is the laboratory (vacuum) center in microns -- the
    quantity a ``kind="bandpass"`` profile is built at (plus ``prior_offset``,
    the explicit replacement for the old magic ``d_shift``). For a manual kind
    (``"hflattop"``/``"hflatline"``/``"gauss"``) it is metadata only; the profile
    is built from ``params``. For ``kind="callable"`` the profile comes from
    ``shape_factory()``. ``blend_group`` tags lines whose individual amplitudes
    are only trustworthy as a sum (design-matrix collinear), e.g. He I / Pa-gamma.
    """
    name: str
    band: int
    rest_wavelength: float
    prior_offset: float = 0.0
    kind: str = "bandpass"          # bandpass | hflattop | hflatline | gauss | callable
    halfwidth: float = 0.08
    blend_group: Optional[str] = None
    notes: str = ""
    norm: str = "peak"              # "peak" | "area" | "none" (default unit peak)
    params: dict = field(default_factory=dict, compare=False)
    shape_factory: Optional[Callable[[], Callable]] = field(default=None, repr=False,
                                                            compare=False)

    @property
    def center(self):
        return self.rest_wavelength + self.prior_offset

    def profile(self, *, shift=0.0, band=None, norm=None):
        """Bare ``wvl -> ndarray`` callable for this line, normalised to unit
        peak (default), unit area (``norm="area"``) or unnormalised
        (``norm="none"``). ``norm`` overrides the line's own ``self.norm``."""
        mode = self.norm if norm is None else norm
        if self.kind == "bandpass":
            b = self.band if band is None else band
            return bandpass_line(self.rest_wavelength, b,
                                 prior_offset=self.prior_offset - shift,
                                 halfwidth=self.halfwidth, norm=mode)
        if self.kind in _MANUAL_SHAPES:
            raw = _MANUAL_SHAPES[self.kind](**self.params)
            return _normalize_manual(raw, mode, self.kind, self.params)
        if self.kind == "callable":
            return self.shape_factory()   # escape hatch: factory owns its own scaling
        raise ValueError(f"{self.name!r}: unknown kind {self.kind!r}; must be one of"
                         f" 'bandpass', {sorted(_MANUAL_SHAPES)}, 'callable'")

    def to_linespec(self):
        """As a :class:`ssiaat.model.multiband_lines.LineSpec` (bandpass lines
        only) so the same registry can feed the multi-band fitter."""
        if self.kind != "bandpass":
            raise ValueError(f"{self.name!r} is kind={self.kind!r}; only"
                             " bandpass lines map to a LineSpec")
        from .multiband_lines import LineSpec
        return LineSpec(self.name, self.rest_wavelength, self.prior_offset)


# band-1 bandpass lines use a 0.06 um peak-norm half-width, band 2 -> 0.08,
# band 3 -> 0.10 (the SPHEREx bandpass widens with wavelength); preserved from
# the original per-file _bandpass_line copies.
_BANDPASS = [
    LineModel("he_i", 1, 1.0830, halfwidth=0.06, blend_group="he_pag",
              notes="He I 2s-2p triplet; blended with Pa-gamma (trust sum)"),
    LineModel("pa_gamma", 1, 1.0941, halfwidth=0.06, blend_group="he_pag",
              notes="Paschen-gamma (n=6->3); blended with He I (trust sum)"),
    LineModel("s3_9069", 1, 0.9069, halfwidth=0.06,
              notes="[S III] 9069; resolved from 9531"),
    LineModel("s3_9531", 1, 0.9531, halfwidth=0.06,
              notes="[S III] 9531; 9531/9069 ~2.5-2.6 is a reddening diagnostic"),
    LineModel("feii_1257", 2, 1.2570, halfwidth=0.08, blend_group="feii_pab",
              notes="[Fe II] a4D-a4F; mildly blended with Pa-beta"),
    LineModel("pa_beta", 2, 1.2822, halfwidth=0.08, blend_group="feii_pab",
              notes="Paschen-beta (n=5->3); mildly blended with [Fe II] 1.257"),
    LineModel("feii_1644", 3, 1.644, halfwidth=0.10,
              notes="[Fe II] 1.644; shock/SNR tracer, ~absent in PNe"),
]

# Manually-specified analytic profiles. The legacy hand-tuned values are kept as
# exact expressions (incl. the old d_shift = 0.012 - 0.002 and a = 0.75 + 0.8) so
# the profiles are byte-identical to the previous getters. Band 5/6 lines are
# hand-tuned here precisely because BandpassShapeLite is least precise there.
_SHAPE = [
    LineModel("br_a", 5, 4.0523, kind="hflattop",
              params=dict(mu0=4.0372 - (0.012 - 0.002), mu1=4.0724 - (0.012 - 0.002),
                          a=0.75 + 0.8),
              notes="H I Br-alpha (band 5); hand-tuned hflattop with legacy d_shift"),
    LineModel("pa_b", 2, 1.2822, kind="hflattop",
              params=dict(mu0=1.2693 - 0.004, mu1=1.2988 - 0.004, a=0.75 + 0.8),
              notes="H I Pa-beta (band 2); hand-tuned hflattop. cf bandpass 'pa_beta'"),
    LineModel("br_b", 4, 2.6259, kind="hflattop",
              params=dict(mu0=2.605, mu1=2.647, a=1.5),
              notes="H I Br-beta (band 4); hand-tuned hflattop"),
    LineModel("pah_narrow", 4, 3.2941, kind="hflattop",
              params=dict(mu0=3.2481, mu1=3.3382, a=0.7),
              notes="3.3 um PAH aromatic component (resolved/broad, not a bandpass line)"),
    LineModel("pah_broad", 4, 3.42, kind="hflattop",
              params=dict(mu0=3.32, mu1=3.52, a=0.9),
              notes="3.4 um PAH aliphatic component (resolved/broad)"),
    LineModel("h2o_ice", 4, 3.07, kind="hflatline",
              params=dict(mu_center_left=3.07, dmu_left=0.23, a_left=1.3, a_right=0.7),
              notes="H2O-ice absorption ~3.05 um (broad; coeff comes out negative)"),
    LineModel("h2_46957", 6, 4.6947, kind="hflattop",
              params=dict(mu0=4.6947 - 0.017 - (0.012 - 0.002),
                          mu1=4.6947 + 0.017 - (0.012 - 0.002), a=0.75 + 0.8),
              notes="H2 0-0 S(9) 4.6947 um (band 6); hand-tuned, BandpassShapeLite weak here"),
]

LINES: dict[str, LineModel] = {lm.name: lm for lm in (_BANDPASS + _SHAPE)}


@dataclass(frozen=True)
class Feature:
    """A fittable feature: an ordered set of (output-name, registry line-name)
    component pairs, plus the continuum span == default fit window. ``band`` is
    the band(s) the feature is fit in. ``blended`` flags a feature whose
    component amplitudes should be trusted only as a sum."""
    components: tuple           # tuple[(out_name, line_key), ...]
    window: tuple               # (w0, w1)
    band: object                # int or tuple[int, ...]
    blended: bool = False
    notes: str = ""


FEATURES: dict[str, Feature] = {
    "br_b": Feature((("br_b", "br_b"),), (2.50, 2.75), 4),
    "br_a": Feature((("br_a", "br_a"),), (3.9, 4.2), 5),
    "pah33": Feature((("pah_narrow", "pah_narrow"), ("pah_broad", "pah_broad")),
                     (3.10, 3.65), 4),
    "ice_pah": Feature((("ice", "h2o_ice"), ("pah_narrow", "pah_narrow"),
                        ("pah_broad", "pah_broad")), (2.70, 3.65), 4),
    "h2o_ice": Feature((("h2o_ice", "h2o_ice"),), (2.80, 3.35), 4),
    "h2_46957": Feature((("h2_46957", "h2_46957"),), (4.5, 4.9), 6),
    "he_pag": Feature((("he_i", "he_i"), ("pa_gamma", "pa_gamma")),
                      (1.02, 1.13), 1, blended=True,
                      notes="He I / Pa-gamma blended -> trust the sum"),
    "s3": Feature((("s3_9069", "s3_9069"), ("s3_9531", "s3_9531")),
                  (0.85, 1.0), 1),
    "feii_pab": Feature((("feii_1257", "feii_1257"), ("pa_beta", "pa_beta")),
                        (1.20, 1.35), 2,
                        notes="mildly blended but separable; sum most robust"),
    "feii_1644": Feature((("feii_1644", "feii_1644"),), (1.58, 1.74), 3,
                         notes="best fit combines band 2 (blue cont) + band 3"),
}


def get_line(name, *, shift=0.0, band=None, norm=None):
    """Bare ``wvl -> ndarray`` callable for the registered line ``name``
    (``norm`` overrides the line's default unit-peak normalization)."""
    try:
        lm = LINES[name]
    except KeyError:
        raise KeyError(f"unknown line {name!r}; available: {sorted(LINES)}") from None
    return lm.profile(shift=shift, band=band, norm=norm)


def feature_spec(name, *, shift=0.0, band=None, norm=None):
    """``dict(lines={out_name: callable}, window=(w0, w1))`` for a feature --
    the shape the ``work_hips`` drivers expect from their ``FEATURES`` registry."""
    feat = FEATURES[name]
    lines = {out: get_line(key, shift=shift, band=band, norm=norm)
             for out, key in feat.components}
    return dict(lines=lines, window=feat.window)


def get_line_model(feature=None, *, lines=None, window=None, shift=0.0,
                   band=None, norm=None, cont_a=0.3):
    """A fittable :class:`ssiaat.model.fitting.Model` = the feature's line
    components + two continuum ramps spanning ``window`` (default: the feature's
    own window). Pass either ``feature`` (a key of :data:`FEATURES`) or an
    explicit ``lines`` dict + ``window``. ``norm`` sets the line normalization
    (default unit peak)."""
    if feature is not None:
        feat = FEATURES[feature]
        comp = {out: get_line(key, shift=shift, band=band, norm=norm)
                for out, key in feat.components}
        win = feat.window if window is None else window
    else:
        if lines is None or window is None:
            raise ValueError("pass either feature=, or both lines= and window=")
        comp = lines
        win = window
    conts = dict(zip(("cont_lo", "cont_hi"), get_cont_model(*win, a=cont_a)))
    return Model(comp, conts)


def fit_hflatline_params(center, band, *, mu_center=None, iy=None, symmetric=False):
    """Derive ``hflatline`` params by least-squares fitting the REAL tabulated
    SPHEREx bandpass at ``center`` (or detector row ``iy``) in ``band``.

    A more accurate manual profile than ``BandpassShapeLite`` where the latter's
    polynomial fit is imprecise (notably bands 5-6). Returns a dict of hflatline
    kwargs (``mu_center_left, dmu_left, a_left, dmu_right, a_right``) with the
    fit's ``amplitude`` dropped -- ready to paste as ``params=`` on a
    ``kind="hflatline"`` :class:`LineModel`::

        p = fit_hflatline_params(4.052, 5)
        LineModel("br_a", 5, 4.0523, kind="hflatline", params=p)

    OFFLINE tool: needs the optional ``spherex_tabular_bandpass`` package (the
    full tabulated bandpass, not a hard dependency of ssiaat); raises
    ``ImportError`` with guidance if it is absent. Not called at import time.
    """
    try:
        from spherex_tabular_bandpass import Tabular_Bandpass
    except ImportError as e:
        raise ImportError(
            "fit_hflatline_params needs the optional 'spherex_tabular_bandpass'"
            " package (the full tabulated bandpass). It is not a hard dependency"
            " of ssiaat; install/checkout it to use this helper."
        ) from e
    from ..spherex_table import BandpassTool

    tool = BandpassTool(Tabular_Bandpass(), band=band)
    if iy is not None:
        p = tool.fit_hflatline_at_iy(iy, symmetric=symmetric)
    else:
        mc = center if mu_center is None else mu_center
        p = tool.fit_hflatline(center, mu_center=mc, symmetric=symmetric)
    p = dict(p)
    p.pop("amplitude", None)
    return p
