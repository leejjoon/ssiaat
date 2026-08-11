"""Line-model registry: the single inventory of SPHEREx line/feature models.

Two kinds of line model share one registry:

* ``kind="bandpass"`` -- an *unresolved* emission line whose observed profile IS
  the SPHEREx instrument bandpass at its wavelength. Built by :func:`bandpass_line`
  from :class:`ssiaat.bandpass_shape_lite.BandpassShapeLite` (band-specific,
  center-specific), unit-peak normalized so the fitted coefficient is the line's
  peak amplitude in image units. This supersedes the four near-identical
  ``_bandpass_line`` copies that used to live in ``work_hips/model_*.py``.

* ``kind="shape"`` -- a legacy hand-tuned analytic shape (a
  :class:`~ssiaat.model.shapes.hflattop`/:class:`~ssiaat.model.shapes.hflatline`),
  including a broad feature (PAH, H2O-ice) that is *not* an unresolved line. These
  reproduce the old ``sed.get_*`` getters bit-for-bit; the ad-hoc ``d_shift`` they
  baked in is preserved verbatim inside each factory (a documented ``prior_offset``
  in spirit) so existing maps are unchanged.

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
from .shapes import get_cont_model, hflattop, hflatline

# One shared bandpass reconstruction (central column, all bands).
_BP = BandpassShapeLite()


def bandpass_line(rest_wavelength, band, *, prior_offset=0.0, halfwidth=0.08,
                  normalize=True):
    """Callable line profile = the SPHEREx instrument bandpass centred at
    ``rest_wavelength + prior_offset``, unit-peak normalised (so the fitted
    coefficient is the line's peak amplitude in image units).

    The unit-peak scale is fixed once from a dense local grid of half-width
    ``halfwidth`` (um), so the returned profile is independent of the caller's
    ``wvl`` sampling. This is the single factory the per-feature ``work_hips``
    models used to each re-implement (only ``halfwidth`` differed: 0.06 for the
    band-1 lines, 0.08 for band 2, 0.10 for band 3).
    """
    center = rest_wavelength + prior_offset
    peak = 1.0
    if normalize:
        wfine = np.linspace(center - halfwidth, center + halfwidth, 2001)
        _, tf = _BP(center, array=band, wvl=wfine)
        peak = float(tf.max())

    def line(wvl):
        _, t = _BP(center, array=band, wvl=np.asarray(wvl, dtype=float))
        return t / peak

    line.__name__ = f"bp_line_b{band}_{rest_wavelength}"
    return line


# --- behavior-preserving legacy analytic shapes (verbatim old getter bodies) ---

def _br_a_shape():
    d_shift2 = 0.012 - 0.002        # legacy: "wavelength solution seem to be off"
    a2 = 0.75 + 0.8
    return hflattop(4.0372 - d_shift2, 4.0724 - d_shift2, a=a2)


def _pa_b_shape():
    d_shift2 = 0.004
    a2 = 0.75 + 0.8
    return hflattop(1.2693 - d_shift2, 1.2988 - d_shift2, a=a2)


def _br_b_shape():
    return hflattop(2.605, 2.647, a=1.5)        # center ~2.626 um (band 4)


def _pah_narrow_shape():
    return hflattop(3.2481, 3.3382, a=0.7)      # aromatic component


def _pah_broad_shape():
    return hflattop(3.32, 3.52, a=0.9)          # aliphatic component


def _h2o_ice_shape():
    return hflatline(3.07, 0.23, 1.3, a_right=0.7)   # broad absorption


def _h2_46957_shape():
    mu_center = 4.6947
    dmu = 0.017
    a2 = 0.75 + 0.8
    d_shift2 = 0.012 - 0.002
    return hflattop(mu_center - dmu - d_shift2, mu_center + dmu - d_shift2, a=a2)


@dataclass(frozen=True)
class LineModel:
    """One registered line/feature model.

    ``rest_wavelength`` is the laboratory (vacuum) center in microns -- the
    quantity a ``kind="bandpass"`` profile is built at (plus ``prior_offset``,
    the explicit replacement for the old magic ``d_shift``). For ``kind="shape"``
    it is metadata only; the actual callable comes from ``shape_factory``.
    ``blend_group`` tags lines whose individual amplitudes are only trustworthy
    as a sum (design-matrix collinear), e.g. He I / Pa-gamma.
    """
    name: str
    band: int
    rest_wavelength: float
    prior_offset: float = 0.0
    kind: str = "bandpass"          # "bandpass" | "shape"
    halfwidth: float = 0.08
    blend_group: Optional[str] = None
    notes: str = ""
    shape_factory: Optional[Callable[[], Callable]] = field(default=None, repr=False)

    @property
    def center(self):
        return self.rest_wavelength + self.prior_offset

    def profile(self, *, shift=0.0, band=None):
        """Bare ``wvl -> ndarray`` callable for this line."""
        if self.kind == "shape":
            return self.shape_factory()
        b = self.band if band is None else band
        return bandpass_line(self.rest_wavelength, b,
                             prior_offset=self.prior_offset - shift,
                             halfwidth=self.halfwidth)

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

_SHAPE = [
    LineModel("br_a", 5, 4.0523, kind="shape", shape_factory=_br_a_shape,
              notes="H I Br-alpha (band 5); legacy hflattop with d_shift"),
    LineModel("pa_b", 2, 1.2822, kind="shape", shape_factory=_pa_b_shape,
              notes="H I Pa-beta (band 2); legacy hflattop. cf bandpass 'pa_beta'"),
    LineModel("br_b", 4, 2.6259, kind="shape", shape_factory=_br_b_shape,
              notes="H I Br-beta (band 4); legacy hflattop"),
    LineModel("pah_narrow", 4, 3.2941, kind="shape", shape_factory=_pah_narrow_shape,
              notes="3.3 um PAH aromatic component (broad, not an unresolved line)"),
    LineModel("pah_broad", 4, 3.42, kind="shape", shape_factory=_pah_broad_shape,
              notes="3.4 um PAH aliphatic component (broad)"),
    LineModel("h2o_ice", 4, 3.07, kind="shape", shape_factory=_h2o_ice_shape,
              notes="H2O-ice absorption ~3.05 um (broad; coeff comes out negative)"),
    LineModel("h2_46957", 6, 4.6947, kind="shape", shape_factory=_h2_46957_shape,
              notes="H2 0-0 S(9) 4.6947 um (band 6); shocked/hot molecular gas"),
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


def get_line(name, *, shift=0.0, band=None):
    """Bare ``wvl -> ndarray`` callable for the registered line ``name``."""
    try:
        lm = LINES[name]
    except KeyError:
        raise KeyError(f"unknown line {name!r}; available: {sorted(LINES)}") from None
    return lm.profile(shift=shift, band=band)


def feature_spec(name, *, shift=0.0, band=None):
    """``dict(lines={out_name: callable}, window=(w0, w1))`` for a feature --
    the shape the ``work_hips`` drivers expect from their ``FEATURES`` registry."""
    feat = FEATURES[name]
    lines = {out: get_line(key, shift=shift, band=band)
             for out, key in feat.components}
    return dict(lines=lines, window=feat.window)


def get_line_model(feature=None, *, lines=None, window=None, shift=0.0,
                   band=None, cont_a=0.3):
    """A fittable :class:`ssiaat.model.fitting.Model` = the feature's line
    components + two continuum ramps spanning ``window`` (default: the feature's
    own window). Pass either ``feature`` (a key of :data:`FEATURES`) or an
    explicit ``lines`` dict + ``window``."""
    if feature is not None:
        feat = FEATURES[feature]
        comp = {out: get_line(key, shift=shift, band=band)
                for out, key in feat.components}
        win = feat.window if window is None else window
    else:
        if lines is None or window is None:
            raise ValueError("pass either feature=, or both lines= and window=")
        comp = lines
        win = window
    conts = dict(zip(("cont_lo", "cont_hi"), get_cont_model(*win, a=cont_a)))
    return Model(comp, conts)
