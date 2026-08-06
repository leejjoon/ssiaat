"""Dependency-free analytic bandpass SHAPE, central column (ix=1024) only.

`tabular_bandpass_lite.Tabular_Bandpass_Lite` already gives the central
wavelength at any (band, pixel) with zero dependency on the ~75MB
`spherex_tabular_bandpass` package. This module does the analogous thing
for the *full transmission curve*: `BandpassShapeLite(center, array=band)`
returns a (wavelength, transmission) curve without loading that package.

How: `ssiaat.model.sed.hflatline`'s shape, fit with `mu_center_left` FIXED
to the analytic central wavelength (removes a real optimizer degeneracy
-- see `BandpassTool.fit_hflatline`'s docstring), plus a small additive
`floor0 + floor1*(w - mu_center)` term (the real tabulated bandpass
carries a genuine ~0.01-0.1%-of-peak broadband transmission floor across
the whole band that a single-peak shape can't otherwise represent -- see
the derivation script's module docstring). Each of the seven resulting
quantities (`amplitude`, `dmu_left`, `a_left`, `dmu_right`, `a_right`,
`floor0`, `floor1`) is a low-order Chebyshev polynomial of the central
wavelength, one set of coefficients per band, hardcoded below.

Provenance
----------
Derived by `examples/derive_bandpass_shape_coeffs.py`, 2026-08-06.
Validity domain per band is the trimmed central-column wavelength range
(dichroic-contaminated rows near band edges excluded -- see
`SHAPE_CHEB_DOMAIN`); outside it, `params()`/`__call__` extrapolate the
polynomials and emit a `UserWarning` rather than raising, since the
model's error there is unvalidated. `TYPICAL_RMS` is each band's median
held-out curve RMS (raw transmission units) against the real tabulated
bandpass -- the error budget to expect for any single reconstructed
curve.

Central-column-only by construction: there is no `ix` parameter. The
LVF response varies with detector column (the "smile"), and these
coefficients were fit only along ix=1024.
"""
import warnings

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from .model.sed import hflatline

_SHAPE_PARAM_NAMES = ("amplitude", "dmu_left", "a_left", "dmu_right", "a_right",
                     "floor0", "floor1")
_HFLATLINE_KWARGS = ("dmu_left", "a_left", "dmu_right", "a_right")


class BandpassShapeLite:
    """Analytic (hflatline + floor) stand-in for the tabulated SPHEREx
    bandpass, central column (ix=1024) only. See the module docstring."""

    # band -> param name -> Chebyshev coefficient array. Plain np.array
    # literals (never pickled/repr'd Chebyshev objects) so this stays
    # source/diff/review friendly; reconstructed into Chebyshev objects
    # lazily in __init__.
    SHAPE_CHEB_COEFF = {
        1: {
            'amplitude': np.array([0.08383927241925888, -0.020674875542959256, 0.0022040837884425945, -0.0003051807768598931]),
            'dmu_left': np.array([0.011388927480334486, 0.0019779864540144052, 1.712947933490416e-05, -0.00031251988951175335]),
            'a_left': np.array([1.5063493423310355, -0.006099112756156345, -0.029882911481004165, -0.03600096594483556]),
            'dmu_right': np.array([0.011173978943257739, 0.0034790905389656067, -2.2926306646839547e-05, 0.0004363766064878233]),
            'a_right': np.array([1.2446982866454745, 0.10301706514300892, -0.031259314124948236, 0.039536270207036844]),
            'floor0': np.array([0.0002592417033567769, -7.253837163377787e-05, -5.177531814311854e-06, 7.771524421152826e-06]),
            'floor1': np.array([-0.0006538110658209463, 9.295816399737539e-05, 0.0001375278921384967, 7.984527902861531e-06]),
        },
        2: {
            'amplitude': np.array([0.09055906861234196, -0.020002206077954228, 0.0035671733577591846, -0.0008114621745176395, 0.0010138688390075388, -0.00021707043969222063, -0.0008375983457312688]),
            'dmu_left': np.array([0.0149871706686211, 0.002872721617009503, -5.97275906705776e-05, -0.0007520150138085318, 9.229948119663841e-05, 0.00043385272814599275, -3.723711394620134e-05]),
            'a_left': np.array([1.5864346927038309, 0.16427971553187964, -0.04036185906414694, -0.04632669732808259, 0.0014500078606710315, 0.06339425166251081, -0.027961062749105564]),
            'dmu_right': np.array([0.017469755011678807, 0.004063879553929612, -0.0001176019173277399, 0.0006893756364003922, -0.00038444098645228753, -0.00032994547703756337, 0.0001490249819722165]),
            'a_right': np.array([1.8860862006917867, 0.20623914399525742, -0.030461495726204303, 0.06651954906173399, -0.014757202543960419, -0.0612539439325555, 0.024188710767197707]),
            'floor0': np.array([9.034094475520971e-05, -2.166805876234715e-05, -2.446358031901378e-05, 1.886882230485109e-05, 2.079587053165098e-06, -6.938235621799013e-06, 6.232076233899812e-06]),
            'floor1': np.array([-0.0001622602138764904, -3.424288247074685e-05, 0.00010293866707460163, -4.6489443985797284e-07, 1.0150426355217854e-06, 1.58238326226146e-05, -1.5020461711472156e-05]),
        },
        3: {
            'amplitude': np.array([0.08251721459206175, -0.016858972810991284, 0.0024805511399547363, -0.0003854688569221568, 8.10779219978012e-05, -0.00010470149192465224, -0.0006706290120414272]),
            'dmu_left': np.array([0.022766854855903042, 0.004552021442552092, -0.0005782658494773986, -3.588290492826828e-05, 0.00015016697650743586, 0.0004446312238516585, 0.0002418956260071708]),
            'a_left': np.array([1.5291909935307724, 0.13502353578169946, -0.050211515222734995, 0.014422307109953328, 0.006988115542698708, 0.038310668260488706, 0.0026050307577863]),
            'dmu_right': np.array([0.025126148551929652, 0.005006814561070473, 0.00015869800310467107, 1.216368841875945e-05, -0.00023318853125219786, -0.00040177074998401144, 0.00015216653380819808]),
            'a_right': np.array([1.5112239393202829, 0.10686071487695806, 0.007206261687085005, -0.004025633798716362, -0.012874173226457664, -0.031390212356860216, -0.0027556989479430955]),
            'floor0': np.array([7.502504465986331e-05, -1.7163652886922458e-05, 2.128666618210778e-05, -2.4597256153268017e-07, -1.0304685401615262e-06, -7.692065189899089e-07, -1.4425585326340824e-06]),
            'floor1': np.array([-3.1967298343452076e-05, 0.00012468218557061317, 8.467781959072821e-06, -1.0208524591936206e-06, -2.8838898538416792e-06, -1.326159779623818e-06, -1.6577149844826425e-07]),
        },
        4: {
            'amplitude': np.array([0.042799711035504874, -0.007239833357742334, 0.0005962783868323179, -0.00014435775982336724, -0.0002778930120005303, 5.807123207112569e-05]),
            'dmu_left': np.array([0.04466078918041415, 0.006743315090356755, 0.002832894461720057, -0.0002916261767774885, 9.034935751938749e-05, -0.0006759898649151297]),
            'a_left': np.array([1.5565145039677826, 0.041353092352655156, 0.1252499805322046, -0.008983180000758244, -0.013870398786503379, -0.014915445618752377]),
            'dmu_right': np.array([0.045354421650702074, 0.010101778586320661, -0.0034422136587898304, 0.0008202324218969194, 0.0005917162074930024, 0.0006157714433617351]),
            'a_right': np.array([1.3842878096054605, 0.12492658197991463, -0.09759250173756362, 0.022108849981150654, 0.010677841027525739, 0.024378134283844215]),
            'floor0': np.array([5.41087795489499e-05, -6.010797228994158e-08, 1.1044505173686951e-05, 2.495407536372415e-06, -4.0992006764433455e-06, 3.267885296154056e-06]),
            'floor1': np.array([3.119610209940231e-06, 2.9596697609788237e-05, -8.423885981844897e-08, -3.848823918325653e-06, 1.8302882343976654e-06, -3.1975913652048765e-06]),
        },
        5: {
            'amplitude': np.array([0.09019907214612634, -0.004328706861989034, 0.002472978014005138, -0.0006038771894876279]),
            'dmu_left': np.array([0.014410427893867343, 0.0012016251596857395, -0.00034179313947436615, 0.000571023401977304]),
            'a_left': np.array([0.6161412000460527, 0.059155965279250405, -0.028194724611665195, 0.0274895381299271]),
            'dmu_right': np.array([0.018666808045979956, 0.0005313505370029615, -0.00046620595445059817, -0.00029384860641490384]),
            'a_right': np.array([0.7987174519191421, 0.0016213048465977121, -0.050177359909714246, -0.028194836425235213]),
            'floor0': np.array([-3.9837222463133443e-05, 1.4725731314575645e-05, -1.6085315802131268e-05, -4.5799126083581236e-07]),
            'floor1': np.array([2.3578340955676615e-05, -0.0001061977663038584, 3.210239636845841e-06, 4.6284389123131635e-06]),
        },
        6: {
            'amplitude': np.array([0.12393633161529719, -0.007009151034833564, 0.007232114995411294, -0.00017136474755960037, 0.003971207595542362, 0.0007866371402076308, 0.0009158681124566136]),
            'dmu_left': np.array([0.013489823824972842, 0.0007653845394400779, -0.001148765671758252, -4.791847763798829e-05, -0.00093966259452593, -0.0002577835667469655, 0.00017330436841998632]),
            'a_left': np.array([0.5186453326842347, 0.020156123250395865, -0.06731170323872594, 0.004583821545373315, -0.053051520097041824, -0.006286112247186304, 0.0033308600010715467]),
            'dmu_right': np.array([0.0182349908226654, 0.0005959133409328101, -0.0007408160260054051, -9.573969480102015e-06, 4.7883160958318934e-05, 4.863325635658867e-05, -0.00036294253330004903]),
            'a_right': np.array([0.7769710556053067, -0.015825174472795918, -0.07626067034353015, -0.00839329286217043, -0.00958615818235925, 0.008457028031919198, -0.010257457586550958]),
            'floor0': np.array([-4.554859161612024e-05, -1.1645518096129583e-05, -3.941615954987379e-05, -5.442535092634544e-06, -7.429265224259928e-06, -5.977218775384842e-06, -4.4458516648107554e-06]),
            'floor1': np.array([-1.2482059772829195e-05, -0.0002599462575496264, -2.601041322240849e-05, -2.030967918165225e-05, -6.3467034137498835e-06, -1.2977202962721728e-05, -7.116522841287702e-06]),
        },
    }

    # band -> (wc_min, wc_max): the validated central-column wavelength
    # range each band's polynomials were fit over (dichroic-contaminated
    # rows near band edges already trimmed out).
    SHAPE_CHEB_DOMAIN = {
        1: (0.7384984421704566, 1.1099784291457622),
        2: (1.0896649349147056, 1.6415666291688382),
        3: (1.6216170951886264, 2.3475442903400645),
        4: (2.5012185365184934, 3.803015664866847),
        5: (3.7950936778876567, 4.40986974142866),
        6: (4.397894597481459, 4.9917419727267385),
    }

    # band -> (w_start, dw, n): the real package's fixed per-band
    # wavelength grid, reproduced exactly so `grid="native"` output lands
    # on the same abscissa as the real tabulated bandpass.
    NATIVE_GRID = {
        1: (0.684398, 0.0020000000000000018, 274),
        2: (1.013944, 0.0029999999999998916, 269),
        3: (1.496867, 0.0040000000000000036, 297),
        4: (2.221877, 0.0040000000000000036, 525),
        5: (3.688934, 0.0030000000000001137, 304),
        6: (4.26602, 0.0039999999999995595, 208),
    }

    FIT_DEGREE = {1: 3, 2: 6, 3: 6, 4: 5, 5: 3, 6: 6}

    # band -> median held-out curve RMS (raw transmission units) of the
    # polynomial reconstruction against the real tabulated bandpass --
    # the error budget to expect from this module's own output (not the
    # per-row direct-fit RMS, which is a tighter lower bound).
    TYPICAL_RMS = {
        1: 0.0004943949224051526,
        2: 0.0002507405025509444,
        3: 0.00037129333881309815,
        4: 0.0001675205538054391,
        5: 0.0003863045149838027,
        6: 0.0004750970656028969,
    }

    def __init__(self):
        self._polys = {
            band: {name: Chebyshev(coef, domain=self.SHAPE_CHEB_DOMAIN[band])
                  for name, coef in params.items()}
            for band, params in self.SHAPE_CHEB_COEFF.items()
        }

    def _check_array(self, array):
        if array not in self._polys:
            raise ValueError(f"array={array!r} not supported; must be one of"
                             f" {sorted(self._polys)}")

    def params(self, center, array=2):
        """Evaluate all seven fitted quantities at `center` (microns).

        Returns a dict with keys ``mu_center_left``, ``dmu_left``,
        ``a_left``, ``dmu_right``, ``a_right`` (hflatline's constructor
        kwargs verbatim, same convention as `BandpassTool.fit_hflatline`'s
        return value) plus ``amplitude``, ``floor0``, ``floor1``, none of
        which are hflatline parameters:
        ``amplitude * hflatline(mu_center_left, dmu_left, a_left,
        dmu_right=dmu_right, a_right=a_right)(w) + floor0 +
        floor1*(w - mu_center_left)`` reconstructs the curve -- exactly
        what `__call__` does.

        Outside `SHAPE_CHEB_DOMAIN[array]`, extrapolates the polynomials
        and emits a `UserWarning` (the model's error there is
        unvalidated -- see the module docstring) rather than raising.
        """
        self._check_array(array)
        lo, hi = self.SHAPE_CHEB_DOMAIN[array]
        if not (lo <= center <= hi):
            warnings.warn(
                f"center={center!r} is outside band {array}'s validated"
                f" range [{lo!r}, {hi!r}]; extrapolating polynomials fit"
                " only inside that range -- treat the result as unvalidated.",
                stacklevel=2)
        polys = self._polys[array]
        out = {name: float(polys[name](center)) for name in _SHAPE_PARAM_NAMES}
        out["mu_center_left"] = float(center)
        return out

    def __call__(self, center, array=2, wvl=None, grid="native"):
        """The reconstructed (wavelength, transmission) curve at `center`.

        Parameters
        ----------
        center : float
            Central wavelength, microns, central column (ix=1024).
        array : int
            SPHEREx band, 1-6.
        wvl : array_like, optional
            Evaluate on this caller-supplied wavelength grid instead of
            `grid`.
        grid : {"native", "local"}
            "native" (default): the real package's fixed per-band grid
            (`NATIVE_GRID`) -- directly comparable to real
            `Tabular_Bandpass` output, at the cost of mostly-negligible
            values far from `center` for the narrower bands. "local": a
            compact window sized to the fitted profile,
            `center +/- 6*max(dmu_left, dmu_right)`, spaced at the same
            `dw` as the native grid.
        """
        self._check_array(array)
        p = self.params(center, array)
        shape = hflatline(p["mu_center_left"], p["dmu_left"], p["a_left"],
                          dmu_right=p["dmu_right"], a_right=p["a_right"])

        if wvl is not None:
            w = np.asarray(wvl, dtype=float)
        elif grid == "native":
            w_start, dw, n = self.NATIVE_GRID[array]
            w = w_start + dw * np.arange(n)
        elif grid == "local":
            _, dw, _ = self.NATIVE_GRID[array]
            halfwidth = 6 * max(p["dmu_left"], p["dmu_right"])
            n = max(int(round(2 * halfwidth / dw)), 2)
            w = center - halfwidth + dw * np.arange(n)
        else:
            raise ValueError(f"grid={grid!r}; must be 'native' or 'local'")

        floor = p["floor0"] + p["floor1"] * (w - p["mu_center_left"])
        t = p["amplitude"] * shape(w) + floor
        return w, t

    def at_pixel(self, iy, array=2):
        """`params()` at detector row `iy`'s own analytic central
        wavelength (looked up via `Tabular_Bandpass_Lite`), instead of a
        caller-supplied wavelength. Central column (ix=1024) only -- the
        only column these coefficients are valid for -- hence no `ix`
        argument. For the curve rather than just the parameters, look up
        the center yourself and call `self(center, array=array)`."""
        from .tabular_bandpass_lite import Tabular_Bandpass_Lite
        center, _ = Tabular_Bandpass_Lite()(1024, iy, array=array,
                                            central_bandpass_only=True)
        return self.params(float(np.asarray(center)), array)
