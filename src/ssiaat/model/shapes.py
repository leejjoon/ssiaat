"""Profile-shape primitives for line/continuum models.

Smooth top-hat-like shapes built from :func:`tanh_step`, plus the two
continuum-ramp helpers. These are pure functions of wavelength (each object's
``__call__(mu) -> ndarray``) with no line-specific constants baked in -- the
line inventory itself lives in :mod:`ssiaat.model.lines`.

Historically these lived in :mod:`ssiaat.model.sed`; ``sed`` now re-exports them
for backward compatibility. Import from here (or from ``ssiaat.model.shapes``)
in new code.
"""
import numpy as np


def tanh_step(a, x1, xx):
    """ smooth transition from x=0 to x=x1.
    Postive a for y~1 at x=0 y~0 at x=x1.
    """
    dx = x1
    yy = (1 + np.tanh(a - xx*a*2/dx)) / 2 # + np.tanh(xx)# *np.tanh(18-x)
    return yy


class const:
    def __init__(self):
        """
        step-like function. The transition begins neart 75% point until 125% point.
        """
        pass
        # self.a = a

    def __call__(self, mu):
        # return tanh_step(self.a, 1, (mu-self.mu0)/self.dmu/2+0.5)
        return np.ones_like(mu)


class cont_left:
    def __init__(self, mu0, mu1, a):
        """
        step-like function. The transition begins neart 75% point until 125% point.
        """
        mu0, mu1 = sorted([mu0, mu1])
        dmu = (mu1 - mu0)/2.
        self.mu0 = 0.5 * (mu0 + mu1)
        self.dmu = dmu
        self.a = a

    def __call__(self, mu):
        # return tanh_step(self.a, 1, (mu-self.mu0)/self.dmu/2+0.5)
        return tanh_step(self.a, self.dmu, (mu-self.mu0) - 0.5*self.dmu)


class cont_right:
    def __init__(self, mu0, mu1, a):
        """
        step-like function. The transition begins neart -25% point until 25% point.

        """
        mu0, mu1 = sorted([mu0, mu1])
        dmu = (mu1 - mu0)/2.
        self.mu0 = 0.5 * (mu0 + mu1)
        self.dmu = dmu
        self.a = a

    def __call__(self, mu):
        # return tanh_step(self.a, 1, (mu-self.mu0)/self.dmu/2+0.5)
        return tanh_step(-self.a, self.dmu, (mu-self.mu0 + 1.5*self.dmu))


class hflattop:
    # This is derecated. Use hflatline instead
    def __init__(self, mu0, mu1, a):
        # mu_mean = np.mean(mu0, mu1)
        mu0, mu1 = sorted([mu0, mu1])
        dmu = (mu1 - mu0)/2.
        self.mu0 = 0.5 * (mu0 + mu1)
        self.dmu = dmu

        # mu_mean = np.mean(mu0, mu1)
        self.a = a
        self._left = cont_left(mu0, mu1, a)
        self._right = cont_right(mu0, mu1, a)

    def __call__(self, mu):
        return (
            self._left(mu) * self._right(mu)
        )

    @property
    def area(self):
        r"""Analytic integral of the top-hat: ``(mu1-mu0)/(1 - e^{-8a})``.

        The profile is a product of two facing sigmoids
        ``S(k(dmu-u)) S(k(dmu+u))`` (``S(x)=(1+tanh x)/2``, ``k=2a/dmu``); the
        substitution ``p = e^{2(k u - 2a)}`` reduces the integral to
        ``(1/2) ln(m)/(m-1)`` with ``m = e^{-8a}``, giving ``2*dmu/(1-e^{-8a})``.
        (Limits: ``a -> inf`` -> box width ``mu1-mu0``; ``a -> 0`` -> infinite.)
        """
        return 2.0 * self.dmu / (1.0 - np.exp(-8.0 * self.a))


class hflatline:
    def __init__(self, mu_center_left, dmu_left, a_left, *, dmu_right=None, a_right=None):
        self.mu_center_left = mu_center_left # 0 = 0.5 * (mu0 + mu1)
        self.mu_center_right = mu_center_left
        self.dmu_left = dmu_left
        self.dmu_right = dmu_left if dmu_right is None else dmu_right

        self.a_left = a_left
        self.a_right = a_left if a_right is None else a_right

        # Note that cont_right/left is reversed. cont_right is left side of the
        # liine, for example.
        self._left = cont_right(self.mu_center_left - self.dmu_left,
                                self.mu_center_left + self.dmu_left,
                                self.a_left)
        self._right = cont_left(self.mu_center_right - self.dmu_right,
                                self.mu_center_right + self.dmu_right,
                                self.a_right)

    def __call__(self, mu):
        return (
            self._left(mu) * self._right(mu)
        )

    @property
    def area(self):
        r"""Analytic integral for the SYMMETRIC case (``a_left == a_right`` and
        ``dmu_left == dmu_right``): ``2*dmu/(1 - e^{-8a})`` (same closed form as
        :class:`hflattop`). The asymmetric case (different edge rates) has no
        clean elementary closed form -- raises ``NotImplementedError`` so callers
        fall back to numeric integration.
        """
        if self.dmu_left == self.dmu_right and self.a_left == self.a_right:
            return 2.0 * self.dmu_left / (1.0 - np.exp(-8.0 * self.a_left))
        raise NotImplementedError(
            "hflatline.area has no clean closed form for asymmetric edges"
            " (a_left != a_right or dmu_left != dmu_right); integrate numerically")


class gauss:
    """Gaussian line profile, unit peak at ``center``.

    ``gauss(center, sigma)(mu) = exp(-0.5*((mu-center)/sigma)**2)`` -- peak 1 at
    ``center``, so a fitted coefficient is the line's peak amplitude. Unlike the
    tanh top-hats (which model an unresolved line broadened by the instrument
    bandpass), this is for a spectrally *resolved* line whose own profile is
    peaked. FWHM = ``2*sqrt(2*ln 2)*sigma`` ~= ``2.3548*sigma``.
    """
    def __init__(self, center, sigma):
        self.center = center
        self.sigma = sigma

    def __call__(self, mu):
        return np.exp(-0.5 * ((np.asarray(mu, dtype=float) - self.center)
                              / self.sigma) ** 2)

    @property
    def area(self):
        """Analytic integral of the unit-peak Gaussian: ``sigma*sqrt(2*pi)``."""
        return self.sigma * np.sqrt(2.0 * np.pi)


def get_conts(w0, w1, a=0.3):
    w01 = 0.5 * (w0 + w1)
    return cont_left(w0, w01, a), cont_right(w01, w1, a)


def get_cont_model(cw0, cw1, a=0.3):
    """Two continuum ramps (low, high) spanning ``[cw0, cw1]``, split at the
    window midpoint -- the standard continuum baseline for a line ``Model``."""
    cw_middle = 0.5 * (cw0 + cw1)
    cont_models = [cont_left(cw0, cw_middle, a), cont_right(cw_middle, cw1, a)]
    return cont_models
