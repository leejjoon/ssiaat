import numpy as np
from spherex_utils.utils.mosaic_utils import SpectralChannelDefinition
scd = SpectralChannelDefinition()


def tanh_step(a, x1, xx):
    """ smooth transition from x=0 to x=x1.
    Postive a for y~1 at x=0 y~0 at x=x1.
    """
    dx = x1
    yy = (1 + np.tanh(a - xx*a*2/dx)) / 2 # + np.tanh(xx)# *np.tanh(18-x)
    return yy


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
            # tanh_step(-self.a, (mu-self.mu0 + self.dmu)/self.dmu/2)
            # tanh_step(-self.a, (mu-self.mu0 + self.dmu)/self.dmu/2)
            # *
            # tanh_step(self.a, (mu-self.mu0-self.dmu)/self.dmu/2+0.5)
        )

class hflatline:
    def __init__(self, mu_center_left, dmu_left, a_left, *, mu_center_right=None, dmu_right=None, a_right=None):
        self.mu_center_left = mu_center_left # 0 = 0.5 * (mu0 + mu1)
        self.mu_center_right = mu_center_left if mu_center_right is None else mu_center_right
        self.dmu_left = dmu_left
        self.dmu_right = dmu_left if dmu_right is None else dmu_right

        self.a_left = a_left
        self.a_right = a_left if a_left is None else a_right

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


def channel_model(band: int | tuple[int, int], channel: int | tuple[int, int],
                  model: str, a: float):

    match band:
        case int():
            b1 = band
            b2 = band
        case (int(b1), int(b2)):
            pass
        case _:
            raise ValueError()

    match channel:
        case int():
            c1 = scd.wavelength_range(b1, channel)
            c2 = scd.wavelength_range(b2, channel)
        case (int(ch1), int(ch2)):
            c1 = scd.wavelength_range(b1, ch1)
            c2 = scd.wavelength_range(b2, ch2)
        case _:
            raise ValueError()

    w1, w2 = c1.value[0], c2.value[1]

    if model == "cont_left":
        return cont_left(w1, w2, a)
    elif model == "cont_right":
        return cont_right(w1, w2, a)
    elif model == "flattop":
        return hflattop(w1, w2, a)
    else:
        raise KeyError("Unknown Model name:", model)


# def cont_across_channel(band: int, channel_start: int, channel_end: int, model: str, a: float):
#     c1  = scd.wavelength_range(band, channel_start)
#     c2  = scd.wavelength_range(band, channel_end)
#     w1, w2 = c1.value[0], c2.value[1]

#     if model == "cont_left":
#         return cont_left(w1, w2, a)
#     elif model == "cont_right":
#         return cont_right(w1, w2, a)
#     elif model == "step":
#         return hstep(w1, w2, a)
#     else:
#         raise KeyError("Unknown Model name:", model)


def test():
    a = 0.6


    band = 4
    model_defs = [((9, 12), "cont_left"),
                  (10, "flattop"),
                  (11, "flattop"),
                  (12, "flattop"),
                  (13, "flattop"),
                  (14, "flattop"),
                  ((13, 16), "cont_right"),
                  ]

    models = [channel_model(band, ch, model, a) for ch, model in model_defs]

    mu0 = models[0].mu0 - 2*models[0].dmu
    mu1 = models[-1].mu0 + 2*models[-1].dmu

    mu = np.linspace(mu0, mu1, 128)

    fig, axs = plt.subplots(2, 1, num=1, clear=True)
    ax = axs[0]
    for m in models:
        ax.plot(mu, m(mu))

        _mu0 = m.mu0 - m.dmu
        _mu1 = m.mu0 + m.dmu
        ax.axvline(_mu0, ls=":")
        ax.axvline(_mu1, ls=":")


    k = np.array([1, 2, 5, 25, 4, 2, 2])[:, np.newaxis]
    k = np.array([1, 2, 5, 5.5, 4, 2, 2])[:, np.newaxis]
    # k = np.array([1, 1, 1, 1])[:, np.newaxis]
    mu_c = [m.mu0 for m in models]

    ax = axs[1]
    yy = np.array([k1*m(mu) for (k1, m) in zip(k, models)])

    for y1 in yy:
        ax.plot(mu, y1, ":")

    ax.plot(mu, np.sum(yy, axis=0))
    ax.plot(mu_c, k[:len(mu_c)], "o")


def get_h2o_ice():
    h2o_ice = hflatline(3.07, 0.23, 1.3, a_right=0.7)

    return h2o_ice

def get_pah():
    w1 = 3.2481
    w2 = 3.3382
    pah_narrow = hflattop(w1, w2, a=0.7) # aromatic component

    w11 = 3.32
    w31 = 3.52 # we try to make broad component slightly narrower based on the observed spectra.
    pah_broad = hflattop(w11, w31, a=0.9) # aliphatic component

    return pah_narrow, pah_broad

def get_conts(w0, w1, a=0.3):
    w01 = 0.5 * (w0 + w1)
    return cont_left(w0, w01, a), cont_right(w01, w1, a)
