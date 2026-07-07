"""End-to-end fitting demo on the eso_244 data set.

Moved out of ssiaat.spherex_table (where it lived with a broken bare
`sed` import). Expects eso_244_template.fits and eso_244_b{3,4,5}.parquet
in the working directory; `main()` additionally needs pyregion and
matplotlib.
"""
import numpy as np
import pandas as pd

from ssiaat.model.sed import hflattop, cont_left, cont_right
from ssiaat.spherex_table import Model, SsiaatConverter


def get_test_model():
    z = 0.023
    u_narrow = 3.29315 * (1+z)
    du_narrow = 0.04505 * (1+z)
    pah_narrow = hflattop(u_narrow - du_narrow, u_narrow + du_narrow, a=0.7)

    u_broad = 3.420 * (1+z)
    du_broad = 0.100 * (1+z)
    pah_broad = hflattop(u_broad - du_broad, u_broad + du_broad, a=0.9)

    models = [pah_narrow, pah_broad]
    cont_models = [cont_left(2.6, 3.3, 0.3), cont_right(3.3, 4.0, 0.3)]

    spectral_model = Model(models, cont_models)

    return spectral_model


def test_save():
    root = "eso_244"
    template_name = f"{root}_template.fits"
    ssiaat_converter = SsiaatConverter.from_file(template_name)

    from pathlib import Path
    datadir = Path(".")

    fnlist = [str(datadir / f"{root}_b{band}.parquet") for band in [3, 4, 5]]

    stable_ = ssiaat_converter.read_stable(*fnlist)
    stable = stable_.query("(2.6 < wvl) and (wvl < 4.0)")

    stable.to_parquet("a.parquet")


def test_load():
    stable = pd.read_parquet("a.parquet")
    print(stable.spectral.converter.tmpl_shape)


def main():
    stable = pd.read_parquet("a.parquet")

    converter = stable.spectral.converter

    im = stable.spectral.make_simple_image(3.1, 4.0)
    # fits.PrimaryHDU(data=im).writeto("a.fits", overwrite=True)

    spectral_model = get_test_model()

    fitted_model = spectral_model.least_square_fit(stable)

    print(fitted_model.C[0])
    itable = fitted_model.contC[1]
    im = itable.itable.to_image()
    # fits.PrimaryHDU(data=im).writeto("b.fits", overwrite=True)

    # spatial filtering
    sreg = "image;ellipse(31.403764,28.577416,4.3068155,8.0752791,353.88636)"
    import pyregion
    reg = pyregion.parse(sreg)
    msk = reg.get_mask(shape=(61, 61))

    s = stable.spectral.filter_with_image_mask(msk)

    param_i = 0
    imsk = converter.image_to_itable(msk)

    ss_contsub = fitted_model.cont_sub(s["wvl"], s["image"])

    import matplotlib.pyplot as plt
    plt.scatter(s["wvl"],
                fitted_model.cont_sub_n_norm(s["wvl"], s["image"], param_i),
                s=1)

    xx = np.linspace(2.6, 4.0, 100)

    c0 = fitted_model.C[0]
    c1 = fitted_model.C[1]
    median_c1_c0 = np.nanmedian(c1[imsk] / c0[imsk])

    cc0 = spectral_model.models[0](xx)
    cc1 = median_c1_c0 * spectral_model.models[1](xx)

    plt.plot(xx,  cc0 + cc1, "-", lw=3, alpha=0.5)
    plt.plot(xx, cc0)
    plt.plot(xx, cc1)
    plt.show()


if __name__ == '__main__':
    # test_save(); test_load()
    main()
