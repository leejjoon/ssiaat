
# %%
import pandas as pd
import numpy as np
# from numpy import NDData
import numpy.typing as npt
from astropy.io import fits




# %%
# template_file = "template_gal_cyg_x.fits"
#tmpl = fits.open("template_gal_cyg_x.fits")
## wcs_tmpl = WCS(tmpl[0].header)
#tmpl_shape = tmpl[0].data.shape

#tmpl_ind = np.sum(np.indices(tmpl_shape) * np.array([tmpl_shape[-1], 1]).reshape((2, 1, 1)),
#                  axis=0, dtype="int32")

def get_test_model():
    from sed import hflattop, cont_left, cont_right
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

# %%
import astropy.units as u
from ssiaat.wcs_helper import get_wcs
from ssiaat.spherex_table import TemplateHeaderCards


ra = 24.1770616 * u.deg
dec = 15.7869095 * u.deg
side1 = 0.9 * u.deg

tmpl_wcs = get_wcs(ra, dec, side1)
# header_cards = TemplateHeaderCards.from_header(tmpl_wcs.to_header())
tmpl_header = tmpl_wcs.to_header()
tmpl_header["NAXIS"] = 2
tmpl_header["NAXIS2"], tmpl_header["NAXIS1"] = tmpl_wcs.array_shape


# %%

from pathlib import Path

root = "ngc628"
rootdir = Path(f"{root}")

# %%
# this is to add metadata to the parquet files.

if False:
    for band in [1, 2, 3, 4, 5, 6]:
        df = pd.read_parquet(f"{root}_raw/{root}_b{band}.parquet")
        TemplateHeaderCards.update_dataframe_from_header(df, tmpl_header)
        df.to_parquet(rootdir / f"{root}_b{band}.parquet")

# %%

from ssiaat.spherex_table import read_stable

root = "ngc628"
# ssiaat_converter = SsiaatConverter.from_file(template_name)

# datadir = Path(".")

fnlist = [str(rootdir / f"{root}_b{band}.parquet") for band in [3, 4, 5]]

stable_all = read_stable(fnlist)

fnlist = [str(rootdir / f"{root}_b{band}.parquet") for band in [1, 2, 3, 4, 5, 6]]

stable_all_bands = read_stable(fnlist)


# %%

# this is the fit range
stable = stable_all.query("(2.6 < wvl) and (wvl < 3.7)")

# %%
import matplotlib.pyplot as plt

if True:
    print(stable.spectral.converter.tmpl_shape)

im = stable.spectral.make_simple_image(3.1, 3.8)

im_slice = slice(190, 330), slice(190, 330)

fig, ax = plt.subplots(num=1, clear=True)
ax.imshow(im[im_slice], origin="lower", vmin=0, vmax=1)

# %%
import pyregion

sreg = """image
circle(264.32301,261.90043,34.104124)
-circle(288.71035,243.32729,4.2553258)
-circle(251.40256,269.6527,4.1554991)
-circle(242.19675,279.98905,2.8586342)
-circle(265.61506,262.70796,6.6381761)
"""
sreg_center = """image
circle(265.61506,262.70796,6.6381761)
"""

reg_msk = pyregion.parse(sreg).get_mask(shape=im.shape)
reg_center_msk = pyregion.parse(sreg_center).get_mask(shape=im.shape)

msk = (im > 0.4) & reg_msk
# %%

fig, ax = plt.subplots(num=2, clear=True)
ax.imshow(msk[im_slice], origin="lower", vmin=0, vmax=1)

# %%

# s_all = stable_all.spectral.filter_with_image_mask(msk)
s = stable.spectral.filter_with_image_mask(msk)

# imsk = stable.spectral.converter.image_to_itable(msk)
norm = stable.spectral.converter.image_to_itable(im, mask=msk)#[imsk]

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(num=3, clear=True)
ax.scatter(s["wvl"],
           s["image"] / norm,
           s=1)

# %%
from ssiaat.spherex_table import Model

from sed import hflattop, cont_left, cont_right

def get_cont_model(cw0, cw1):
    cw_middle = 0.5 * (cw0 + cw1)
    cont_models = [cont_left(cw0, cw_middle, 0.3), cont_right(cw_middle, cw1, 0.3)]

    spectral_model = Model([], cont_models)

    return spectral_model

cw0 = 2.5
cw1 = 3.8
spectral_model = get_cont_model(cw0, cw1)

fitted_model = spectral_model.least_square_fit(stable)

# %%

cont0 = fitted_model.contC[0].itable.to_image()
cont1 = fitted_model.contC[1].itable.to_image()

fig, axs = plt.subplots(1, 2, num=4, clear=True)
kw = dict(vmin=0, vmax=1, origin="lower")
axs[0].imshow(cont0[im_slice], **kw)
axs[1].imshow(cont1[im_slice], **kw)

# %%

# sall_contsub = fitted_model.cont_sub(s_all["wvl"], s_all["image"])
# s2 = s.copy()
stable_contsub = fitted_model.cont_sub(stable["wvl"], stable["image"])
stable.loc[:, "contsub"] = stable_contsub


# w1, w2 = 3.1, 3.8
# column = "contsub"
# dfc = stable.query(f"({w1} < wvl) and (wvl < {w2})")
# _ = dfc.groupby(by=dfc.index)[column].mean()
# im_contsub = stable.spectral.converter.itable_to_image(_)

# # print(fitted_model.C[0])
# # itable = fitted_model.contC[1]

# %%

s_all = stable.spectral.filter_with_image_mask(msk)
# norm2 = stable.spectral.converter.image_to_itable(im_contsub)[imsk]

fig, ax = plt.subplots(num=5, clear=True)
ax.scatter(s_all["wvl"], s_all["contsub"],
           # s_all["contsub"],
           s=1)

# %%

def get_pah_model(z=0., cw0=2.6, cw1=4.0):
    from sed import hflattop, cont_left, cont_right
    u_narrow = 3.29315 * (1+z)
    du_narrow = 0.04505 * (1+z)
    pah_narrow = hflattop(u_narrow - du_narrow, u_narrow + du_narrow, a=0.7)

    u_broad = 3.420 * (1+z)
    du_broad = 0.100 * (1+z)
    pah_broad = hflattop(u_broad - du_broad, u_broad + du_broad, a=0.9)

    models = [pah_narrow, pah_broad]

    cont_models_ = get_cont_model(cw0, cw1)

    cont_models = cont_models_.cont_models
    # cont_models = [cont_left(2.6, 3.3, 0.3), cont_right(3.3, 4.0, 0.3)]

    spectral_model = Model(models, cont_models)

    return spectral_model

xx = np.linspace(2.6, 4.0, 128)
pah_model = get_pah_model(z=0)
for m in pah_model.models:
    ax.plot(xx, 0.1*m(xx))

# %%

fitted_pah = pah_model.least_square_fit(stable)

# %%

# itable = fitted_model.C[0]
pah0 = fitted_pah.C[0].itable.to_image()
pah1 = fitted_pah.C[1].itable.to_image()
cont0 = fitted_pah.contC[0].itable.to_image()
cont1 = fitted_pah.contC[1].itable.to_image()

fig, axs = plt.subplots(1, 4, num=6, clear=True)
for ax, a, vmax in zip(axs, [cont0, pah0, pah1, cont1], [1, 0.5, 0.1, 1]):
    kw1 = dict(**kw)
    kw1["vmax"] = vmax
    ax.imshow(a[im_slice], **kw1)

# %%

cont_at_pah0 = ((cont0 + pah_model.cont_models[0](3.29)) +
                (cont1 + pah_model.cont_models[1](3.29)))
# cont_at_pah1 = ((cont0 + pah_model.cont_models[0](3.4)) +
#                 (cont1 + pah_model.cont_models[1](3.4)))

# cont_at_3 = ((cont0 + pah_model.cont_models[0](3.)) +
#              (cont1 + pah_model.cont_models[1](3.3)))

# %%

# pah_msk = (pah0 > 0.05) &  ~reg_center_msk
pah_msk = (pah0 > 0.05) & ((pah0 > 0.15 * cont_at_pah0)) & reg_msk

fig, ax = plt.subplots(1, 1, num=7, clear=True)
ax.imshow(pah_msk[im_slice])


# %% 
s = stable.spectral.filter_with_image_mask(pah_msk)
param_i = 0

fig, ax = plt.subplots(num=8, clear=True)
ax.scatter(s["wvl"],
           fitted_pah.cont_sub_n_norm(s["wvl"], s["image"], param_i),
           s=1)

median_c1_c0 = float(np.nanmedian(pah1[pah_msk] / pah0[pah_msk]))

y0 = pah_model.models[0](xx)
y1 = median_c1_c0 * pah_model.models[1](xx)

ax.plot(xx,  y0 + y1,
        "-", lw=3, alpha=0.5)
ax.plot(xx, y0)
ax.plot(xx, y1)

# %%

# stable_all_bands

s0 = stable_all_bands.spectral.filter_with_image_mask(pah_msk)
param_i = 0

fig, ax = plt.subplots(num=9, clear=True)
for b, s in s0.groupby("band"):
    ax.scatter(s["wvl"],
               fitted_pah.cont_sub_n_norm(s["wvl"], s["image"], param_i),
               s=1)
