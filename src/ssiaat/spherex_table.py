"""
SSIAAT Table and Image handling.

This module provides specialized classes for handling SPHEREx spectral data in tabular 
and image formats. It integrates closely with NumPy and Pandas using modern 
extension patterns.

Core Components:
---------------
1. Image (np.ndarray subclass):
   A 2D NumPy array that carries an attached `_ssiaat_converter`.
   It behaves exactly like a NumPy array (slicing, math, etc.) but can be 
   converted back to a table format using `.to_itable()`.

2. SpectralTable (pd.DataFrame accessor: .spectral):
   Adds spectral analysis methods to standard Pandas DataFrames.
   Example: `df.spectral.make_simple_image(3.1, 4.0)`

3. ImageTable (pd.Series accessor: .itable):
   Adds image-table mapping methods to standard Pandas Series (where the index 
   represents pixel indices).

4. SsiaatConverter:
   The core bridge between spatial (Image) and tabular (DataFrame/Series) formats.
   It handles reprojection and indexing.

Usage Example:
--------------
    converter = SsiaatConverter("template.fits")
    
    # Load data into a standard DataFrame with spectral powers
    df = converter.read_stable("data.parquet")
    
    # Use the .spectral accessor to create an Image object
    img = df.spectral.make_simple_image(3.2, 3.4)
    
    # 'img' is an ndarray, so you can plot it or do math directly
    import matplotlib.pyplot as plt
    plt.imshow(img)
    img_multiplied = img * 2
    
    # Metadata is preserved during slicing
    sliced_img = img[10:20, 10:20]
    print(sliced_img._ssiaat_converter) 

"""

# %%
import pandas as pd
import numpy as np
# from numpy import NDData
import numpy.typing as npt
from astropy.io import fits


# %%
from pandas.api.types import is_integer_dtype

def check_index_stable(df):
    """
    we want the index of the given dataframe is of the kind we want, i.e., integer
    with duplicates.
    """
    return is_integer_dtype(df.index.dtype) and df.index.has_duplicates

def check_index_itable(df):
    """
    we want the index of the given dataframe is of the kind we want, i.e., integer
    with NO duplicates.
    """
    return is_integer_dtype(df.index.dtype) and not df.index.has_duplicates


@pd.api.extensions.register_dataframe_accessor("spectral")
class SpectralTable:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def converter(self):
        return getattr(self._obj, "_ssiaat_converter", None)

    def make_simple_image(self, w1, w2):
        dfc = self._obj.query(f"({w1} < wvl) and (wvl < {w2})")
        s = dfc.groupby(by=dfc.index)["image"].mean()
        # Propagate converter to the resulting series
        s._ssiaat_converter = self.converter
        return self.converter.itable_to_image(s)

    def filter_with_image_mask(self, msk):
        itable = self.converter.image_to_itable(msk)
        return self._obj.loc[itable]

@pd.api.extensions.register_series_accessor("itable")
class ImageTable:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def converter(self):
        return getattr(self._obj, "_ssiaat_converter", None)


class Image(np.ndarray):
    def __new__(cls, input_array, ssiaat_converter=None):
        # We cast the input to an ndarray and then into our subclass view
        obj = np.asarray(input_array).view(cls)
        # Add the custom metadata
        obj._ssiaat_converter = ssiaat_converter
        return obj

    def __array_finalize__(self, obj):
        # Called when:
        # 1. Explicitly created: Image(...) -> obj is None
        # 2. View casting: arr.view(Image) -> obj is arr
        # 3. Slicing: img[1:2] -> obj is img
        if obj is None: return
        self._ssiaat_converter = getattr(obj, '_ssiaat_converter', None)

    def to_itable(self):
        """Converts the image back to an ImageTable using its converter."""
        if self._ssiaat_converter is None:
            raise ValueError("No ssiaat_converter attached to this Image.")
        return self._ssiaat_converter.image_to_itable(self)


class SsiaatConverter:
    def __init__(self, template_file):
        self.tmpl = fits.open(template_file)
        # wcs_tmpl = WCS(tmpl[0].header)
        self.tmpl_shape = self.tmpl[0].data.shape

        # 2d array
        self.tmpl_ind = np.sum(np.indices(self.tmpl_shape)
                               * np.array([self.tmpl_shape[-1], 1]).reshape((2, 1, 1)),
                               axis=0, dtype="int32")
        self.tmpl_ind_flat = np.ravel(self.tmpl_ind)

    def itable_to_image(self, itable: pd.Series, ignore_index_name=False):
        # itable should be a series whose index is a subset of tmpl_ind.
        im_ = itable.reindex(self.tmpl_ind_flat).array.reshape(self.tmpl_shape)
        im = Image(im_, ssiaat_converter=self)
        return im

    def image_to_itable(self, image: Image | np.ndarray):
        itable = pd.Series(np.ravel(image), index=self.tmpl_ind_flat)
        return itable

    def read_stable(self, *fnlist, index_column="tmpl_ind", ignore_index_check=False):
        dfl = [pd.read_parquet(fn) for fn in fnlist]
        df = pd.concat(dfl, axis=0).set_index(index_column)
        if not ignore_index_check and not check_index_stable(df):
             raise ValueError("The input dataframe's index is no of integer of do"
                             " not have duplicates which is unusual. If you are sure"
                             "with the input, set `ignore_index_check` to True.")

        df._ssiaat_converter = self
        return df


def read_stable(*fnlist, tmpl_name="", index_column="tmpl_ind"):
    pass


# class TableToImage:
#     def __init__(self, template_file):
#         self.tmpl = fits.open(template_file)
#         # wcs_tmpl = WCS(tmpl[0].header)
#         self.tmpl_shape = self.tmpl[0].data.shape

#         self.tmpl_ind = np.sum(np.indices(self.tmpl_shape) * np.array([self.tmpl_shape[-1], 1]).reshape((2, 1, 1)),
#                           axis=0, dtype="int32")

#     def series_to_image(self, s, ignore_index_name=False):
#         # s should be a series whose index is a subset of tmpl_ind.
#         if not ignore_index_name and s.index.name != "tmpl_ind":
#             raise ValueError("input s needs to have an index named 'tmpl_ind' unless 'ignore_index_name' is True")
#         im = s.reindex(self.tmpl_ind.flat).array.reshape(self.tmpl_shape)
#         return im



# %%

# class TableTool:
#     """
#     table should have
#       tmpl_ind : value identifying a single pixel in the image.
#       wvl : wavelength in um
#     """
#     def __init__(self):
#         self.table = None

#     def load_table(self):
#         pass

#     def make_simple_image(self, um0, um1):
#         dff = self.table.query(f"({um0} < wvl) and (wvl < {um1})").copy()



# %%
from scipy.interpolate import interp1d
from spherex_tabular_bandpass import Tabular_Bandpass

class BandpassTool:
    def __init__(self, bandpass_model, band):
        self.bandpass_model = bandpass_model
        self.band = band
        
        iy = np.arange(0, 2048)
        ix = np.zeros_like(iy) + 1024

        wvl = self.bandpass_model(ix, iy, central_bandpass_only=True, array=band)
        self.wvl_to_iy = interp1d(wwl[0], iy)
        
    def get_bp_at_wvl(self, center, as_knots=False):
        """
        as_knots : return the interpolated model.
        """
        iyy = self.wvl_to_iy(center)

        w1, t1 = self.bandpaas_modelbp(1024, iyy, array=self.band)

        if as_knots:
            return interp1d(w1, t1)
        else:
            return w1, t1
        
# knots = interp1d(w1, t1)

# band = 5
# bp = Tabular_Bandpass()
# kknots.append(knots)
# from sed import hflattop
# d_shift = 0.012 # somehow wavelength solutionseem to be off
# br_a = hflattop(4.0372-d_shift, 4.0724-d_shift, a=0.75) # a adjusted to fit the bandpss



# %%
from itertools import chain
from vectorized_lstsq import vectorized_lstsq_numpy

class FitResults:
    def __init__(self, idx, C, Cerr=None, *, model=None):
        self.idx = idx
        self._C = C
        self.C = [C[:, i] for i in range(len(model.model_names))]
        self.contC = [C[:, i] for i in range(len(model.model_names),
                                             len(model.all_model_names))]
        self._Cerr = Cerr
        self.model = model

    def cont_sub(self, wvl, spec):
        # wvl = stable["wvl"]
        cont_amps = [wvl.align(pd.Series(c, index=self.idx), join="left")[1] for c
                     in self.contC]
        cont = np.sum([a*m(wvl) for a, m in zip(cont_amps, self.model.cont_models)], axis=0)
        return spec - pd.Series(cont, index=wvl.index)

    def norm(self, wvl, spec, param_i):
        norm = wvl.align(pd.Series(self.C[param_i], index=self.idx), join="left")[1]
        return spec / norm


class Model:
    """
    linear combination of models.
    """
    def __init__(self, models, cont_models):
        self.models = models
        self.cont_models = cont_models

        self.model_names = [self._get_model_name(i, m) for (i, m) in enumerate(models)]
        self.cont_model_names = [self._get_cont_model_name(i, m) for (i, m) in enumerate(cont_models)]
        self.all_model_names = self.model_names + self.cont_model_names
    
    def _get_model_name(self, i, m):
        return f"model{i}"

    def _get_cont_model_name(self, i, m):
        return f"cmodel{i}"

    def _populate_table_with_model_eval(self, stable, inplace=False):
        df = stable # stable is now a DataFrame
        k = {}
        for mid, m in chain(zip(self.model_names, self.models),
                            zip(self.cont_model_names, self.cont_models)):
            k[mid] = m(df["wvl"])

        if inplace:
            for n in k:
                df.loc[:, n] = k[n]

            df2 = df
        else:
            df2 = pd.DataFrame(k, index=df.index)
            # df2.loc[:, "tmpl_ind"] = df["tmpl_ind"]
            df2.loc[:, "wvl"] = df["wvl"]
            df2.loc[:, "image"] = df["image"]
            df2.loc[:, "variance"] = df["variance"]
            
        return df2

    def _least_square_fit(self, df, variance_column="variance", return_error=False):
        if return_error:
            C, C_err, idx = vectorized_lstsq_numpy(df, self.all_model_names, variance_column=variance_column, return_error=True)
            return idx, C, C_err
        else:
            C, idx = vectorized_lstsq_numpy(df, self.all_model_names,
                                            variance_column=variance_column,
                                            return_error=False)
            return idx, C

    def least_square_fit(self, stable, variance_column="variance", return_error=False):
        df = self._populate_table_with_model_eval(stable)

        idx_C_Cerr = self._least_square_fit(df,
                                            variance_column=variance_column,
                                            return_error=return_error)
        return FitResults(*idx_C_Cerr, model=self)

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


def main():
    root = "eso_244"
    template_name = f"{root}_template.fits"
    ssiaat_converter = SsiaatConverter(template_name)

    from pathlib import Path
    datadir = Path(".")

    fnlist = [str(datadir / f"{root}_b{band}.parquet") for band in [3, 4, 5]]

    stable_ = ssiaat_converter.read_stable(*fnlist)
    stable = stable_.query("(2.6 < wvl) and (wvl < 4.0)")
    stable._ssiaat_converter = stable_._ssiaat_converter

    im = stable.spectral.make_simple_image(3.1, 4.0)
    # fits.PrimaryHDU(data=im).writeto("a.fits", overwrite=True)

    spectral_model = get_test_model()

    # df = spectral_model._populate_table_with_model_eval(stable)
    # C, idx = spectral_model._least_square_fit(df)
    fitted_model = spectral_model.least_square_fit(stable)

    print(fitted_model.C[0])
    itable = pd.Series(fitted_model.contC[1], index=fitted_model.idx)
    itable._ssiaat_converter = ssiaat_converter
    im = ssiaat_converter.itable_to_image(itable) #, # pd.Series(C[:, 1], index=idx),
                                          # ignore_index_name=True)
    fits.PrimaryHDU(data=im).writeto("b.fits", overwrite=True)

    # spatial filtering
    sreg = "image;ellipse(31.403764,28.577416,4.3068155,8.0752791,353.88636)"
    import pyregion
    reg = pyregion.parse(sreg)
    msk = reg.get_mask(shape=(61, 61))

    s = stable.spectral.filter_with_image_mask(msk)

    param_i = 0
    imsk = ssiaat_converter.image_to_itable(msk)
    c0 = pd.Series(fitted_model.C[0], index=fitted_model.idx)
    c1 = pd.Series(fitted_model.C[1], index=fitted_model.idx)
    # c1 = ssiaat_converter.image_to_itable(fitted_model.C[1])

    ss_contsub = fitted_model.cont_sub(s["wvl"], s["image"])
    # ss_contsub_n_normed = fitted_model.norm(s["wvl"], ss_contsub, param_i)

    _, ccc = s["wvl"].align(c0[imsk], join="left")

    import matplotlib.pyplot as plt
    w = s["wvl"]
    plt.scatter(s["wvl"], ss_contsub / ccc, s=1)

    xx = np.linspace(2.6, 4.0, 100)

    median_c1_c0 = np.nanmedian(c1[imsk] / c0[imsk])
    cc0 = spectral_model.models[0](xx)
    cc1 = median_c1_c0 * spectral_model.models[1](xx)
    plt.plot(xx,  cc0 + cc1, "-", lw=3, alpha=0.5)
    plt.plot(xx, cc0)
    plt.plot(xx, cc1)
    plt.show()
    # stable.

if __name__ == '__main__':
    main()

