#

"""
OriginalSpectralImage (original)

SpectralImage (reprojected)

SpectralTable (stable) : reprojected spectral images as a table. should have a column of tmpl_ind. There can be multiple rows of same tmpl_ind value.

ImageTable (itable) : image as table. should have a index of tmpl_ind that maps to a unique pixel in the template image. The index of tmpl_ind should be uniquely valued.

Image (image) : image ()in reprojected template.)

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


class SpectralTable:
    def __init__(self, df, ssiaat_converter=None, ignore_index_check=False):
        if not ignore_index_check and not check_index_stable(df):
            raise ValueError("The input dataframe's index is no of integer of do"
                             " not have duplicates which is unusual. If you are sure"
                             "with the input, set `ignore_index_check` to True.")

        self.df = df

        self._ssiaat_converter = ssiaat_converter

    # @functools.cached_property
    # def _df_groupby(self):
    #     return self._df.groupby(by=self._df.index)
    #     #"tmpl_ind"

    @classmethod
    def read_parquet(cls, *fnlist, index_column="tmpl_ind", ssiaat_converter=None):
        dfl = [pd.read_parquet(fn) for fn in fnlist]
        df = pd.concat(dfl, axis=0).set_index(index_column)
        return cls(df, ssiaat_converter)

    def make_simple_image(self, w1, w2):
        dfc = self.df.query(f"({w1} < wvl) and (wvl < {w2})")
        # ggc = dfc.groupby("tmpl_ind")

        s = dfc.groupby(by=dfc.index)["image"].mean()
        itable = ImageTable(s)
        return self._ssiaat_converter.itable_to_image(itable)
        # smeanR = s.reindex(tmpl_ind.flat).array.reshape(tmpl_ind.shape)


class ImageTable:
    def __init__(self, df, ssiaat_converter=None, ignore_index_check=False):
        if not ignore_index_check and not check_index_itable(df):
            raise ValueError("The input dataframe's index is no of integer of do"
                             " not have duplicates which is unusual. If you are sure"
                             "with the input, set `ignore_index_check` to True.")

        self.df = df

        self._ssiaat_converter = ssiaat_converter


class Image:
    def __init__(self, image: np.ndarray, ssiaat_converter=None):
        self.image = image
        self._ssiaat_converter = ssiaat_converter


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

    def itable_to_image(self, itable: ImageTable, ignore_index_name=False):
        # s should be a series whose index is a subset of tmpl_ind.
        # if not ignore_index_name and itable.index.name != "tmpl_ind":
        #     raise ValueError("input needs to have an index named 'tmpl_ind' unless 'ignore_index_name' is True")
        im_ = itable.df.reindex(self.tmpl_ind_flat).array.reshape(self.tmpl_shape)
        im = Image(im_, ssiaat_converter=self)
        return im

    def image_to_itable(self, image: Image | np.ndarray):
        itable = pd.Series(np.ravel(image), index=self.tmpl_ind)
        return itable

    def read_stable(self, *fnlist, index_column="tmpl_ind"):
        return SpectralTable.read_parquet(*fnlist, index_column="tmpl_ind",
                                          ssiaat_converter=self)


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
        df = stable.df
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
            return C, C_err, idx
        else:
            C, idx = vectorized_lstsq_numpy(df, self.all_model_names,
                                            variance_column=variance_column,
                                            return_error=False)
            return C, idx



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

    fnlist = [str(datadir / f"{root}_b{band}.parquet") for band in [3, 4]]

    stable = ssiaat_converter.read_stable(*fnlist)

    im = stable.make_simple_image(3.1, 4.0)
    # fits.PrimaryHDU(data=im).writeto("a.fits", overwrite=True)

    spectral_model = get_test_model()

    df = spectral_model._populate_table_with_model_eval(stable)
    C, idx = spectral_model._least_square_fit(df)

    itable = ImageTable(pd.Series(C[:, 1], index=idx))
    im = ssiaat_converter.itable_to_image(itable) #, # pd.Series(C[:, 1], index=idx),
                                          # ignore_index_name=True)
    fits.PrimaryHDU(data=im.image).writeto("b.fits", overwrite=True)

if __name__ == '__main__':
    main()

