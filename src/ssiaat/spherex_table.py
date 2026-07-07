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
from .wcs_helper import TemplateHeaderCards

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


def _get_converter(obj):
    """Return the converter of a stable/itable pandas object.

    Uses the cached live instance when present, otherwise reconstructs one
    from the template header cards in ``obj.attrs`` (and caches it).
    Raises with an actionable message instead of returning None -- a None
    here only surfaces later as a confusing AttributeError deep inside
    itable_to_image.
    """
    conv = getattr(obj, "_ssiaat_converter", None)
    if conv is not None:
        return conv

    header_cards = obj.attrs.get("ssiaat_template_header")
    if header_cards:
        header = fits.Header.fromstring("".join(header_cards))
        conv = SsiaatConverter(header)
        # Cache it as a private attribute for subsequent fast access
        try:
            obj._ssiaat_converter = conv
        except Exception:
            pass
        return conv

    raise ValueError(
        "no template header metadata in .attrs, so the converter cannot"
        " be reconstructed. Run promote_to_stable(df, header=...) or read"
        " the data via SsiaatConverter.read_stable(...).")


@pd.api.extensions.register_dataframe_accessor("spectral")
class SpectralTable:
    def __init__(self, pandas_obj):
        if not check_index_stable(pandas_obj):
            raise AttributeError("Spectral accessor only available for DataFrames with integer index and duplicates.")
        self._obj = pandas_obj

    @property
    def converter(self):
        return _get_converter(self._obj)

    def make_simple_itable(self, w1, w2, column="image", agg="mean"):
        """Aggregate `column` per pixel over the (w1, w2) wavelength window.

        Returns an itable (Series with unique pixel index) carrying the
        template metadata, so ``result.itable.to_image()`` works.
        """
        dfc = self._obj.query(f"({w1} < wvl) and (wvl < {w2})")
        s = dfc.groupby(by=dfc.index)[column].agg(agg)
        s.attrs["ssiaat_template_header"] = \
            self._obj.attrs.get("ssiaat_template_header")
        return s

    def make_simple_image(self, w1, w2, column="image", agg="mean"):
        """make_simple_itable rendered onto the template grid as an Image."""
        s = self.make_simple_itable(w1, w2, column=column, agg=agg)
        return self.converter.itable_to_image(s)

    def binned_spectrum(self, w1=None, w2=None, column="image", bins=50,
                        agg="median"):
        """Wavelength-binned aggregate spectrum of `column`.

        Returns a Series indexed by bin-center wavelength (plot-ready:
        ``stable.spectral.binned_spectrum().plot()``). `bins` is either
        the number of equal-width bins over the (selected) wavelength
        range, or an explicit array of bin edges.
        """
        df = self._obj
        if w1 is not None or w2 is not None:
            lo = w1 if w1 is not None else -np.inf
            hi = w2 if w2 is not None else np.inf
            df = df.query(f"({lo} < wvl) and (wvl < {hi})")

        w = df["wvl"]
        if np.isscalar(bins):
            bins = np.linspace(w.min(), w.max(), bins)
        else:
            bins = np.asarray(bins)

        values = df[column].groupby(pd.cut(w, bins), observed=False).agg(agg)
        centers = 0.5 * (bins[1:] + bins[:-1])
        return pd.Series(values.array, index=pd.Index(centers, name="wvl"),
                         name=column)

    def filter_with_image_mask(self, msk):
        itable = self.converter.image_to_itable(msk)
        return self._obj.loc[itable]

    def broadcast(self, values):
        """Align per-pixel values onto this table's (pixel x wavelength) rows.

        Parameters
        ----------
        values : pd.Series or np.ndarray
            Per-pixel values, either an itable (series with unique integer
            index) or a 2d image of the template shape.

        Returns
        -------
        pd.Series aligned row-by-row with the table, so it can be used in
        arithmetic with the table's columns, e.g.
        ``df["image"] - df.spectral.broadcast(cont)``.
        """
        if isinstance(values, np.ndarray):
            values = self.converter.image_to_itable(values)
        return values.reindex(self._obj.index)

@pd.api.extensions.register_series_accessor("itable")
class ImageTable:
    def __init__(self, pandas_obj):
        if not check_index_itable(pandas_obj):
            raise AttributeError("ImageTable accessor only available for Series with unique integer index.")
        self._obj = pandas_obj

    @property
    def converter(self):
        return _get_converter(self._obj)

    def to_image(self):
        """Converts the image back to its tabular form using its converter."""
        return self.converter.itable_to_image(self._obj)


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
        """Converts the image back to its tabular form using its converter."""
        if self._ssiaat_converter is None:
            raise ValueError("No ssiaat_converter attached to this Image.")
        return self._ssiaat_converter.image_to_itable(self)


def promote_to_stable(df, index_column="tmpl_ind", header=None,
                      ignore_index_check=False):
    """Turn an in-memory dataframe into a stable usable with the .spectral accessor.

    Sets `index_column` as the index (unless it already is, or index_column
    is None), checks that the index looks like a stable's (integer with
    duplicates), and makes sure the template header metadata needed to
    reconstruct the converter is available in df.attrs. Pass `header` (a
    fits.Header) to attach the metadata to a dataframe that does not have it.
    """
    if index_column is not None and df.index.name != index_column:
        df = df.set_index(index_column)

    if not ignore_index_check and not check_index_stable(df):
        raise ValueError("The dataframe's index is not integer or does not"
                         " have duplicates, which is unusual for a stable."
                         " If you are sure about the input, set"
                         " `ignore_index_check` to True.")

    if header is not None:
        TemplateHeaderCards.from_header(header).update_dataframe(df)
    elif df.attrs.get("ssiaat_template_header") is None:
        raise ValueError("No template header metadata found in df.attrs, so"
                         " the .spectral accessor will not be able to"
                         " reconstruct the converter. Pass `header` to"
                         " attach one.")

    return df

def read_stable(*fnlist, index_column="tmpl_ind", header=None,
                columns=None, wvl_range=None, ignore_index_check=False):
    """Read one or more parquet stables and concatenate them.

    Accepts either varargs or a single list:
    ``read_stable("a.parquet", "b.parquet")`` and
    ``read_stable(["a.parquet", "b.parquet"])`` are equivalent.

    Parameters
    ----------
    index_column : str or None
        Column to promote to the index (skipped when the parquet index is
        already named so).
    header : fits.Header, optional
        Template header to (re)attach; without it the parquet files must
        already carry the template metadata in their attrs.
    columns : list of str, optional
        Read only these columns (pyarrow column pruning). Most analyses
        never need src_x/src_y.
    wvl_range : (float, float), optional
        Read only rows with w1 < wvl < w2 (pyarrow predicate pushdown --
        much cheaper than reading everything and calling .query).
    """
    if len(fnlist) == 1 and isinstance(fnlist[0], (list, tuple)):
        fnlist = fnlist[0]

    read_kwargs = {}
    if columns is not None:
        read_kwargs["columns"] = columns
    if wvl_range is not None:
        w1, w2 = wvl_range
        read_kwargs["filters"] = [("wvl", ">", w1), ("wvl", "<", w2)]

    header_cards = None
    dfl = []
    for fn in fnlist:
        df = pd.read_parquet(fn, **read_kwargs)
        dfl.append(df)
        header_cards_ = TemplateHeaderCards.from_dataframe(df)
        if header_cards is not None and header_cards != header_cards_:
            raise ValueError("the input files have inconsistent metadata.")
        header_cards = header_cards_

    df = pd.concat(dfl, axis=0)
    return promote_to_stable(df, index_column=index_column, header=header,
                             ignore_index_check=ignore_index_check)


class SsiaatConverter:
    def __init__(self, header: fits.Header):
        self.header = header
        self.tmpl_shape = (self.header['NAXIS2'], self.header['NAXIS1'])

        # 2d array for pixel indices
        self.tmpl_ind = np.sum(np.indices(self.tmpl_shape)
                               * np.array([self.tmpl_shape[-1], 1]).reshape((2, 1, 1)),
                               axis=0, dtype="int32")
        self.tmpl_ind_flat = np.ravel(self.tmpl_ind)

    @classmethod
    def from_file(cls, template_file):
        with fits.open(template_file) as hdul:
            header = hdul[0].header.copy()
        return cls(header)

    def itable_to_image(self, itable: pd.Series, ignore_index_name=False):
        # itable should be a series whose index is a subset of tmpl_ind.
        im_ = itable.reindex(self.tmpl_ind_flat).array.reshape(self.tmpl_shape)
        im = Image(im_, ssiaat_converter=self)
        return im

    def image_to_itable(self, image: Image | np.ndarray,
                        mask: None | Image | np.ndarray = None):
        data = np.ravel(image)
        if not data.dtype.isnative:
            # FITS data is big-endian; pandas requires native byte order
            data = data.astype(data.dtype.newbyteorder("="))
        itable = pd.Series(data, index=self.tmpl_ind_flat)
        if mask is not None:
            itable_msk = pd.Series(np.ravel(mask), index=self.tmpl_ind_flat)
            itable = itable[itable_msk]

        return itable

    def read_stable(self, *fnlist, index_column="tmpl_ind", columns=None,
                    wvl_range=None, ignore_index_check=False):
        """Like the module-level read_stable, with this converter's header
        attached and the live converter cached on the result."""
        df = read_stable(*fnlist, index_column=index_column,
                         header=self.header, columns=columns,
                         wvl_range=wvl_range,
                         ignore_index_check=ignore_index_check)
        df._ssiaat_converter = self
        return df


# %%
from scipy.interpolate import interp1d
# from spherex_tabular_bandpass import Tabular_Bandpass
from .tabular_bandpass_lite import Tabular_Bandpass_Lite as Tabular_Bandpass

class BandpassTool:
    def __init__(self, bandpass_model, band):
        self.bandpass_model = bandpass_model
        self.band = band
        
        iy = np.arange(0, 2048)
        ix = np.zeros_like(iy) + 1024

        wvl, _ = self.bandpass_model(ix, iy, central_bandpass_only=True, array=band)
        self.wvl_to_iy = interp1d(wvl, iy)
        
    def get_bp_at_wvl(self, center, as_knots=False):
        """
        as_knots : return the interpolated model.
        """
        iyy = self.wvl_to_iy(center)

        w1, t1 = self.bandpass_model(1024, iyy, array=self.band)

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
from .model.vectorized_lstsq import vectorized_lstsq_numpy

class FitResults:
    def __init__(self, idx, C, Cerr=None, *, model,
                 ssiaat_template_header=None):
        self.idx = idx
        self._C = C
        # Store coefficients as Series to leverage pandas index alignment
        self.C = [pd.Series(C[:, i], index=idx) for i in range(len(model.model_names))]
        self.contC = [pd.Series(C[:, i], index=idx) for i in range(len(model.model_names),
                                             len(model.all_model_names))]

        if ssiaat_template_header is not None:
            for s in chain(self.C, self.contC):
                s.attrs["ssiaat_template_header"] = ssiaat_template_header

        if Cerr is not None:
            self.Cerr = [pd.Series(Cerr[:, i], index=idx) for i in range(len(model.model_names))]
            self.contCerr = [pd.Series(Cerr[:, i], index=idx)
                             for i in range(len(model.model_names),
                                            len(model.all_model_names))]

            if ssiaat_template_header is not None:
                for s in chain(self.Cerr, self.contCerr):
                    s.attrs["ssiaat_template_header"] = ssiaat_template_header

        self._Cerr = Cerr
        self.model = model

    def _name_index(self, name):
        try:
            return self.model.all_model_names.index(name)
        except ValueError:
            raise KeyError(
                f"unknown model name {name!r};"
                f" available: {self.model.all_model_names}") from None

    def coef(self, name):
        """Coefficient Series (indexed by pixel) for the named model."""
        i = self._name_index(name)
        n = len(self.model.model_names)
        return self.C[i] if i < n else self.contC[i - n]

    def err(self, name):
        """Coefficient-error Series for the named model."""
        if self._Cerr is None:
            raise ValueError("no errors available: run least_square_fit"
                             " with return_error=True")
        i = self._name_index(name)
        n = len(self.model.model_names)
        return self.Cerr[i] if i < n else self.contCerr[i - n]

    def to_frame(self):
        """All coefficients (and errors, when fitted) as a DataFrame with
        named columns, indexed by pixel -- directly parquet-serializable."""
        data = {name: self.coef(name) for name in self.model.all_model_names}
        if self._Cerr is not None:
            for name in self.model.all_model_names:
                data[f"{name}_err"] = self.err(name)
        return pd.DataFrame(data)

    def image(self, name):
        """Coefficient map of the named model rendered on the template."""
        return self.coef(name).itable.to_image()

    def cont_sub(self, wvl, spec):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        # as wvl, with values broadcasted to each duplicate index in the original order.
        cont = sum(amp.reindex(wvl.index).values * m(wvl)
                   for amp, m in zip(self.contC, self.model.cont_models))
        return spec - cont

    def norm(self, wvl, spec, param_i):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        return spec / self.C[param_i].reindex(wvl.index).values

    def cont_sub_n_norm(self, wvl, spec, param_i):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        spec_cont_sub = self.cont_sub(wvl, spec)
        return spec_cont_sub / self.C[param_i].reindex(wvl.index).values



class Model:
    """
    Linear combination of models (line models + continuum models).

    Each of `models` and `cont_models` is either a list of callables of
    wavelength (auto-named model0/model1/... and cmodel0/...) or a dict
    mapping names to callables, e.g.::

        Model({"br_a": get_br_a()}, {"cont": const()})

    Coefficients can then be read back by name: fitted.coef("br_a").
    """
    def __init__(self, models, cont_models):
        self.model_names, self.models = self._normalize(models, "model")
        self.cont_model_names, self.cont_models = self._normalize(cont_models,
                                                                  "cmodel")
        self.all_model_names = self.model_names + self.cont_model_names

        if len(set(self.all_model_names)) != len(self.all_model_names):
            raise ValueError("model names must be unique across models and"
                             f" cont_models: {self.all_model_names}")

    @staticmethod
    def _normalize(models, prefix):
        from collections.abc import Mapping
        if isinstance(models, Mapping):
            return list(models.keys()), list(models.values())
        models = list(models)
        return [f"{prefix}{i}" for i in range(len(models))], models

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
        ssiaat_template_header = stable.attrs.get("ssiaat_template_header", None)
        return FitResults(*idx_C_Cerr, model=self,
                          ssiaat_template_header=ssiaat_template_header)
