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
    converter = SsiaatConverter.from_file("template.fits")
    
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
from astropy.io import fits


# %%
from pandas.api.types import is_integer_dtype
from .wcs_helper import TemplateHeaderCards
from .indexing import make_pixel_index

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
        # Only an integer index is required -- NOT duplicates. A stable
        # normally has many wavelengths per pixel (duplicated tmpl_ind),
        # but a tile covered by a single exposure legitimately has one row
        # per pixel (unique index) and every method here (groupby(index),
        # pd.cut, reindex) works fine on it. check_index_stable stays
        # strict for promote_to_stable's safety net; the accessor does not.
        if not is_integer_dtype(pandas_obj.index.dtype):
            raise AttributeError(
                "Spectral accessor only available for DataFrames with an"
                " integer index.")
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
        """Render this itable onto the template grid as an Image."""
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
        if obj is None:
            return
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
        self.tmpl_ind = make_pixel_index(self.tmpl_shape)
        self.tmpl_ind_flat = np.ravel(self.tmpl_ind)

    @classmethod
    def from_file(cls, template_file):
        with fits.open(template_file) as hdul:
            header = hdul[0].header.copy()
        return cls(header)

    def itable_to_image(self, itable: pd.Series, ignore_index_name=False):
        # itable should be a series whose index is a subset of tmpl_ind.
        # tmpl_ind_flat is arange(n), so index values ARE flat positions:
        # preallocate + assign beats reindexing over the full template
        # (several x at HiPS scales).
        n = self.tmpl_shape[0] * self.tmpl_shape[1]
        out = np.full(n, np.nan)
        out[itable.index.to_numpy()] = itable.to_numpy(dtype="float64")
        im = Image(out.reshape(self.tmpl_shape), ssiaat_converter=self)
        return im

    def _itable_to_image_reindex(self, itable: pd.Series):
        # Reference implementation of itable_to_image (kept for
        # correctness comparison in tests).
        im_ = itable.reindex(self.tmpl_ind_flat).array.reshape(self.tmpl_shape)
        return Image(im_, ssiaat_converter=self)

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

    def fit_hflatline(self, center, *, mu_center=None, symmetric=False,
                      full_output=False):
        """Fit sed.hflatline's shape to the real bandpass at `center`.

        hflatline is meant for broad, genuinely flat-topped continuum/line
        models elsewhere in `model.sed`, but the fit is excellent here too:
        the narrow LVF bandpass (FWHM ~ center/R) is well matched by the
        degenerate case where the plateau shrinks to (near) a point.

        `amplitude` is fit jointly with the shape parameters (not just
        read off the data's peak): hflatline is a product of two
        tanh-based edges, and its own peak value is generically *below* 1
        (how far below depends on dmu/a), so fixing amplitude from the raw
        peak and fitting a peak-normalized target biases the recovered
        shape to compensate. Letting `curve_fit` solve for amplitude too
        avoids that bias.

        When `mu_center_left` is left free (the default) and the edges are
        asymmetric, it trades off against `dmu_left`/`dmu_right`: shifting
        the shared center while adjusting both half-widths to compensate
        reproduces an almost identical curve, so individual parameter
        values can wobble noticeably run to run even though the
        reconstructed curve and `amplitude` are essentially exact
        (measured: `corr(dmu_left, a_left)` = +1.000 within a 4-pixel
        window on the real SPHEREx bandpass). **Pass `mu_center=` whenever
        it's already known** -- e.g. from `Tabular_Bandpass_Lite`'s central
        wavelength -- which removes the degeneracy at no cost: measured
        fit quality is identical to 4 significant figures whether
        `mu_center_left` is fixed at the true value or left free. Passing
        `symmetric=True` instead also removes the degeneracy, but at a
        real cost in fit quality (up to ~4x worse RMS in some bands) --
        prefer a fixed `mu_center` over `symmetric=True` when the center is
        known.

        Parameters
        ----------
        center : float
            Requested central wavelength (microns), central column
            (ix=1024) -- same convention as `get_bp_at_wvl`. Used as the
            free-fit initial guess for `mu_center_left` when `mu_center`
            is not given; ignored (but still required) when it is.
        mu_center : float, optional
            Hold `mu_center_left` fixed at this value instead of fitting
            it, removing the degeneracy described above. Must lie strictly
            inside the curve's wavelength window, or `ValueError` is
            raised. Default None (free-fit, today's original behavior).
        symmetric : bool
            Force dmu_right=dmu_left and a_right=a_left (3 free shape
            parameters instead of 5, plus amplitude). The real bandpass is
            usually slightly asymmetric, so leave this False unless you
            need a matched pair of edges; see the note above on cost vs.
            fixing `mu_center` instead.
        full_output : bool
            If True, also return a dict with the (w, t) curve, the fit
            residual, and its RMS.

        Returns
        -------
        dict with keys ``mu_center_left``, ``dmu_left``, ``a_left``,
        ``dmu_right``, ``a_right`` (hflatline's constructor kwargs
        verbatim -- pass ``**params`` after popping ``amplitude``) plus
        ``amplitude``, the fitted peak scale. NOT an hflatline parameter;
        kept alongside so ``amplitude * hflatline(**params)(w)``
        approximates the original tabulated bandpass.
        """
        w1, t1 = self.get_bp_at_wvl(center)
        return self._fit_hflatline_curve(w1, t1, center, mu_center=mu_center,
                                         symmetric=symmetric,
                                         full_output=full_output)

    def fit_hflatline_at_iy(self, iy, *, mu_center="analytic",
                            symmetric=False, full_output=False):
        """Like `fit_hflatline`, but at an exact detector row `iy`.

        `get_bp_at_wvl(center)` inverts wavelength -> row via `interp1d`
        and the bandpass model rounds internally, so asking for the curve
        at a wavelength and separately asking for the analytic central
        wavelength "at the same row" can silently land on different rows
        near template-quantization boundaries (adjacent rows often share
        the identical tabulated curve). This method instead calls the
        bandpass model directly at the same integer `iy` for both the
        curve and (when needed) the analytic center -- no inversion, no
        rounding mismatch. Intended for bulk per-row sweeps (see
        `examples/derive_bandpass_shape_coeffs.py`).

        Parameters
        ----------
        iy : int
            Detector row, central column (ix=1024).
        mu_center : "analytic", float, or None
            "analytic" (default): fix `mu_center_left` to this row's own
            `central_bandpass_only=True` value. A float: fix it to that
            value instead. None: free-fit it (seeded from the analytic
            value, but not held to it).
        symmetric, full_output : see `fit_hflatline`.
        """
        w1, t1 = self.bandpass_model(1024, iy, array=self.band)
        wc, _ = self.bandpass_model(1024, iy, array=self.band,
                                    central_bandpass_only=True)
        wc = float(np.asarray(wc))

        if mu_center == "analytic":
            mu_center = wc

        return self._fit_hflatline_curve(w1, t1, wc, mu_center=mu_center,
                                         symmetric=symmetric,
                                         full_output=full_output)

    @staticmethod
    def _fit_hflatline_curve(w1, t1, center, *, mu_center=None,
                             symmetric=False, full_output=False):
        from scipy.optimize import curve_fit
        from .model.sed import hflatline

        w = np.asarray(w1, dtype=float)
        t = np.asarray(t1, dtype=float)

        fixed = mu_center is not None
        if fixed and not (w.min() < mu_center < w.max()):
            raise ValueError(
                f"mu_center={mu_center!r} is outside the curve's"
                f" wavelength window [{w.min()!r}, {w.max()!r}]")

        amplitude0 = float(t.max())

        # Initial guess for the half-width: half the empirical FWHM.
        above_half = w[t >= 0.5 * amplitude0]
        fwhm0 = ((above_half.max() - above_half.min())
                if len(above_half) >= 2 else (w.max() - w.min()) * 0.1)
        dmu0 = max(fwhm0 / 2, 1e-4)
        span = w.max() - w.min()

        if fixed and symmetric:
            def model_fn(mu, amplitude, dmu, a):
                return amplitude * hflatline(mu_center, dmu, a)(mu)
            p0 = [amplitude0, dmu0, 1.0]
            bounds = ([0, 1e-5, 1e-3], [10 * amplitude0, span, 10.0])
        elif fixed:
            def model_fn(mu, amplitude, dmu_left, a_left, dmu_right, a_right):
                return amplitude * hflatline(mu_center, dmu_left, a_left,
                                             dmu_right=dmu_right,
                                             a_right=a_right)(mu)
            p0 = [amplitude0, dmu0, 1.0, dmu0, 1.0]
            bounds = ([0, 1e-5, 1e-3, 1e-5, 1e-3],
                      [10 * amplitude0, span, 10.0, span, 10.0])
        elif symmetric:
            def model_fn(mu, amplitude, mu_c, dmu, a):
                return amplitude * hflatline(mu_c, dmu, a)(mu)
            p0 = [amplitude0, center, dmu0, 1.0]
            bounds = ([0, w.min(), 1e-5, 1e-3],
                      [10 * amplitude0, w.max(), span, 10.0])
        else:
            def model_fn(mu, amplitude, mu_c, dmu_left, a_left,
                         dmu_right, a_right):
                return amplitude * hflatline(mu_c, dmu_left, a_left,
                                             dmu_right=dmu_right,
                                             a_right=a_right)(mu)
            p0 = [amplitude0, center, dmu0, 1.0, dmu0, 1.0]
            bounds = ([0, w.min(), 1e-5, 1e-3, 1e-5, 1e-3],
                      [10 * amplitude0, w.max(), span, 10.0, span, 10.0])

        popt, _ = curve_fit(model_fn, w, t, p0=p0, bounds=bounds, maxfev=20000)

        if fixed and symmetric:
            amplitude, dmu, a = popt
            params = dict(mu_center_left=float(mu_center), dmu_left=dmu,
                         a_left=a, dmu_right=dmu, a_right=a)
        elif fixed:
            amplitude, dmu_left, a_left, dmu_right, a_right = popt
            params = dict(mu_center_left=float(mu_center), dmu_left=dmu_left,
                         a_left=a_left, dmu_right=dmu_right, a_right=a_right)
        elif symmetric:
            amplitude, mu_c, dmu, a = popt
            params = dict(mu_center_left=mu_c, dmu_left=dmu, a_left=a,
                         dmu_right=dmu, a_right=a)
        else:
            amplitude, mu_c, dmu_left, a_left, dmu_right, a_right = popt
            params = dict(mu_center_left=mu_c, dmu_left=dmu_left,
                         a_left=a_left, dmu_right=dmu_right, a_right=a_right)
        params["amplitude"] = amplitude

        if full_output:
            resid = t - model_fn(w, *popt)
            diagnostics = dict(w=w, t=t, resid=resid,
                              rms=float(np.sqrt(np.mean(resid**2))))
            return params, diagnostics
        return params

# knots = interp1d(w1, t1)

# band = 5
# bp = Tabular_Bandpass()
# kknots.append(knots)
# from sed import hflattop
# d_shift = 0.012 # somehow wavelength solutionseem to be off
# br_a = hflattop(4.0372-d_shift, 4.0724-d_shift, a=0.75) # a adjusted to fit the bandpss

# Model / FitResults moved to ssiaat.model.fitting; re-exported here since
# scripts and tests import them from this module.
from .model.fitting import Model, FitResults  # noqa: E402,F401
