import warnings

import numpy as np
import pandas as pd


def vectorized_lstsq_arrays(model_arrays, target, codes, n_groups, *,
                            weights=None, return_error=False):
    """Per-group linear least squares via normal equations, numpy-only.

    Parameters
    ----------
    model_arrays : sequence of 1d ndarrays
        One array per model, evaluated on every row.
    target : 1d ndarray
        The data to fit.
    codes : 1d int ndarray
        Group label per row, in [0, n_groups) (e.g. from ``pd.factorize``).
    n_groups : int
        Number of groups.
    weights : 1d ndarray, optional
        Per-row weights (typically 1/variance). None means unweighted.
    return_error : bool
        Also return per-coefficient error estimates.

    Returns
    -------
    coeffs : (n_groups, n_models) ndarray
    coeffs_err : (n_groups, n_models) ndarray, only when return_error

    Notes
    -----
    Each cross product is accumulated per group with ``np.bincount`` and
    freed before the next one, so peak extra memory is about one
    row-length float64 array -- unlike the previous pandas groupby
    implementation, which materialized every product at once.
    """
    n_models = len(model_arrays)
    target = np.asarray(target, dtype="float64")

    def groupsum(values):
        return np.bincount(codes, weights=values, minlength=n_groups)

    ATA = np.zeros((n_groups, n_models, n_models))
    ATy = np.zeros((n_groups, n_models, 1))

    for i in range(n_models):
        mi_w = np.asarray(model_arrays[i], dtype="float64")
        if weights is not None:
            mi_w = mi_w * weights
        for j in range(i, n_models):
            val = groupsum(mi_w * model_arrays[j])
            ATA[:, i, j] = val
            if i != j:
                ATA[:, j, i] = val
        ATy[:, i, 0] = groupsum(mi_w * target)

    # Solve for all groups at once. Using pinv handles singular matrices gracefully.
    inv_ATA = np.linalg.pinv(ATA)
    coeffs = (inv_ATA @ ATy).squeeze(axis=-1)

    if return_error:
        # Error estimation
        wy2 = target * target
        if weights is not None:
            wy2 = wy2 * weights
        sum_wy2 = groupsum(wy2)
        sum_c_ATy = np.einsum('ni,ni->n', coeffs, ATy.squeeze(axis=-1))
        rss = np.maximum(sum_wy2 - sum_c_ATy, 0)  # RSS should be non-negative

        if weights is None:
            n_samples = np.bincount(codes, minlength=n_groups)
            dof = n_samples - n_models
            dof = np.where(dof > 0, dof, np.nan)
            sigma2 = rss / dof
            coeffs_cov = inv_ATA * sigma2[:, np.newaxis, np.newaxis]
        else:
            coeffs_cov = inv_ATA

        coeffs_err = np.sqrt(np.maximum(np.diagonal(coeffs_cov, axis1=1, axis2=2), 0))
        return coeffs, coeffs_err

    return coeffs


def vectorized_lstsq_numpy(df, model_columns,
                           target_column="image", group_column=None, # "tmpl_ind",
                           variance_column=None, return_error=False):
    """
    Perform least squares fitting for multiple groups in parallel using normal equations.
    Supports weighted least squares if variance_column is provided.

    The input is dataframe that will be grouped by `group_column`. If `group_column`
    is None, index will be used. The fit will be
    basically done for each group which could have different number of rows. Only the
    amplitude of each model is fitted.
    """
    labels = df.index if group_column is None else df[group_column]
    codes, uniques = pd.factorize(labels, sort=True)
    index_values = np.asarray(uniques)

    if variance_column is not None:
        weights = 1.0 / df[variance_column].to_numpy(dtype="float64")
    else:
        weights = None

    model_arrays = [df[col].to_numpy() for col in model_columns]
    target = df[target_column].to_numpy()

    result = vectorized_lstsq_arrays(model_arrays, target, codes,
                                     len(index_values), weights=weights,
                                     return_error=return_error)
    if return_error:
        coeffs, coeffs_err = result
        return coeffs, coeffs_err, index_values

    return result, index_values


def vectorized_lstsq_chunked(df, model_columns, chunk_size=1000, **kwargs):
    """Deprecated: the bincount-based vectorized_lstsq_numpy no longer
    builds large intermediates, so chunking buys nothing. This wrapper
    keeps the old default of grouping by the "tmpl_ind" column (the numpy
    version defaults to the index) and ignores chunk_size.
    """
    warnings.warn(
        "vectorized_lstsq_chunked is deprecated: vectorized_lstsq_numpy now"
        " runs at near-constant memory; call it directly.",
        DeprecationWarning, stacklevel=2)
    kwargs.setdefault("group_column", "tmpl_ind")
    return vectorized_lstsq_numpy(df, model_columns, **kwargs)
