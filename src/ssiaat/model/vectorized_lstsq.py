import pandas as pd
import numpy as np


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
    if variance_column is not None:
        weights = 1.0 / df[variance_column]
    else:
        weights = 1.0

    # Precompute all necessary products (weighted if applicable)
    products = {}
    for i, col1 in enumerate(model_columns):
        for j, col2 in enumerate(model_columns):
            if i <= j:
                products[f"m{i}m{j}"] = df[col1] * df[col2] * weights
        products[f"m{i}y"] = df[col1] * df[target_column] * weights

    if return_error:
        products["yy"] = df[target_column]**2 * weights
        if variance_column is None:
            products["count"] = np.ones(len(df))

    # Aggregate products by group
    if group_column is None:
        aggregated = pd.DataFrame(products).groupby(by=df.index).sum()
    else:
        aggregated = pd.DataFrame(products).groupby(df[group_column]).sum()

    # Construct the (N, M, M) ATA matrices and (N, M, 1) ATy vectors
    n_groups = len(aggregated)
    n_models = len(model_columns)
    ATA = np.zeros((n_groups, n_models, n_models))
    ATy = np.zeros((n_groups, n_models, 1))

    for i in range(n_models):
        for j in range(i, n_models):
            val = aggregated[f"m{i}m{j}"].values
            ATA[:, i, j] = val
            if i != j:
                ATA[:, j, i] = val
        ATy[:, i, 0] = aggregated[f"m{i}y"].values

    # Solve for all groups at once. Using pinv handles singular matrices gracefully.
    inv_ATA = np.linalg.pinv(ATA)
    coeffs = (inv_ATA @ ATy).squeeze(axis=-1)

    if return_error:
        # Error estimation
        sum_wy2 = aggregated["yy"].values
        sum_c_ATy = np.einsum('ni,ni->n', coeffs, ATy.squeeze(axis=-1))
        rss = np.maximum(sum_wy2 - sum_c_ATy, 0)  # RSS should be non-negative

        if variance_column is None:
            n_samples = aggregated["count"].values
            dof = n_samples - n_models
            dof = np.where(dof > 0, dof, np.nan)
            sigma2 = rss / dof
            coeffs_cov = inv_ATA * sigma2[:, np.newaxis, np.newaxis]
        else:
            coeffs_cov = inv_ATA

        coeffs_err = np.sqrt(np.maximum(np.diagonal(coeffs_cov, axis1=1, axis2=2), 0))
        return coeffs, coeffs_err, aggregated.index.values

    return coeffs, aggregated.index.values


def vectorized_lstsq_chunked(df, model_columns, chunk_size=1000, **kwargs):
    """
    Divide and conquer version of vectorized_lstsq_numpy.
    Splits the dataframe into chunks of groups to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    model_columns : list of str
        Columns to use as models.
    chunk_size : int, optional
        Number of groups per chunk. Default is 1000.
    **kwargs : dict
        Additional arguments passed to vectorized_lstsq_numpy.
    """
    group_column = kwargs.get("group_column", "tmpl_ind")
    return_error = kwargs.get("return_error", False)

    # Sort by group_column to allow efficient slicing
    df_sorted = df.sort_values(group_column)
    
    # Get unique groups and their counts in sorted order
    unique_groups, group_counts = np.unique(df_sorted[group_column].values, return_counts=True)
    
    # Calculate cumulative indices for slicing
    cum_indices = np.cumsum(np.insert(group_counts, 0, 0))

    results = []
    for i in range(0, len(unique_groups), chunk_size):
        start_idx = cum_indices[i]
        end_idx = cum_indices[min(i + chunk_size, len(unique_groups))]
        
        df_chunk = df_sorted.iloc[start_idx:end_idx]
        res = vectorized_lstsq_numpy(df_chunk, model_columns, **kwargs)
        results.append(res)

    if not results:
        if return_error:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    if return_error:
        coeffs = np.concatenate([r[0] for r in results], axis=0)
        errs = np.concatenate([r[1] for r in results], axis=0)
        indices = np.concatenate([r[2] for r in results], axis=0)
        return coeffs, errs, indices
    else:
        coeffs = np.concatenate([r[0] for r in results], axis=0)
        indices = np.concatenate([r[1] for r in results], axis=0)
        return coeffs, indices
