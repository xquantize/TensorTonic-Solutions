import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    if strategy not in ('mean' , 'median'):
        raise ValueError('eh wrong')

    X = np.asarray(X, dtype=np.float64).copy()

    was_1d = (X.ndim == 1)
    if was_1d:
        X = X.reshape(-1, 1)

    nan_mask = np.isnan(X)

    # da da da da
    with np.errstate(invalid='ignore'):
        if strategy == 'mean':
            col_stat = np.nanmean(X, axis=0)
        else:
            col_stat = np.nanmedian(X, axis=0)

    all_nan_cols = np.isnan(col_stat)
    col_stat = np.where(all_nan_cols, 0.0, col_stat)

    row_idx, col_idx = np.where(nan_mask)
    X[row_idx, col_idx] = col_stat[col_idx]

    if was_1d:
        X = X.ravel()

    return X
