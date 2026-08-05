import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    mean = np.mean(series)
    
    centered = series - mean
    c0 = np.sum(centered ** 2)

    if c0 == 0:
        return [1.0] + [0.0] * max_lag

    acf = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            if lag >= n:
                acf.append(0.0)
            else:
                numerator = np.sum(centered[lag:] * centered[:-lag])
                acf.append(float(numerator / c0))

    return acf
