def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    max_lag = max(lags)
    n = len(series)    

    matrix = []

    for t in range(max_lag, n):
        row = [series[t - lag] for lag in lags]
        matrix.append(row)

    return matrix
