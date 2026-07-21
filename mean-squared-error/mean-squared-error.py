import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if y_pred.shape != y_true.shape:
        return None

    mse = np.mean((y_pred - y_true) ** 2)

    return float(mse)
