import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # codes
    # check x is numpy array
    X = np.asanyarray(X, dtype=float)

    # calc min and max along axis
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    # calc range (denominator)
    diff = x_max - x_min

    # avoid div by zero
    # if max == min range = eps
    denom = np.maximum(diff, eps)

    # top part of form
    top = X - x_min

    # apply form
    X_scaled = top / denom
    
    return X_scaled
    