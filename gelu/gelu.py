import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asarray(x, dtype=np.float64)

    erf_vec = np.vectorize(math.erf)

    out = 0.5 * x * (1 + erf_vec(x / np.sqrt(2.0)))

    return out
