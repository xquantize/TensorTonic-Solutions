import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if x.shape != p.shape:
        raise ValueError("must have same shape")

    if not np.allclose(p.sum(), 1.0, atol=1e-6):
        raise ValueError("must sum to 1")

    probs = float(np.sum(x * p))

    return probs
