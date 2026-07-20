import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 3:
        # (C, H, W) -> average over H, W (axes 1, 2) -> (C,)
        return np.mean(x, axis=(1, 2))
    elif x.ndim == 4:
        # (N, C, H, W) -> average over H, W (axes 2, 3) -> (N, C)
        return np.mean(x, axis=(2, 3))
    else:
        raise ValueError('expected shape (C,H,W) or (N,C,H,W)')
