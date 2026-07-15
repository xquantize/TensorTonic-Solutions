import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = scores.shape[-1]

    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    out = np.where(future_mask, mask_value, scores)

    return out
