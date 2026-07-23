import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    W_hh = np.asarray(W_hh, dtype=np.float64)

    spectral_norm = np.linalg.norm(W_hh, ord=2)
    norms = []
    grad_norm = 1.0

    for _ in range(T):
        norms.append(grad_norm)
        grad_norm *= spectral_norm

    return norms
