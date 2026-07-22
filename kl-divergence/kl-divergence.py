import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    q_safe = q + eps

    terms = np.where(p > 0, p * np.log(p / q_safe), 0.0)

    return float(np.sum(terms))
