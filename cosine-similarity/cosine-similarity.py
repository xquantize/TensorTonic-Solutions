import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a ==0 or norm_b == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(a, b) / (norm_a * norm_b))

    return similarity
