import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    distance = float(np.sum(np.abs(x - y)))
    return distance
