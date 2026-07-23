import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y)

    if num_classes is None:
        num_classes = int(y.max()) + 1

    if np.any(y >= num_classes):
        raise ValueError("all labels must be less than num classes")

    N = y.shape[0]

    one_hot_matrix = np.zeros((N, num_classes), dtype=np.float64)
    one_hot_matrix[np.arange(N), y] = 1.0

    return one_hot_matrix
