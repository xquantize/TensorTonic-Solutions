import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    C = (X_centered.T @ X_centered) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(C)

    order = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, order]

    W = eigvecs_sorted[:, :k]
    X_proj = X_centered @ W

    return X_proj.tolist()
