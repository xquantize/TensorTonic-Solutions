import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # pairwise squared diff via broad
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    # sum of squares over feature dim then sqrt
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    sorted_idx = np.argsort(dist, axis=1)

    k_eff = min(k, n_train)
    top_k = sorted_idx[:, :k_eff]

    if k_eff < k:
        pad_width = k - k_eff
        pad = np.full((n_test, pad_width), -1, dtype=top_k.dtype)
        result = np.concatenate([top_k, pad], axis=1)
    else:
        result = top_k

    return result.astype(int)
