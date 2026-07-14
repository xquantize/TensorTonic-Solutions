import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("must be same shape")

    if y_true.size == 0:
        K = num_classes if num_classes is not None else 0
        if normalize == 'none':
            return np.zeros((K, K), dtype=int)
        return np.zeros((K, K), dtype=float)

    if num_classes is None:
        K = int(max(y_true.max(), y_pred.max())) + 1
    else:
        K = num_classes

    if y_true.min() < 0 or y_true.max() >= K:
        raise ValueError(f"y_true contains labels outside [0, {K-1}]")
    if y_pred.min() < 0 or y_pred.max() >= K:
        raise ValueError(f"y_pred contains labels outside [0, {K-1}]")

    indices = y_true * K + y_pred
    counts = np.bincount(indices, minlength=K * K)
    cm = counts.reshape(K, K).astype(np.int64)

    if normalize == 'none':
        return cm

    cm = cm.astype(np.float64)
    eps = 1e-12

    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / (row_sums + eps)
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm = cm / (col_sums + eps)
    elif normalize == 'all':
        total = cm.sum()
        total = total if total !=0 else 1
        cm = cm / (total + eps)
    else:
        raise ValueError("normalize tag not found")

    return cm
