import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    unique_labels = np.unique(y)

    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError("must contain 0 and 1")

    diff = a - b
    d = np.sqrt(np.sum(diff ** 2, axis=-1))

    # y * d^2 + (1-y) * max(0, margin - d)^2
    positive_term = y * d ** 2
    negative_term = (1 - y) * np.maximum(0.0, margin - d) ** 2
    losses = positive_term + negative_term

    if reduction == "mean":
        result = np.mean(losses)
    elif reduction == "sum":
        result = np.sum(losses)
    else:
        raise ValueError("unknwon mode")

    return float(result)
