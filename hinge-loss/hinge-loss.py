import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape != y_score.shape:
        raise ValueError("must have same shape")
    if y_true.ndim != 1:
        raise ValueError("must be 1d array")

    unique_labels = np.unique(y_true)

    if not np.all(np.isin(unique_labels, [-1.0, 1.0])):
        raise ValueError("must contain -1 and +1")

    losses = np.maximum(0.0, margin - y_true * y_score)

    if reduction == "mean":
        result = np.mean(losses)
    elif reduction == "sum":
        result = np.sum(losses)
    else:
        raise ValueError("unknown mode")

    return float(result)
