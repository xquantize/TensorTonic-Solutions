import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    positive_term = (1 - p) ** gamma * y * np.log(p)
    negative_term = p ** gamma * (1 - y) * np.log(1 - p)

    loss = -(positive_term + negative_term)

    return float(np.mean(loss))
