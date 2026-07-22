import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # if y=1, p_t = p if y=0, p_t = 1-p
    p_t = np.where(targets == 1, predictions, 1 - predictions)

    focal_loss = -alpha * (1 - p_t) ** gamma * np.log(p_t)

    return float(np.mean(focal_loss))
