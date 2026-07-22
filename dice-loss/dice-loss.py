import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    intersection = np.sum(p * y)
    total = np.sum(p) + np.sum(y)

    dice_coeff = (2 * intersection + eps) / (total + eps)
    loss = 1 - dice_coeff

    return float(loss)
