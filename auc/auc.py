import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)

    if fpr.shape != tpr.shape:
        raise ValueError("must have same shapes")
    if fpr.size < 2:
        raise ValueError("must have atleast 2 points")

    area = np.trapezoid(tpr, fpr)

    return float(area)
