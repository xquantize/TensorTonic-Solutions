import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)

    if n == 0:
        return 0.0

    matches = (y_true == y_pred)
    tp_total = int(np.sum(matches))

    mismatches = n - tp_total
    fp_total = mismatches
    fn_total = mismatches

    denom = 2 * tp_total + fp_total + fn_total

    if denom == 0:
        return 0.0

    f1 = (2 * tp_total) / denom

    return float(f1)
