import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.asarray(C, dtype=np.float64)

    row_totals = np.sum(C, axis=1, keepdims=True)
    col_totals = np.sum(C, axis=0, keepdims=True)

    grand_total = np.sum(C)

    expected = (row_totals @ col_totals) / grand_total

    chi2 = np.sum((C - expected) ** 2 / expected)

    return float(chi2), expected
