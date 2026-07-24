import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    def single_pmf(i):
        if p == 0:
            return 1.0 if i == 0 else 0.0
        if p == 1:
            return 1.0 if i == n else 0.0

        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    pmf = single_pmf(k)
    cdf = sum(single_pmf(i) for i in range(k + 1))

    return float(pmf), float(cdf)
