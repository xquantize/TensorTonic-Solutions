import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)
    
    if not (0 <= p <= 1):
        raise ValueError("probability p must be between 0 and 1")
        
    # P(X = x) = p^x * (1-p)^(1-x) for x in {0, 1}, else 0
    pmf_values = np.where((x == 0) | (x == 1), (p ** x) * ((1 - p) ** (1 - x)), 0.0)
    
    if pmf_values.ndim == 0:
        pmf_values = float(pmf_values)
        
    mean = float(p)
    variance = float(p * (1 - p))
    
    return pmf_values, mean, variance
