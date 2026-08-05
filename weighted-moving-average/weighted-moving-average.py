import numpy as np

def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n = len(values)
    k = len(weights)
    
    if k == 0 or n < k:
        return []
        
    weight_sum = np.sum(weights)

    if weight_sum == 0:
        return [0.0] * (n - k + 1)
        
    weights = weights / weight_sum
    wma = np.convolve(values, weights[::-1], mode='valid')
    
    return [float(x) for x in wma]
