def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    n = len(values)
    
    if window_size == 1:
        return [0.0] * n

    result = []
    
    for i in range(n - window_size + 1):
        window = values[i:i + window_size]
        mean = sum(window) / window_size
        variance = sum((x - mean) ** 2 for x in window) / window_size
        result.append(math.sqrt(variance))
        
    return result
