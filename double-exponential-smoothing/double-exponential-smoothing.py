def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    if not series or len(series) < 2:
        return []

    level = float(series[0])
    trend = float(series[1] - series[0])
    
    result = [level]

    for t in range(1, len(series)):
        value = float(series[t])
        old_level = level
        
        level = alpha * value + (1 - alpha) * (old_level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        
        result.append(level)

    return result
