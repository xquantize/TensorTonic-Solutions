import pandas as pd

def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """    
    if isinstance(series, pd.Series):
        return series.pct_change().tolist()
        
    series = list(series)
    if len(series) == 0:
        return []
        
    result = []

    for i in range(1, len(series)):
        prev = series[i - 1]
        curr = series[i]
        
        if prev is None or curr is None:
            result.append(float('nan'))
        elif prev == 0:
            result.append(0.0)
        else:
            result.append((curr - prev) / prev)
            
    return result
