def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []
    
    for v in x:
        if v > 0:
            result.append(float(v))
        else:
            result.append(alpha * (math.exp(v) - 1))

    return result
    