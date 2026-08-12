def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    result = []
    running_product = 1.0

    for r in returns:
        running_product *= (1.0 + r)
        result.append(running_product - 1.0)

    return result
