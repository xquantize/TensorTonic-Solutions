def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    n = len(values)
    k = window_size

    result = []
    window_sum = sum(values[:k])
    result.append(window_sum / k)

    for i in range(1, n - k + 1):
        window_sum += values[i + k - 1] - values[i - 1]
        result.append(window_sum / k)

    return result
