def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    n = len(values)
    result_len = n - window_size + 1

    if result_len <= 0:
        return []

    window = sorted(values[:window_size])
    result = [0.0] * result_len

    def get_medium(w, size):
        mid = size // 2

        if size % 2 == 1:
            return float(w[mid])
        else:
            return (w[mid - 1] + w[mid]) / 2.0

    result[0] = get_medium(window, window_size)

    import bisect

    for i in range(1, result_len):
        leaving = values[i - 1]
        idx = bisect.bisect_left(window, leaving)
        window.pop(idx)

        entering = values[i + window_size - 1]
        bisect.insort(window, entering)

        result[i] = get_medium(window, window_size)

    return result
        