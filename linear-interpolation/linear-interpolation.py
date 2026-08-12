def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    n = len(values)
    result = list(values)

    i = 0

    while i < n:
        if result[i] is None:
            start = i - 1

            while i < n and result[i] is None:
                i += 1
            end = i

            num_steps = end - start
            start_val = result[start]
            end_val = result[end]

            for step in range(1, num_steps):
                fraction = step / num_steps
                result[start + step] = start_val + fraction * (end_val - start_val)

        else:
            i += 1

    return result
    