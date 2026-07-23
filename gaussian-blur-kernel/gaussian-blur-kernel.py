def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    center = size // 2
    kernel = []
    total = 0.0

    for i in range(size):
        row = []

        for j in range(size):
            x = j - center
            y = i - center

            weight = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            row.append(weight)
            total += weight
        kernel.append(row)

    normalized = [[w / total for w in row] for row in kernel]

    return normalized
    