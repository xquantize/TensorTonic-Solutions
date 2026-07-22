def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    limit = math.sqrt(6 / (fan_in + fan_out))

    scaled = [
        [w * 2 * limit - limit for w in row]
        for row in W
    ]

    return scaled
