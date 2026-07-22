def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    limit = math.sqrt(6 / fan_in)

    scaled = [
        [w * 2 * limit - limit for w in row]
        for row in W
    ]

    return scaled
