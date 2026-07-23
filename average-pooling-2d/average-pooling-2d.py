def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    H = len(X)
    W = len(X[0])

    p = pool_size

    H_out = H // p
    W_out = W // p

    output = []

    for i in range(H_out):
        row = []

        for j in range(W_out):
            total = sum(
                X[i * p + a][j * p + b]
                for a in range(p)
                for b in range(p)
            )
            row.append(total / (p * p))

        output.append(row)

    return output
