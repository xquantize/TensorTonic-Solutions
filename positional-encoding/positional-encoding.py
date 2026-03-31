import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # codes
    # out matrix shape (seq_len, d_model)
    pe = np.zeros((seq_len, d_model), dtype=float)

    # column vector of the positions
    pos = np.arange(seq_len).reshape(-1, 1)

    # row vector of dimension indicies i = 0 ......
    # one frequency per pair of colms
    half_ceil = (d_model + 1) // 2
    half_floor = d_model // 2

    i = np.arange(half_ceil, dtype=float).reshape(1, -1)
    divisor = np.power(base, (2 * i) / d_model)

    # angles via broadcasting
    angles = pos / divisor

    # fill alternating columns
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :half_floor])

    return pe
