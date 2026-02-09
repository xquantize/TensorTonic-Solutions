import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # codes
    # init pe matrix with zeros
    pe = np.zeros((seq_length, d_model))

    # create column vector of positions (seq_length, 1)
    position = np.arange(seq_length).reshape(-1, 1)

    # division term, frequencies
    # for i up to d_model/2
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # apply to even ind
    pe[:, 0::2] = np.sin(position * div_term)

    # apply to odd ind
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
