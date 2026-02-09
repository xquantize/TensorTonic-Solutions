import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # codes
    # W1 first weight matrix
    # b1 first bias vector
    # W2 second weight matrix
    # b2 second bias vector

    # first linear transformation for expansion
    # projects d_model to higher dim d_ff
    # so (batch, seq, d_model) @ (d_model, d_ff) + b1 gives (batch, seq, d_ff)
    hidden = np.dot(x, W1) + b1

    # do relu activation
    # introduce non linearity to model
    # bloew 0 is blocked and above 0 passes through
    relu_out = np.maximum(0, hidden)

    # second linear for contraction
    # project back from d_ff to d_model
    # (batch, seq, d_ff) @ (d_ff, d_model) + b2 gives (batch, seq, d_model)
    output = np.dot(relu_out, W2) + b2

    return output
    