import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    D = Wz.shape[0]
    H = Uz.shape[0]

    x2d, x_was_1d = _as2d(x, D)
    h2d, h_was_1d = _as2d(h_prev, H)

    # update gate z_t = sigmoid(x@Wz + h_prev@Uz + bz)
    z = _sigmoid(x2d @ Wz + h2d @ Uz + bz)

    # reset gate r_t = sigmoid(x@Wr + h_prev@Ur + br)
    r = _sigmoid(x2d @ Wr + h2d @ Ur + br)

    # hidden state h_tilde = tanh(x@Wh + (r * h_prev)@Uh + bh)
    h_tilde = np.tanh(x2d @ Wh + (r * h2d) @ Uh + bh)

    # new hidden state: h_t = (1-z)*h_prev + z*h_tilde
    h_new = (1 - z) * h2d + z * h_tilde

    if x_was_1d and h_was_1d:
        return h_new.reshape(H)

    return h_new
