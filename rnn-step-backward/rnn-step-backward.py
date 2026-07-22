import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t, h_prev, h_t, W, U, b = cache

    dh = np.asarray(dh, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)

    h_prev = np.asarray(h_prev, dtype=np.float64)
    h_t = np.asarray(h_t, dtype=np.float64)

    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    dz = dh * (1 - h_t ** 2)
    db = dz

    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)

    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    return dx_t, dh_prev, dW, dU, db
