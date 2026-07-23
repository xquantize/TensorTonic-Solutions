import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    dh_next = np.asarray(dh_next, dtype=np.float64)
    h_t = np.asarray(h_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)

    # tanh: dtanh = (1 - h_t^2) * dh_next
    dtanh = (1 - h_t ** 2) * dh_next
    
    #W_hh: dtanh.T @ h_prev
    dW_hh = np.dot(dtanh.T, h_prev)

    #dtanh @ W_hh
    dh_prev = np.dot(dtanh, W_hh)

    return dh_prev, dW_hh
    