import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    X = np.asarray(X, dtype=np.float64)
    h = np.asarray(h_0, dtype=np.float64)

    W_xh = np.asarray(W_xh, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)

    b_h = np.asarray(b_h, dtype=np.float64)

    T = X.shape[1]
    hidden_list = []

    for t in range(T):
        x_t = X[:, t, :]
        h = np.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)
        hidden_list.append(h)

    hidden_states = np.stack(hidden_list, axis=1)
    h_final = h

    return hidden_states, h_final
