import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    N, C_in, H, Wd = x.shape
    C_out, C_in_w, KH, KW = W.shape

    windows = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2,3 ))
    y = np.einsum('ncijuv,ocuv->noij', windows, W)
    y = y + b.reshape(1, C_out, 1, 1)

    return y
