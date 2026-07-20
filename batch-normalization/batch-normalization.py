import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if x.ndim == 2:
        # (N, D) normalize each feature column over the batch axis 0
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        gamma_r = gamma.reshape(1, -1)
        beta_r = beta.reshape(1, -1)

        y = gamma_r * x_hat + beta_r

    elif x.ndim == 4:
        # (N, C, H, W) normalize each channel over axis (0, 2, 3)
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        C = x.shape[1]
        gamma_r = gamma.reshape(1, C, 1, 1)
        beta_r = beta.reshape(1, C, 1, 1)

        y = gamma_r * x_hat + beta_r

    else:
        raise ValueError('expected shape (N,D) or (N,C,H,W)')

    return y
