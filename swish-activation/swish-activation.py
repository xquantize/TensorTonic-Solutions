import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    def stable_sigmoid(z):
        out = np.empty_like(z)
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[neg_mask])
        out[neg_mask] = exp_z / (1.0 + exp_z)

        return out

    sig = stable_sigmoid(x)
    out = x * sig

    return out
