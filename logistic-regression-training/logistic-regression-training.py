import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # codes
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    # params init
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # forward pass
        # (N, D).(D,) + scalar
        z = X @ w + b
        # (N,) predicted probabilities
        p = _sigmoid(z)

        # error signal (p-y) both gradients
        error = p - y

        # gradients averange n samples
        grad_w = (X.T @ error) / N
        grad_b = error.mean()

        # gradient decent update
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b
