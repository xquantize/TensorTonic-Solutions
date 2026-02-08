import numpy as np

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # codes
    # unit gradient from loss dL/dy = 1
    # apply chain rule, multiply jacobians so jacobians @ vectors
    # handles (2,) vs (2, 1) vs (1, 2)
    original_shape = x.shape
    grad = x.reshape(-1, 1)

    # interate backwards from the last layer to the first
    for dF in reversed(gradients_F):
        grad = dF @ grad

    # rehsape back to input passed
    return grad.reshape(original_shape)


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # codies
    # grad = grad * (jacobians + I)
    # + I is the derivative of the identity skip connections
    original_shape = x.shape
    grad = x.reshape(-1, 1)

    # identity matrix of correct size
    dims = gradients_F[0].shape[0]
    I = np.eye(dims)

    for dF in reversed(gradients_F):
        # jacobian + identity then @ column vector
        out = dF + I
        grad = out @ grad

    # rehsape back to input passed
    return grad.reshape(original_shape)
