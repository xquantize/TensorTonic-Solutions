import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        # codes
        if training:
            # we calc the mean and varience across the bacth axis 0
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # update the running state for inferecne
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            mean = self.running_mean
            var = self.running_var

        # normalise so (x - mean) / sqrt(var + eps)
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        # scale and shift w the learnable params gamma and beta
        return self.gamma * x_hat + self.beta

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    # codies
    identity = x

    # path is conv - bn - relu
    out = x @ W1
    out = bn1.forward(out, training=True)
    out = relu(out)

    # path conv - bn
    out = out @ W2
    out = bn2.forward(out, training=True)

    # add the idenityt then relu
    out = relu(out + identity)

    return out

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    # codes
    identity = x

    # path bn - relu - conv
    out = bn1.forward(x, training=True)
    out = relu(out)
    out = out @ W1

    # path bn - relu - conv
    out = bn2.forward(out, training=True)
    out = relu(out)
    out = out @ W2

    # no relu, skip connection is clean

    out = out + identity

    return out
