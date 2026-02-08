import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        # codes
        # save input for the skip connection
        identity = x

        # W1 @ x followed by ReLU is the first transformation
        out = x @ self.W1.T
        out = relu(out)

        # then W2 @ out is the second transformation
        out = out @ self.W2.T

        # add the identity
        out = relu(out) + identity

        return out
