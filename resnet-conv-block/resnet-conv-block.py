import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        # codes
        # path f(x), first transform linear -> relu 
        # (batch, in_channels)  @ (in_channels, out_channels) -> (batch, out_channels)
        h = x @ self.W1
        h = relu(h)

        # second transform is linear
        # (batch, out_channels)  @ (out_channels, out_channels) -> (batch, out_channels)
        z = h @ self.W2

        # third is the shortcut Ws @ x , the projection
        # transform x from in_channels to out_channels to add to z
        # (batch, in_channels)  @ (in_channels, out_channels) -> (batch, out_channels)
        s = x @ self.Ws

        # y = relu(z + s)
        # both z and s are (batch, out_channels)
        out = relu(z + s)

        return out
