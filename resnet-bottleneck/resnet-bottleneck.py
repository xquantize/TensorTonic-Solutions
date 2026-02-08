import numpy as np

def relu(x):
    return np.maximum(0, x)

class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels  # Compressed dimension
        self.out_ch = out_channels
        
        # 1x1 reduce
        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        # 3x3 (simplified as dense)
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        # 1x1 expand
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01
        
        # Shortcut (if dimensions differ)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Bottleneck forward: compress -> process -> expand + skip
        """
        # codes
        # save the identity
        identity = x

        # first layer compress channels
        # (Batch, In_Ch) @ (In_Ch, BN_Ch) -> (Batch, BN_Ch)
        out = x @ self.W1
        out = relu(out)

        # second apply the 3x3 convolution
        # (Batch, BN_Ch) @ (BN_Ch, BN_Ch) -> (Batch, BN_Ch)
        out = out @ self.W2
        out = relu(out)

        # thirdly is expands the channels back
        # (Batch, BN_Ch) @ (BN_Ch, Out_Ch) -> (Batch, Out_Ch)
        out = out @ self.W3

        # handle the skip connection
        # in_channels != out_channels, we use Ws projection
        if self.Ws is not None:
            identity = x @ self.Ws

        out = out + identity

        return out
