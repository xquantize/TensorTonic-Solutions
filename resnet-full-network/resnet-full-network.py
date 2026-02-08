import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Conv -> ReLU -> Conv -> Add Skip -> ReLU
        """
        # codies
        identity = x

        # conv - relu - conv
        out = x @ self.W1
        out = relu(out)
        out = out @ self.W2

        # shortcut, identity of projection
        if self.W_proj is not None:
            identity = x @ self.W_proj

        # final relu
        out = out + identity
        out = relu(out)

        return out

class ResNet18:
    """
    Simplified ResNet-18 architecture.
    
    Structure:
    - conv1: 3 -> 64 channels
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels (first block downsamples)
    - layer3: 2 BasicBlocks, 256 channels (first block downsamples)
    - layer4: 2 BasicBlocks, 512 channels (first block downsamples)
    - fc: 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10):
        self.conv1_W = np.random.randn(3, 64) * 0.01
        
        # Build layers
        # self.layer1 = None  # 2 blocks: 64 -> 64
        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        # self.layer2 = None  # 2 blocks: 64 -> 128 (first downsamples)
        self.layer2 = [BasicBlock(64, 128, downsample=True), BasicBlock(128, 128)]
        # self.layer3 = None  # 2 blocks: 128 -> 256 (first downsamples)
        self.layer3 = [BasicBlock(128, 256, downsample=True), BasicBlock(256, 256)]
        # self.layer4 = None  # 2 blocks: 256 -> 512 (first downsamples)
        self.layer4 = [BasicBlock(256, 512, downsample=True), BasicBlock(512, 512)]
        
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.
        """
        # codies
        out = x @ self.conv1_W
        out = relu(out)

        # pass through the 4 stages (8 basic blocks)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out = block.forward(out)

        # global average pooling, mean across spatial if x were 4d
        # assume x is batch, channels

        out = out @ self.fc 

        return out
