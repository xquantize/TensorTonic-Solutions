import numpy as np

def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    norm = (x - mean) / (std + eps)
    return norm

def _gelu(x: np.ndarray) -> np.ndarray:
    gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    return gelu

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    softmax = e_x / e_x.sum(axis=axis, keepdims=True)
    return softmax

def _as_array(v):
    return np.asarray(v, dtype=float) if v is not None else None

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        D = embed_dim
        P = patch_size
        N = self.num_patches
        hidden_dim = int(D * mlp_ratio)

        W_patch = _as_array(W_patch)
        cls_token = _as_array(cls_token)
        pos_embed = _as_array(pos_embed)
        W_head = _as_array(W_head)
        
        patch_dim = P * P * 3

        if W_patch is not None:
            patch_dim = W_patch.shape[0]

        if W_patch is not None:
            self.W_patch = W_patch
        else:
            self.W_patch = np.random.randn(patch_dim, D) * 0.02

        if cls_token is not None:
            self.cls_token = cls_token
        else:
            self.cls_token = np.random.randn(1, 1, D) * 0.02

        if pos_embed is not None:
            self.pos_embed = pos_embed
        else:
            self.pos_embed = np.random.randn(1, N + 1, D) * 0.02

        self.encoder_weights = []

        for layer_idx in range(depth):
            if encoder_weights is not None and layer_idx < len(encoder_weights):
                self.encoder_weights.append(encoder_weights[layer_idx])
            else:
                self.encoder_weights.append({
                    "Wq": np.random.randn(D, D) * 0.02,
                    "Wk": np.random.randn(D, D) * 0.02,
                    "Wv": np.random.randn(D, D) * 0.02,
                    "Wo": np.random.randn(D, D) * 0.02,
                    "W1": np.random.randn(D, hidden_dim) * 0.02,
                    "W2": np.random.randn(hidden_dim, D) * 0.02,
                })

        if W_head is not None:
            self.W_head = W_head
        else:
            self.W_head = np.random.randn(D, num_classes) * 0.02

    def _patch_embed(self, image: np.ndarray) -> np.ndarray:
        B, H, W, C = image.shape
        P = self.patch_size
        H_p, W_p = H // P, W // P
        N = H_p * W_p
        patch_dim = P * P * C

        x = image.reshape(B, H_p, P, W_p, P, C).transpose(0, 1, 3, 2, 4, 5)
        patches = x.reshape(B, N, patch_dim)
        patch_embed = patches @ self.W_patch
        return patch_embed

    def _prepend_cls(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        cls = np.tile(self.cls_token, (B, 1, 1))
        out = np.concatenate([cls, x], axis=1)
        return out

    def _msa(self, x: np.ndarray, w) -> np.ndarray:
        B, N, D = x.shape
        H = self.num_heads
        head_dim = D // H

        Q = (x @ w["Wq"]).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        K = (x @ w["Wk"]).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        V = (x @ w["Wv"]).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        attn = _softmax(scores, axis=-1)
        attn_ap = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
        out = attn_ap @ w["Wo"]
        return out

    def _encoder_block(self, x: np.ndarray, w) -> np.ndarray:
        x = x + self._msa(_layer_norm(x), w)
        x = x + _gelu(_layer_norm(x) @ w["W1"]) @ w["W2"]
        return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        z = self._patch_embed(x)
        z = self._prepend_cls(z)
        z = z + self.pos_embed

        for w in self.encoder_weights:
            z = self._encoder_block(z, w)

        cls_final = z[:, 0]
        cls_normed = _layer_norm(cls_final)

        out = cls_normed @ self.W_head
        return out
