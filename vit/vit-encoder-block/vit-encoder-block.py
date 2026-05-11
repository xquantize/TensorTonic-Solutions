import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    normalized = (x - mean) / (std + eps)
    return normalized

def gelu(x: np.ndarray) -> np.ndarray:
    gelu_val = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    return gelu_val

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    softmax_val = e_x / e_x.sum(axis=axis, keepdims=True)
    return softmax_val

def multi_head_self_attention(x: np.ndarray, Wq, Wk, Wv, Wo, num_heads: int) -> np.ndarray:
    B, N, D = x.shape
    head_dim = D // num_heads

    # Q, K, V projection, (B, N, D)
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # split heads
    # (B, N, D) -> (B, N, H, head_dim) -> (B, H, N, N)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # scaled dot per head
    # (B, H, N, head_dim) @ (B, H, head_dim, N) -> (B, H, N, N)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    attn = softmax(scores, axis=-1)

    # apply attn
    # (B, H, N, N) @ (B, H, N, head_dim) -> (B, H, N, head_dim)
    out = attn @ V

    # concat heads
    # (B, H, N, head_dim) -> (B, N, H, head_dim) -> (B, N, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

    out_proj = out @ Wo

    return out_proj

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    D = embed_dim
    hidden_dim = int(D * mlp_ratio)

    # init weights
    if Wq is None:
        Wq = np.random.randn(D, D) * 0.02
    if Wk is None:
        Wk = np.random.randn(D, D) * 0.02
    if Wv is None:
        Wv = np.random.randn(D, D) * 0.02
    if Wo is None:
        Wo = np.random.randn(D, D) * 0.02
    if W1 is None:
        W1 = np.random.randn(D, hidden_dim) * 0.02
    if W2 is None:
        W2 = np.random.randn(hidden_dim, D) * 0.02

    x_norm = layer_norm(x)

    attn_out = multi_head_self_attention(x_norm, Wq, Wk, Wv, Wo, num_heads)
    x = x + attn_out

    x_norm = layer_norm(x)
    mlp_out = gelu(x_norm @ W1) @ W2
    x = x + mlp_out

    return x
