import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    P = patch_size

    H_p = H // P
    W_p = W // P
    N = H_p * W_p
    patch_dim = P * P * C

    # (B, H, W, C) -> (B, H_p, P, W_p, P, C)
    x = image.reshape(B, H_p, P, W_p, P, C)

    # (B, H_p, P, W_p, P, C) -> (B, H_p, W_p, P, P, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)

    # (B, H_p, W_p, P, P, C) -> (B, N, P * P * C)
    patches = x.reshape(B, N, patch_dim)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    # (B, N, patch_dim) @ (patch_dim, embed_dim) -> (B, N, embed_dim)
    embeddings = patches @ W_proj

    return embeddings
    