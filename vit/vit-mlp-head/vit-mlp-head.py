import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # (B, N, D) -> (B, D)
    cls_token = encoder_output[:, 0]
    embed_dim = cls_token.shape[-1]

    # layer norm
    mean = cls_token.mean(axis=-1, keepdims=True)
    std = cls_token.std(axis=-1, keepdims=True)
    cls_normed = (cls_token - mean) / (std + 1e-6)

    # init proj matrix
    if W_head is None:
        W_head = np.random.randn(embed_dim, num_classes) * 0.02

    # linear proj to class digits
    # (B, D) @ (D, C) -> (B, C)
    logits = cls_normed @  W_head

    return logits
