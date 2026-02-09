import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # codes
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # linear projections
    # (batch, seq_len, d_model) @ (d_model, d_model) - (batch, seq_len, d_model)
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # into heads
    # shape into (batch, seq_len, num_heads, d_k)
    # transpose - (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    # scaled dot product atten per head
    # scale shape - (batch, num_heads, seq_len, seq_len)
    # transpose last two dims of K_heads for matmul
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)

    # atten output (batch, num_heads, seq_len, d_k)
    attention_out = np.matmul(weights, V_heads)

    # concat heads
    # transpose back - (batch, seq_len, num_heads. d_k)
    # reshape concat (batch, seq_len, d_model)
    concat_out = attention_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # output projection
    # (batch, seq_len, d_model) @ (d_model, d_model)
    output = np.matmul(concat_out, W_o)

    return output
