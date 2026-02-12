import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # codes
    # calc mean
    mean = np.mean(x, axis=-1, keepdims=True)
    # calc the variance
    variance = np.var(x, axis=-1, keepdims=True)
    # calc the x-hat (secondary comp)
    x_hat = (x - mean) / (np.sqrt(variance + eps))
    # final calc
    out = (gamma * x_hat) + beta

    return out

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # get batchsize, seqlen, dmodel
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    
    # reshape and transpose for multi-head
    # (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        return x.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    Q_heads, K_heads, V_heads = split_heads(Q_proj), split_heads(K_proj), split_heads(V_proj)
    
    # scaled dot product attention
    # scores (batch, num_heads, seq_len, seq_len)
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores)
    attention_output = weights @ V_heads
    
    # concatenate and project back
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return attention_output @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # codes
    # first linear plus relu
    inter = np.maximum(0, x @ W1 + b1)
    #second linear
    out = inter @ W2 + b2

    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # codes
    # sub layer 1
    # multi head att + residual + layernorm
    atten_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_inter = layer_norm(x + atten_out, gamma1, beta1)

    # sub layer 2
    # feed forward + residual + layernorm
    ffn_out = feed_forward(x_inter, W1, b1, W2, b2)
    output = layer_norm(x_inter + ffn_out, gamma2, beta2)

    return output
