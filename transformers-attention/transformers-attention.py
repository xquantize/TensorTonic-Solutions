import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # codes
    # get dims of keys for scale
    d_k = Q.size(-1)

    # compute atten scores - Q @ K^T
    # k.transpose(-2, -1) swaps last tow dims to align for matrix multip
    # scores shapes - (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # compute atten weight using softmax
    # softmax across key dim
    weights = F.softmax(scores, dim=-1)

    # weighted sum of values
    # output - (batch, seq_len_q, d_v)
    output = torch.matmul(weights, V) 

    return output
