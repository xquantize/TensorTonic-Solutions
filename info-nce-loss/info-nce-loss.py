import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)

    N = Z1.shape[0]

    # S[i,j] = (Z1[i] . Z2[j]) / temperature
    # Z1 @ Z2.T -> (N, D) @ (D, N) = (N, N)
    S = (Z1 @ Z2.T) / temperature

    # log(exp(S_ii) / sum_j exp(S_ij))
    # = S_ii - log(sum_j exp(S_ij))
    row_max = np.max(S, axis=1, keepdims=True)
    S_shifted = S - row_max

    log_sum_exp = row_max.squeeze(-1) + np.log(np.sum(np.exp(S_shifted), axis=1))
    diag = np.diagonal(S)

    per_sample_loss = -(diag - log_sum_exp)

    output = float(np.mean(per_sample_loss))

    return output
