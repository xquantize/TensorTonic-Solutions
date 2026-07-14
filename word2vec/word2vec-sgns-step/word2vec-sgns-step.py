import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    W_in_up = W_in.clone()
    W_out_up = W_out.clone()

    v_c = W_in[center_id].clone()
    u_pos = W_out[pos_id].clone()
    u_negs = W_out[neg_ids].clone()

    score_pos = torch.dot(v_c, u_pos)
    sig_pos = torch.sigmoid(score_pos)

    score_negs = torch.mv(u_negs, v_c) if neg_ids.numel() > 0 else torch.empty(0, dtype=W_in.dtype)

    sig_negs = torch.sigmoid(score_negs)

    grad_vc = (sig_pos - 1.0) * u_pos

    if neg_ids.numel() > 0:
        grad_vc += torch.mv(u_negs.t(), sig_negs)

    grad_upos = (sig_pos - 1.0) * v_c

    W_in_up[center_id] -= lr * grad_vc
    W_out_up[pos_id] -= lr * grad_upos

    for i, neg_idx in enumerate(neg_ids.tolist()):
        grad_un_i = sig_negs[i] * v_c
        W_out_up[neg_idx] -= lr * grad_un_i

    return W_in_up, W_out_up
