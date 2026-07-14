import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    pos_score = torch.dot(center_vec, pos_vec)
    neg_scores = torch.mv(neg_vecs, center_vec)

    pos_loss = F.softplus(-pos_score)
    neg_loss = F.softplus(neg_scores).sum()

    sgns_loss = pos_loss + neg_loss

    return sgns_loss
