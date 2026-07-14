import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    counts_float = counts.to(torch.float32)
    total_count = counts_float.sum()

    freqs = counts_float / total_count
    keep_probs = torch.sqrt(t / freqs)

    output = torch.clamp(keep_probs, max=1.0)

    return output
    