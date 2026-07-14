import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    counts_double = torch.as_tensor(counts, dtype=torch.float64)
    smoothed_counts = counts_double ** alpha

    probabilities = smoothed_counts / smoothed_counts.sum()

    return probabilities
    