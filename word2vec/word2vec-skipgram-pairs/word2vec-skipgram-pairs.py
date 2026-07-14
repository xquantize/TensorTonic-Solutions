import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    n = token_ids.numel()
    pairs = []

    for i in range(n):
        start = max(0, i - window)
        end = min(n - 1, i + window)

        for j in range(start, end + 1):
            if j == i:
                continue

            pairs.append([token_ids[i].item(), token_ids[j].item()])

    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64)

    output = torch.tensor(pairs, dtype=torch.int64)

    return output
