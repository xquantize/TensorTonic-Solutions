import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # codes
    x = np.asarray(x, dtype=float)

    rand = (rng.random(x.shape) if rng is not None
           else np.random.random(x.shape))

    # where we keep pro 1-p, 0 where we drop
    keep_mask = (rand < (1 - p)).astype(float)

    # scale kept elements by 1/(1-p) to presever epec value
    pattern = keep_mask / (1 - p)
    
    return x * pattern, pattern
    