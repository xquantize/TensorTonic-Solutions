import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores = np.asarray(real_scores, dtype=np.float64)
    fake_scores = np.asarray(fake_scores, dtype=np.float64)

    loss = np.mean(fake_scores) - np.mean(real_scores)

    return float(loss)
