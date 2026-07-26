import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    losses = []
    
    for yt, yp in zip(y_true, y_pred):
        yp_clipped = max(min(yp, 1 - eps), eps)
        
        sample_loss = - (yt * math.log(yp_clipped) + (1 - yt) * math.log(1 - yp_clipped))
        
        losses.append(sample_loss)
        
    return losses
