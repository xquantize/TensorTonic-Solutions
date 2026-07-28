import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) using floor division for bins.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_samples = len(y_true)
    
    bin_indices = np.floor(y_pred * n_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (bin_indices == i)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_pred[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
            
    return float(ece)
