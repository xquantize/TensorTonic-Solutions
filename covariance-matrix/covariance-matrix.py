import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)
    
    if X.ndim != 2 or X.shape[0] <= 1:
        return None
        
    n_samples = X.shape[0]
    
    X_centered = X - np.mean(X, axis=0)
    
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    return cov_matrix
