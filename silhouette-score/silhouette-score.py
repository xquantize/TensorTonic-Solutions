import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    
    if len(unique_labels) <= 1 or len(unique_labels) == n_samples:
        return 0.0

    # dist[i, j] = ||X[i] - X[j]||
    dists = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
    
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)
    
    for i, label in enumerate(unique_labels):
        cluster_mask = (labels == label)
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size > 1:
            a[cluster_mask] = np.sum(dists[cluster_mask][:, cluster_mask], axis=1) / (cluster_size - 1)
        else:
            a[cluster_mask] = 0.0
            
        for other_label in unique_labels:
            if other_label == label:
                continue
                
            other_mask = (labels == other_label)
            mean_dist_to_other = np.mean(dists[cluster_mask][:, other_mask], axis=1)
            b[cluster_mask] = np.minimum(b[cluster_mask], mean_dist_to_other)
            
    with np.errstate(divide='ignore', invalid='ignore'):
        silhouette_values = (b - a) / np.maximum(a, b)
        silhouette_values = np.nan_to_num(silhouette_values)
        
    return float(np.mean(silhouette_values))
