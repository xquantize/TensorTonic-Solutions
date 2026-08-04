import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_per_query = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        num_relevant = np.sum(y_true)
        if num_relevant == 0:
            ap_per_query.append(0.0)
            continue

        sort_indices = np.argsort(-y_score, kind='stable')
        y_true_sorted = y_true[sort_indices]
        
        if k is not None:
            y_true_sorted = y_true_sorted[:k]
            
        cumulative_true = np.cumsum(y_true_sorted)
        ranks = np.arange(1, len(y_true_sorted) + 1)
        precisions = cumulative_true / ranks
        
        ap = np.sum(precisions * y_true_sorted) / num_relevant
        ap_per_query.append(float(ap))
        
    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    
    return map_value, ap_per_query
