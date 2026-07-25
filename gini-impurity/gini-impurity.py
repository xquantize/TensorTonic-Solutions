import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    n_left = y_left.size
    n_right = y_right.size
    
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
        
    def calculate_node_gini(y, n):
        if n == 0:
            return 0.0
            
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        
        return 1.0 - np.sum(probabilities ** 2)
        
    gini_left = calculate_node_gini(y_left, n_left)
    gini_right = calculate_node_gini(y_right, n_right)
    
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    return float(weighted_gini)
