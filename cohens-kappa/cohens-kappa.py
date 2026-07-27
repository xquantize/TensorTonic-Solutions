import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    
    if rater1.shape != rater2.shape:
        raise ValueError("raters must have same shape")
        
    classes = np.unique(np.concatenate((rater1, rater2)))
    n_classes = len(classes)
    
    # rows: rater1, cols: rater2
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            confusion_matrix[i, j] = np.sum((rater1 == c1) & (rater2 == c2))
            
    n = np.sum(confusion_matrix)
    
    if n == 0:
        return 0.0
        
    sum_diag = np.sum(np.diag(confusion_matrix))
    p_o = sum_diag / n
    
    sum_row = np.sum(confusion_matrix, axis=1)
    sum_col = np.sum(confusion_matrix, axis=0)
    
    p_e = np.sum(sum_row * sum_col) / (n ** 2)
    
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    cohen_out = float((p_o - p_e) / (1 - p_e))
        
    return cohen_out
