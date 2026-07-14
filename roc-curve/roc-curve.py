import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_indx = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indx]
    fps = np.cumsum(1 - y_true)[threshold_indx]

    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos

    fpr = fps / total_neg
    tpr = tps / total_pos
    
    thresholds = y_score[threshold_indx]

    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]

    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds
    