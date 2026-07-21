import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    accuracy = float(np.sum(y_true == y_pred) / n) if n >0 else 0.0
    classes = np.unique(np.concatenate([y_true, y_pred]))

    tp = {}
    fp = {}
    fn = {}
    support = {}
    
    for c in classes:
        true_is_c = (y_true == c)
        pred_is_c = (y_pred == c)

        tp[c] = int(np.sum(true_is_c & pred_is_c))
        fp[c] = int(np.sum(~true_is_c & pred_is_c))
        fn[c] = int(np.sum(true_is_c & ~pred_is_c))

        support[c] = int(np.sum(true_is_c))

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    if average == "micro":
        tp_total = sum(tp.values())
        fp_total = sum(fp.values())
        fn_total = sum(fn.values())
    
        precision = safe_div(tp_total, tp_total + fp_total)
        recall = safe_div(tp_total, tp_total + fn_total)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "macro":
        precisions, recalls, f1s = [], [], []

        for c in classes:
            p = safe_div(tp[c], tp[c] + fp[c])
            r = safe_div(tp[c], tp[c] + fn[c])
            f = safe_div(2 * p * r, p + r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))

    elif average == "weighted":
        total_support = sum(support.values())
        precisions, recalls, f1s, weights = [], [], [], []

        for c in classes:
            p = safe_div(tp[c], tp[c] + fp[c])
            r = safe_div(tp[c], tp[c] + fn[c])
            f = safe_div(2 * p * r, p + r)
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            weights.append(support[c])

        if total_support > 0:
            weights = np.array(weights, dtype=np.float64) / total_support
            precision = float(np.sum(np.array(precisions) * weights))
            recall = float(np.sum(np.array(recalls) * weights))
            f1 = float(np.sum(np.array(f1s) * weights))
        else:
            precision = recall = f1 = 0.0

    elif average == "binary":
        c = pos_label

        if c in tp:
            precision = safe_div(tp[c], tp[c] + fp[c])
            recall = safe_div(tp[c], tp[c] + fn[c])
            f1 = safe_div(2 * precision * recall, precision + recall)

        else:
            precision = recall = f1 = 0.0
        
    else:
        raise ValueError("unknwon mode")

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
