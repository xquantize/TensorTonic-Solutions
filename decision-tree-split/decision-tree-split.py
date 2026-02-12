def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # codes
    def get_gini(labels):
        if not labels:
            return None

        n = len(labels)
        counts = {}

        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        impurity = 1 - sum((count / n) ** 2 for count in counts.values())
        
        return impurity

    n_samples = len(X)
    n_features = len(X[0])
    parent_gini = get_gini(y)

    best_gain = -1
    best_feature = None
    best_threshold = None

    for f_idx in range(n_features):
        # unique values for feature for examples
        values = sorted(list(set(row[f_idx] for row in X)))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2

            # split y based on thresh
            left_y = [y[j] for j in range(n_samples) if X[j][f_idx] <= threshold]
            right_y = [y[j] for j in range(n_samples) if X[j][f_idx] > threshold]

            # calc weighted goal
            n_l, n_r = len(left_y), len(right_y)
            weighted_gini = (n_l / n_samples) * get_gini(left_y) + (n_r / n_samples) * get_gini(right_y)

            gain = parent_gini - weighted_gini

            # update best split
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_feature = f_idx
                best_threshold = threshold

    return [best_feature, best_threshold]
