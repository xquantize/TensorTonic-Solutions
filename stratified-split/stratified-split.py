import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # codes
    X = np.array(X)
    y = np.array(y)

    if rng is None:
            rng = np.random.default_rng(42)
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []

    for cls in classes:
        # get indices for this class
        cls_indices = np.where(y == cls)[0]
        
        # shuff indices for this specific class
        rng.shuffle(cls_indices)
        
        # calc n_test for this class
        n_test = int(np.round(len(cls_indices) * test_size))

        # handle edge cases (ensure at least 1 in train if possible)
        if n_test == len(cls_indices) and len(cls_indices) > 1:
            n_test -= 1
        if n_test == 0 and len(cls_indices) > 1 and test_size > 0:
            n_test = 1

        # distri indices to global lists
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)

    # shuf the final lists so the classes are mixed up
    train_indices.sort()
    test_indices.sort()

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
