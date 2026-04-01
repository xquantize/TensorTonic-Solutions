import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # codes
    # (T, N)
    array = np.array(predictions)

    T, N = array.shape
    result = []

    for i in range(N):
        v = array[:, i]
        cls, cnt = np.unique(v, return_counts=True)
        max_cnt = cnt.max()
        tied = cls[cnt == max_cnt]
        result.append(int(tied.min()))
    
    return result
    
    