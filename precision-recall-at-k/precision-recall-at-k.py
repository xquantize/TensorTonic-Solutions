def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = set(recommended[:k])
    relevant_set = set(relevant)

    hits = len(top_k & relevant_set)

    precision = hits / k
    recall = hits / len(relevant_set)

    return [precision, recall]
