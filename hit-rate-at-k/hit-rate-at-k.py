def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    num_users = len(recommendations)

    if num_users == 0:
        return 0.0

    hits = 0

    for rec_list, relevant in zip(recommendations, ground_truth):
        top_k = set(rec_list[:k])
        relevant_set = set(relevant)

        if top_k & relevant_set:
            hits += 1

    return hits / num_users
