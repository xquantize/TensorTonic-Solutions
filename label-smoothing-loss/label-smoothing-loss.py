def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)
    loss = 0.0

    for i, p_i in enumerate(predictions):
        if i == target:
            q_i = (1 - epsilon) + epsilon / K
        else:
            q_i = epsilon / K

        loss += q_i * math.log(p_i)

    return -loss
