def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    t = step
    W = warmup_steps
    T = total_steps

    if t < W:
        lr = t * initial_lr / W
    elif t <= T:
        if T == W:
            lr = final_lr
        else:
            lr = final_lr + (initial_lr - final_lr) * (T - t) / (T - W)

    else:
        lr = final_lr

    return float(lr)
