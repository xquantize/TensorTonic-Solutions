import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    n_train, d = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)

    log_priors = np.zeros(n_classes, dtype=np.float64)
    log_theta = np.zeros((n_classes, d), dtype=np.float64)
    log_one_minus_theta = np.zeros((n_classes, d), dtype=np.float64)

    alpha = 1.0

    for c_idx, c in enumerate(classes):
        mask = (y_train == c)
        X_c = X_train[mask]
        n_c = X_c.shape[0]

        log_priors[c_idx] = np.log(n_c / n_train)
        count_1 = X_c.sum(axis=0)
        theta = (count_1 + alpha) / (n_c + 2 * alpha)

        log_theta[c_idx] = np.log(theta)
        log_one_minus_theta[c_idx] = np.log(1.0 - theta)

    term1 = X_test @ log_theta.T
    term2 = (1.0 - X_test) @ log_one_minus_theta.T

    log_posterior = term1 + term2 + log_priors[np.newaxis, :]

    return log_posterior
