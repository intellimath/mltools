import numpy as np

def modified_zscore(X):
    median = np.median

    mu = median(X)
    X_mu = X - mu
    sigma = median(abs(X_mu))
    return 0.6745 * X_mu / sigma

def zscore(X):
    from math import sqrt

    mu = X.mean()
    X_mu = X - mu
    sigma2 = (X_mu * X_mu).mean()
    sigma = sqrt(sigma2)
    return X_mu / sigma