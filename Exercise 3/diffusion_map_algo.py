import numpy as np
from numpy import linalg as LA
import math
from scipy.sparse.linalg import eigsh

def diffusion_map_algo(X, L):
    """
    X: Input data matrix
    L: Number of largest eigenvalues to be computed (L+1 largest eigenvalues are computed)

    """
    # Step 1: Form a distance matrix D

    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance = (X[i] - X[j])
            D[i][j] = math.sqrt(np.sum(distance ** 2))

    # Step 2: Set hyperparameter determining range of datapoints in effective neighborhood
    epsilon = 0.05 * np.max(D)

    # Step 3: Form the kernel matrix W
    W = np.exp((-D**2)/epsilon)

    # Step 4: Form the diagonal normalization matrix
    P = np.zeros((N, N))
    for i in range(N):
        P[i, i] = np.sum(W[i])

    # Step 5: Normalize W to form the kernel matrix
    inv_P = LA.inv(P)
    K = inv_P.dot(W.dot(inv_P))

    # Step 6: Form the diagonal normalization matrix
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = np.sum(K[i])

    # Step 7: Form the symmetric matrix
    inv_Q_sqrt = np.sqrt(LA.inv(Q))
    T = inv_Q_sqrt.dot(K.dot(inv_Q_sqrt))

    # Step 8: Find the L + 1 largest eigenvalues a and associated eigenvectors v of T
    a, v = eigsh(T, k = L + 1)

    # Step 9: Compute the eigenvalues _lambda
    _lambda = np.sqrt(a ** (1 / epsilon))

    # Step 10: Compute the eigenvectors _phi
    _phi = inv_Q_sqrt.dot(v)

    return _lambda, _phi


