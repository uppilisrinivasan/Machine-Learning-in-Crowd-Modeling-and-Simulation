import numpy as np
from scipy.spatial.distance import cdist

def rbf(x, x_l, eps):
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)

def approx_nonlin_func(data, eps, centers):
    bases = rbf(data[0], centers, eps)
    sol, residuals, rank, singvals = np.linalg.lstsq(a=bases, b=data[1], rcond=1e-5)
    return sol, residuals, rank, singvals, bases
    
def rbf_approx(t, y, centers, eps, C):
    y = y.reshape(1, y.shape[-1])
    phi = np.exp(-cdist(y, centers) ** 2 / eps ** 2)
    return phi @ C