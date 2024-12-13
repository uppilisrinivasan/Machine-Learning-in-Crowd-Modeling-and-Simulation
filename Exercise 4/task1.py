import numpy as np
import matplotlib.pyplot as plt


def plot_phase_portrait(A: np.ndarray, alpha: float):
    """
    Plots the phase portrait of the linear system Ax, where A is a 2x2 matrix and x is a 2-dim vector
    :param A: system's 2x2 matrix / parameterized matrix
    :param alpha: system's parameter
    """
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)

    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    print("Shape of UV Matrix is: {}".format(UV.shape))
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=.9)
    plt.title("alpha: {}, lambda_1: {}, lambda_2: {} ".format(alpha, np.round(eigenvalues[0], 3),
                                                              np.round(eigenvalues[1], 3)))
    plt.show()


