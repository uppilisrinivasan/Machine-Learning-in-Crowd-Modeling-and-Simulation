import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(dataset_path):
    """
    Load data into columns 'x' and 'f(x)'
    :param dataset_path: path of the dataset
    :return: x_array, x, f
    """
    data_df = pd.read_csv(dataset_path, sep=' ', names=['x', 'f(x)'])
    data = data_df.to_numpy()
    x_array = data[:, 0]
    y_array = data[:, 1]
    x = data[:, 0].reshape((1000, 1))
    f = data[:, 1].reshape((1000, 1))
    return x_array, x, f


def linear_approximation(x, function_values):
    """
    Approximate using least-squares minimization formula
    :param x: x values
    :param function_values: f(x) values
    :return: approximate function f(x)_hat
    """
    inverse = np.linalg.inv(x.T @ x)
    approximation = inverse @ x.T @ function_values
    result_approx = x @ approximation
    return result_approx


def plot_linear_approximation(x_array, x, f, f_approx):
    """
    Plot the x-values to actual f-values and approximated f-values
    :param x_array: x values
    :param x: x values
    :param f: f(x) values
    :param f_approx: approximate function f(x)_hat
    """
    plt.scatter(x_array, f, c='blue', label='actual f(x) values')
    plt.plot(x_array, f_approx, c='red', label='approximated f(x) values')
    plt.xlabel('x values')
    plt.ylabel('f(x) and approximated f(x)')
    plt.legend()
    plt.show()


def radial_basis_function_approximation(path, x_array, L, epsilon):
    """
    Approximate using radial basis functions
    :param path: the path of the data we would like to approximate
    :param x_array: x values
    :param L: number of phi functions needed
    :param epsilon: bandwidth
    :return: approximate function f(x)_hat
    """
    data = pd.read_csv(path, sep=' ', names=['x', 'f(x)']).to_numpy()
    f = data[:, 1].reshape((1000, 1))

    # calculating and drawing phi
    phi = draw_phi(x_array, L, epsilon)

    # computation of the coefficient matrix
    inverse = np.linalg.inv(phi.T @ phi)
    coefficient_matrix = inverse @ phi.T @ f

    # visualization of the data
    plt.scatter(data[:, 0], data[:, 1],
                c='blue',
                label='actual f(x) values')
    plt.scatter(data[:, 0], phi @ coefficient_matrix, c='red', label='approximated f(x)_hat values')
    plt.title("Approximated function plotted over the actual data L = {}, epsilon = {}".format(L, epsilon))
    plt.legend()
    plt.show()


def draw_phi(x_array, L, epsilon):
    phi = np.empty([1000, L])
    temp = []

    for point in range(L):
        difference = x_array - x_array[point]
        phi_l = np.exp(-np.square(difference / epsilon))
        temp.append(phi_l)
        phi[:, point] = phi_l

    for a in temp:
        plt.scatter(x_array, a)

    plt.show()
    return phi


def approximate_data(dataset_path, method='linear', L=None, epsilon=None):
    """
    Main function that uses the above functions to approximate the data
    :param dataset_path: path of the dataset
    :param method: method for approximation, 'linear' or 'radial basis function'
    :param L: number of phi functions needed for radial basis function
    :param epsilon: It is the bandwidth for radial basis function
    """
    x_array, x, f = load_data(dataset_path)

    if method == 'linear':
        function_approximation = linear_approximation(x, f)
        plot_linear_approximation(x_array, x, f, function_approximation)

    elif method == 'radial basis function':
        if L is None or epsilon is None:
            raise ValueError("Both parameters ('L' and 'epsilon') must be set for the radial basis function approach")
        radial_basis_function_approximation(dataset_path, x_array, L, epsilon)
    else:
        raise ValueError("Invalid method specified, choose either 'linear' or 'radial basis function'")
