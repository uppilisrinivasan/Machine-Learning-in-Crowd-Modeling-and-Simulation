import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def load_data(dataset_1_path, dataset_2_path):
    """
    Loading of the data and computation of the 'V' Vector

    :param dataset_1_path: path to the dataset for x0
    :param dataset_2_path: path to the dataset for x1
    :return X0: data of X0
    :return X1: data for X1
    :return vector_v: the V Vector
    """
    data_x0 = pd.read_csv(dataset_1_path, sep=' ', names=['x0_x', 'x0_y'])
    data_x1 = pd.read_csv(dataset_2_path, sep=' ', names=['x1_x', 'x1_y'])
    X0 = data_x0.to_numpy()
    X1 = data_x1.to_numpy()
    vector_v = (X1 - X0) / 0.1
    return X0, X1, vector_v


def compute_A_hat(X0, V):
    """
    Function for computing the A_hat value, that will be needed for further calculations

    :param X0: data of X0
    :param V: the V Vector
    :return a_matrix_hat: the A_hat matrix
    """
    inverse = np.linalg.inv(X0.T @ X0)
    result = inverse @ X0.T @ V
    a_matrix_hat = result.T
    return a_matrix_hat


def derivative(t, x, A):
    """
    Function that represents the derivative function

    """
    return x.dot(A)


def compute_X1_approx(X0, A_hat):
    """

    :param X0: data of X0
    :param A_hat: the A_hat matrix
    :return: the approximated values for the data in X1
    """
    approximation = []
    for row in range(X0.shape[0]):
        sol = solve_ivp(fun=derivative, t_span=[0, 0.2], t_eval=[0.1], y0=X0[row, :], args=(A_hat,))
        res = sol.y
        approximation.append(res)

    result = np.array(approximation)
    result = result.reshape((1000, 2))
    return result


def calculate_MSE(X1, X1_approx):
    """

    :param X1: the real values for X1
    :param X1_approx: the approximated values for X1
    :return: the Mean-Square-Error(MSE) between the real data and the approximated data
    """
    result = np.square(X1 - X1_approx)
    MSE = result.mean()
    return MSE


def plot_trajectory(A_hat):
    """

    :param A_hat: the A_hat matrix

    """
    evaluation = []

    for i in range(1000):
        value = i / 10
        evaluation.append(value)

    sol = solve_ivp(fun=derivative, t_span=[0, 1000], y0=[10, 10], t_eval=evaluation, args=(A_hat,))
    plt.scatter(sol.y[0, :], sol.y[1, :])
    plt.xlabel("x_coordinate")
    plt.ylabel("y_coordinate")
    plt.show()


def plot_trajectory_and_phase_portrait(A_hat):
    """
    Plots the trajectory
    :param A_hat: the A_hat matrix

    """
    evaluation = []

    for i in range(1000):
        value = i / 10
        evaluation.append(value)

    sol = solve_ivp(fun=derivative, t_span=[0, 1000], y0=[10, 10], t_eval=evaluation, args=(A_hat,))

    x = sol.y[0, :-1]
    y = sol.y[1, :-1]
    u = sol.y[0, 1:] - sol.y[0, :-1]
    v = sol.y[1, 1:] - sol.y[1, :-1]
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    plt.xlabel("x_coordinate")
    plt.ylabel("y_coordinate")
    plt.show()


def plot_phase_portrait(starting_point, A_hat):
    """
    Plots the phase portrait
    :param starting_point: the starting point
    :param A_hat: the A_hat matrix

    """
    evaluation = []

    for i in range(100):
        value = i / 10
        evaluation.append(value)

    x1 = np.linspace(-10, 10, 50)
    x2 = np.linspace(-10, 10, 50)

    X1, X2 = np.meshgrid(x1, x2)

    u = A_hat[0][0] * X1 + A_hat[0][1] * X2
    v = A_hat[1][0] * X1 + A_hat[1][1] * X2

    span = 100 / 0.1
    line = np.zeros(shape=(int(span), 2))
    for i in range(int(span)):
        line[i] = starting_point
        starting_point = 0.1 * (starting_point @ A_hat) + starting_point
    line = line.T

    fig5 = plt.figure(figsize=(8, 8))
    fig5.add_subplot()
    plt.streamplot(X1, X2, u, v)
    plt.plot(line[0], line[1], color='red')
    plt.title('Trajectory and phase portrait')


def approximate_linear_vector_fields(dataset_1_path, dataset_2_path):
    """
    This is something like a main function
    :param dataset_1_path: path to the dataset for x0
    :param dataset_2_path: path to the dataset for x1
    """
    X0, X1, V = load_data(dataset_1_path, dataset_2_path)
    A_hat = compute_A_hat(X0, V)
    X1_approx = compute_X1_approx(X0, A_hat)

    X1_df = pd.read_csv(dataset_2_path, sep=' ', names=['x1_x', 'x1_y'])
    X1_np = X1_df.to_numpy()
    MSE = calculate_MSE(X1_np, X1_approx)

    print("A_hat: {}".format(A_hat))
    print("MSE: {}".format(MSE))

    plot_trajectory(A_hat)
    plot_trajectory_and_phase_portrait(A_hat)
    starting_point = [10, 10]
    plot_phase_portrait(starting_point, A_hat)
