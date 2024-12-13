import numpy as np
from sympy import solveset, S
from sympy.abc import x
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def AndronovHopf(t, vec):
    """Defines Andronov-Hopf dynamical system and returns it's update step
    @:param t (int): This is a Variable that is going to be used for the scipy's ivp solver
    @:param vec (np.array): This is the input vector
    @:return dx1,dx2: Andronov-Hopf update step
    """
    alpha = 1
    dx1 = alpha * vec[0] - vec[1] - vec[0] * (vec[0] ** 2 + vec[1] ** 2)
    dx2 = vec[0] + alpha * vec[1] - vec[1] * (vec[0] ** 2 + vec[1] ** 2)
    return dx1, dx2


def phase_portrait_andronov_hopf(alpha):
    """Function that is used in order to plot andronov hopf phase portrait
    @:param alpha (float): This represents the value for alpha in the andronov hopf equation
    """

    x1 = np.linspace(-3.0, 3.0, 15)
    x2 = np.linspace(-3.0, 3.0, 15)

    X1, X2 = np.meshgrid(x1, x2)

    # Andronov_Hopf normal form
    u = alpha * X1 - X2 - X1 * (X1 ** 2 + X2 ** 2)
    v = X1 + alpha * X2 - X2 * (X1 ** 2 + X2 ** 2)

    plt.figure(figsize=(10, 10))
    plt.streamplot(X1, X2, u, v)
    plt.plot(0, 0, marker="o", markersize=20)

    plt.axis('square')
    plt.axis([-3, 3, -3, 3])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('alpha={}'.format(round(alpha, 2)))
    plt.show()


def plot_orbit(starting_point):
    """This function is used for creating the orbits with a given starting point for Task 3.2
    @:param starting_point(float): the starting point where the plot begins
    """
    t_start = 0
    t_end = 15
    number_of_steps = 10000
    t_space = np.linspace(t_start, t_end, number_of_steps)
    y0 = np.array(starting_point)
    solution = solve_ivp(fun=AndronovHopf, t_span=[t_space[0], t_space[-1]], y0=y0, t_eval=t_space)

    plt.figure(figsize=(15, 5))
    plt.scatter(solution.y[0], solution.y[1])
    plt.title('View of Orbit using a Scatter Plot along time. Starting point: {}'.format(starting_point))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def cusp(x, alpha):
    """Cusp dynamical system
    @:param x: The state value
    @:param alpha: the alphas for the cusp bifurcation
    @:return the cusp dynamical system's update state
    """
    return alpha[0] + alpha[1] * x - x ** 3


def plot_surface_cusp():
    """Function used for the visualization of the Cusp Bifurcation in 3D Space
    """

    plt.figure(figsize=(12, 8))

    space_1 = np.linspace(-2, 2, 40)
    space_2 = np.linspace(-2, 2, 40)
    points = []

    ax = plt.axes(projection='3d')
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('x')
    ax.set_title('Visualization of cusp bifurcation surface in 3D')

    for a1 in space_1:
        for a2 in space_2:
            equation_result = solveset(a1 + a2 * x - x ** 3, x, domain=S.Reals)
            for z in equation_result:
                points.append((a1, a2, z))

    for point in points:
        ax.scatter3D(point[0], point[1], point[2], color='green')

    ax.view_init(0, 90)
    plt.show()
