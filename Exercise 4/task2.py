import numpy as np
import matplotlib.pyplot as plt


def equation(alpha, a, b):
    values_alpha = np.where(alpha < b, np.nan, alpha)
    pos_x = np.sqrt((values_alpha - b) / a)
    neg_x = -pos_x
    return (b, 0), pos_x, neg_x


def bifurcation_plot(func, alpha: float, one: bool):
    """Function for plotting the bifurcation phase portrait diagram
    @:param func (function): The equation that we would like to plot
    @:param alpha (float): The value for alpha in the equation to plot
    @:param one (boolean): Boolean variable used in order to know which of the two systems to visualize
    """
    title = ""
    origin_x = ()
    pos_x = None
    neg_x = None
    if one:
        title = r'$\dot{x} = \alpha - x^2$'
        origin_x, pos_x, neg_x = func(alpha, 1, 0)
    else:
        title = r'$\dot{x} = \alpha - 2x^2 - 3$'
        origin_x, pos_x, neg_x = func(alpha, 2, 3)

    plt.plot(alpha, pos_x, color='blue', label='stable equilibrium')
    plt.plot(alpha, neg_x, color='red', label='unstable equilibrium')
    plt.plot(origin_x[0], origin_x[1], 'o', label='bifurcation starting point')
    plt.xlabel('alpha')
    plt.ylabel('x')
    plt.xlim(-10, 10)
    plt.title(title)
    plt.legend()
