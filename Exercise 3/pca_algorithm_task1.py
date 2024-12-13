import matplotlib
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import copy

from PIL import Image, ImageOps
from scipy.linalg import svd
from numpy import linalg as LA
from itertools import cycle

""" 1. Form the data matrix X ∈ RN ×n with rows xi from points (observations) in the data set """

""" Method that reads the pca dataset and creates a Pandas Dataframe out of it"""


def read_data():
    data = pd.read_csv("pca_dataset.txt", sep=" ", header=None)
    return data


""" Method that reads the pedestrians dataset and creates a Pandas Dataframe out of it"""


def read_pedestrians_data():
    data = pd.read_csv("data_DMAP_PCA_vadere.txt", sep=" ", header=None)
    return data


""" 2. Center the matrix by removing the data mean x = 1 from every row (every data point):."""


def calculate_mean(data):
    mean_of_data = np.mean(data, axis=0)
    data_normalized = data - mean_of_data

    return data_normalized


""" 3. Decompose the centered data matrix into singular vectors U, V and values S, such that """



def pca(data):
    """
    Principal Component Analysis with Singular Value Decomposition(SVD) durchfuehren
        :param data : the data where we would like to apply pca
        :return U : Unitary matrix having left singular vectors as columns.
        :return singular_values: The singular values, sorted in non-increasing order.
        :return Vh.T: Unitary matrix having right singular vectors as rows.
        :return S : Diagonal-Matrix composed from the singular values
    """

    normalized_data = calculate_mean(data)
    U, singular_values, Vh = svd(normalized_data)

    """We have to create the S matrix out of the singular values"""
    # Diagonal Matrix of the singular values
    diagonal_matrix = np.diag(singular_values)
    # Zero matrix to fill the missing part for the S matrix
    zero_matrix = np.zeros(shape=(normalized_data.shape[0] - len(singular_values), len(singular_values)))
    S = np.vstack((diagonal_matrix, zero_matrix))

    return U, singular_values, Vh.T, S



def compute_energy(data, number_components):
    """Computes the energy (explained variance) on each principal component
        :param data : the data where we would like to apply pca
        :param number_components (int): Number of principal components we have
        :return ndarray: the energy of the principal component
    """

    # performing pca on data matrix
    U, singular_values, Vh, S = pca(data)
    # calculating trace of squared matrix s
    trace = np.sum(np.square(singular_values))
    # calculating energy matrix
    new_S = shrink_principle_components_matrix(data, number_components)
    energy = np.square(new_S) / trace
    return np.sum(energy.diagonal())



def visualize_pc(data, Vh, singular_values):
    """ Method used for visualization
        :param data:  data we will visualize
        :param Vh: Unitary matrix having right singular vectors as rows.
        :param singular_values : the singular values
     """

    # calculate the center where the arrows of the principal components should start
    center = np.mean(data, axis=0)

    # import color cycler from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    # create a figure where we will visualize the data and the principal components
    fig = plt.figure()
    ax = fig.add_subplot()

    # rename the columns of the data for better style
    data.columns = ['x-coordinate', 'y-coordinate']
    data.plot(ax=ax, kind='scatter', x='x-coordinate', y='y-coordinate', color='cyan', title='Scatter plot of the data',
              xlabel='x', ylabel='f(x)')

    # plotting principal components
    vectors_numbered = enumerate(Vh)
    for pc_number, components in vectors_numbered:
        ax.quiver(center[0], center[1], components[0], components[1],
                  label="Principal Component {} ".format(pc_number + 1),
                  color=next(colors), angles="xy", scale_units="xy", scale=1)

    plt.title("Principal Component-1 singular_value=: {}"
              ", Principal Component-2 singular_value=: {}".format(np.around(singular_values[0], 2),
                                                      np.around(singular_values[1], 2)))
    plt.legend(loc="best")
    plt.show()


def racoon_processing():
    """ Method for processing the raccoon image, which includes loading, resizing and converting it into grayscale
        :return new_raccoon: Newly created raccoon image
    """
    racoon_img = Image.open('raccoon-procyon-lotor-850x638.jpg')
    gray_racoon = ImageOps.grayscale(racoon_img)
    new_racoon = gray_racoon.resize((249, 185))
    new_racoon.save('image_new_format.jpg')

    plt.imshow(new_racoon, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return new_racoon





def shrink_principle_components_matrix(data, number_components):
    """ Method for shrinking the S matrix: The diagonal matrix composed from the singular values
        :return new_S: new diagonal matrix
    """

    U, singular_values, Vh, S = pca(data)
    new_S = copy.deepcopy(S)
    new_S[number_components:, number_components:] = 0
    return new_S


def reconstruction(data, number_components):
    """ Reconstructs the raccoon image and its visualization
        :param number_components: the number of components we would like to have for conducting pca
    """
    U, singular_values, Vh, S = pca(data)
    new_S = shrink_principle_components_matrix(data, number_components)
    mean = np.mean(data)
    reconstruction_result = (U @ new_S @ Vh.T).T + mean

    plt.imshow(reconstruction_result, cmap='gray', vmin=0, vmax=255)
    plt.title("Reconstructions of the image with {} principal components".format(number_components))
    plt.show()


def find_pc(data):
    """ Function that finds the number of principal components that retain the energy loss below one percent"""
    U, singular_values, Vh, S = pca(data)
    temp = 0
    for i in range(len(singular_values), 0, -1):
        current_energy = compute_energy(data, i)
        current_energy_loss = (1 - current_energy) * 100
        if current_energy_loss > 1.0:
            print("less than 1% energy loss at number of principal components: {}".format(i + 1))
            temp = i + 1
            break
    return temp


if __name__ == '__main__':
    data = read_data()
    normalized_data = calculate_mean(data)
    # U, singular_values, Vh, S = pca(data)
    # energy = compute_energy(data)
    # visualize_pc(normalized_data, Vh.T, singular_values)

    """
    racoon_image = racoon_processing()
    racoon_image_np = np.asarray(racoon_image)
    input_racoon = racoon_image_np.T
    U, singular_values, Vh, S = pca(input_racoon)
    energy = compute_energy(input_racoon)
    """

    """
    reconstruction(input_racoon, 249)
    reconstruction(input_racoon, 120)
    reconstruction(input_racoon, 50)
    reconstruction(input_racoon, 10)
    """

    """
    temp = find_pc(input_racoon)
    reconstruction(input_racoon, temp)
    """

    pedestrians = read_pedestrians_data()
    U, singular_values, Vh, S = pca(pedestrians)
    print(type(U))
    print(type(singular_values))
    print(type(Vh))
    print(type(S))
    new_S = shrink_principle_components_matrix(pedestrians, 2)
    print(type(new_S))
    mean = pedestrians.apply(numpy.mean)
    mean_np = mean.to_numpy()
    print(type(mean_np))
    U @ new_S
    U @ new_S @ Vh.T
    reconstruction_result = ((U @ new_S @ Vh.T) + mean_np)

    boolean = True
