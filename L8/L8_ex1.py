import numpy as np
import matplotlib.pyplot as plt

class gp:
    #class for the gaussian processes
    def __init__(self, noise_variance=0):
        #noise is initialized
        self.noise_variance = noise_variance

    def _radial_basis_kernel(self, x1, x2, l):
        #function for the basis kernal
        return np.exp(-(np.transpose(x1 - x2) * (x1 - x2)) / (2 * l**2))

    def _radial_basis_matrix(self, X1, X2, l):
        #function to create the basis matrix
        matrix = np.zeros([np.shape(X1)[0], np.shape(X2)[0]])
        for i in range(np.shape(matrix)[0]):
            for j in range(np.shape(matrix)[1]):
                matrix[i, j] = self._radial_basis_kernel(X1[i], X2[j], l)
        return matrix

    def compute_mean(self, x, y, X, l):
        #function to calc the mean
        return np.dot(np.dot(self._radial_basis_matrix(np.transpose([x]), X, l),
                             np.linalg.inv(self._radial_basis_matrix(X, X, l) +
                                           self.noise_variance**2 * np.eye(np.shape(X)[0]))) , y)

    def compute_variance(self, x, X, l):
        #function to calc the variance
        radial_kernal_1 = self._radial_basis_kernel(x, x, l)
        radial_matrix_1 = self._radial_basis_matrix(np.transpose([x]), X, l)
        radial_matrix_2 = self._radial_basis_matrix(X, X, l)
        noise = self.noise_variance**2 * np.eye(np.shape(X)[0])
        radial_matrix_3 = self._radial_basis_matrix(X, np.transpose([x]), l)
        return radial_kernal_1 - np.dot(np.dot(radial_matrix_1, np.linalg.inv(radial_matrix_2 + noise)), radial_matrix_3)

def visualize(x, y, X, Y, mean, variance):
    #function to plot the results
    plt.plot(x,y,'g', label='target function')
    plt.plot(X,Y,'rx', markersize=15, label='training points')
    plt.errorbar(x,mean[0], yerr=2*np.sqrt(np.abs(variance[0])), label='approximation and standard deviation', marker='.', mec='blue')
    plt.legend(loc='upper right', shadow=False, fontsize='x-small')
    plt.show()

def main_f():
    #main function for the gaussian processes

    #init values
    l = 1

    noise = 0

    # known Points
    X = np.arange(-5, 6, 2)
    Y = np.vstack(np.sin(X))

    # approximation interval
    x = np.arange(-5,5.1,0.1)
    y = np.vstack(np.sin(x))

    #inst the gaussian processes class with noise
    gaussian_process = gp(noise)

    #init the matricees for mean and variance
    mean = np.zeros([1, np.shape(x)[0]])
    variance = np.zeros([1, np.shape(x)[0]])

    #iterate trhough the values
    for idx,t in enumerate(x):
        mean[0, idx] = gaussian_process.compute_mean(t, Y, X, l)
        variance[0, idx] = gaussian_process.compute_variance(t, X, l)

    #visualize the results
    visualize(x, y, X, Y, mean, variance)


if __name__ == '__main__':
    main_f()