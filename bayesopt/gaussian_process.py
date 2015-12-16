import numpy as np


class Kernel(object):

    def compute(self, a, b):
        raise None


class SquaredDistanceKernel(Kernel):

    def __init__(self, kernel_param=0.1):
        self.kernel_parameter = kernel_param

    def compute(self, a, b):
        sq_dist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-.5 * (1/self.kernel_parameter) * sq_dist)


class GaussianProcess(object):
    """
    Implements a GP with mean zero and a custom kernel
    """
    def __init__(self, kernel=SquaredDistanceKernel(), noise_variance=0.00005, x=None, y=None):
        """
        Initialize the GP with the given kernel and a noise parameter for the variance
        Optionally initialize this GP with given X and Y

        :param kernel: kernel function, has to be an instance of Kernel
        :param noise_variance:
        :param x: given input data
        :param y: given input label
        :return:
        """
        self.X = x
        self.Y = y
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.cov = None if self.X is None else kernel.compute(self.X, self.X)

    def predict(self, x, y=None):
        """
        Given data in x, give the mean and covariance of the posterior predictive distribution p(f*|X*, X, f)
        If y is given, the function gives the predicts, as well as update the GP internally

        x should have size N x d1, y of size N x d2, where N is the number of samples

        :param x: the input data
        :param y: optional. If given, the GP parameters will be updated
        :return: a tuple (mu, cov, s):
            - mu: the mean of the posterior predictive distribution, of size N x d1
            - cov: the covariance matrix of the posterior predictive distribution, of size N x N
            - s: the standard deviation vector, convenient for plotting. Of size N x 1
        """
        # covariance of the new data
        k_2star = self.kernel.compute(x, x)

        if self.cov is None:
            # if there is no data in this GP, this is equivalent to the prior distribution (zero mean, unit covariance)
            mu = np.zeros(x.shape)
            cov_posterior = k_2star + (self.noise_variance * np.eye(k_2star.shape[0]))

            if y is not None:
                self.X = x
                self.Y = y
                self.cov = k_2star
        else:
            l = np.linalg.cholesky(self.cov + self.noise_variance * np.eye(self.cov.shape[0]))
            k_star = self.kernel.compute(self.X, x)
            l_div_k_star = np.linalg.solve(l, k_star)

            mu = np.dot(l_div_k_star.T, np.linalg.solve(l, self.Y))
            cov_posterior = k_2star + self.noise_variance * np.eye(k_2star.shape[0]) - np.dot(l_div_k_star.T,
                                                                                              l_div_k_star)
            if y is not None:
                    self.X = np.vstack((self.X, x))
                    self.Y = np.vstack((self.Y, y))
                    self.cov = np.hstack((self.cov, k_star))
                    self.cov = np.vstack((self.cov, np.hstack((k_star.T, k_2star))))

        return mu, cov_posterior, np.sqrt(np.diag(cov_posterior))
