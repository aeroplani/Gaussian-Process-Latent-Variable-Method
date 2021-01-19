from math import exp, log, pi, sqrt
from numpy.linalg import cholesky, det, inv, pinv
from scipy.optimize import fmin_cg
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sys import stdout

import numpy as np
import sys

import kernels 


epsilon = sys.float_info.epsilon


class GP():
    """The class of Gaussian Process to optimize kernel parameters.
    :author: Sifan Jiang, Muhammed Memedi
    """

    def __init__(self, X, Y, kernel=None):
        """Initialization of the gaussian process.
        Used for optimizing the kernel parameters.
        :author: Sifan Jiang
        
        :param X: The latent variables.
        :type X: class 'numpy.ndarray'
        :param Y: The data set.
        :type Y: class 'numpy.ndarray'
        :param beta:
        :type beta:
        :param kernel: [description], defaults to None
        :type kernel: class 'kernels.*', optional
        :param parameter_priors: [description], defaults to None
        :type parameter_priors: [type], optional
        """
        self.scaler = StandardScaler()
        self.set_active(X, Y)

        if kernel is None:
            self.kernel = kernels.RBF(np.array([1.0, 1.0, exp(-1), exp(-1)]))
        else:
            self.kernel = kernel

        self.num_params = self.kernel.num_params
        self.hyper = np.ones(self.kernel.num_params)

    def set_active(self, X, Y):
        """Set and standardize the active set by centering and scaling.
        :author: Sifan Jiang

        :param X: The active set of latent variables.
        :type X: class 'numpy.ndarray'
        :param Y: The active set of data.
        :type Y: class 'numpy.ndarray'
        """
        self.X = X.copy()
        self.Y = Y.copy()

        # self.X = self.scaler.fit_transform(self.X)
        # self.Y = self.scaler.fit_transform(self.Y)

        self.N, self.D = self.Y.shape

    def f(self, params):
        self.update(params)
        return self.log_likelihood() + self.log_prior()

    def fprime(self, params):
        self.update(params)
        return self.grad_log_likelihood() + self.grad_log_prior()

    def update(self, params=None):
        """Update the kernel and the variables used for calculating the likelihood and gradients.

        :param parmas: The kernel parameters.
        :type params: class 'numpy.ndarray'
        """
        if params is not None:
            self.kernel.set_params(params)
        self.K = self.kernel(self.X)
        self.L = cholesky(self.K)
        self.Linv = inv(self.L)
        # self.Linv = pinv(self.L)
		# self.Kinv = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(self.L.shape[0])))
        self.Kinv = np.dot(self.Linv.T, self.Linv)
        self.KinvYYT = np.dot(np.dot(self.Kinv, self.Y), self.Y.T)
        self.dL_dK = np.dot(self.KinvYYT, self.Kinv) - self.D * self.Kinv

    def log_prior(self):
        """Return the nagtive log-prior.
        :author: Sifan Jiang

        :return: The nagtive log-prior.
        :rtype: float
        """
        return 1 / 2 * np.dot(self.hyper, np.square(self.kernel.get_params()))

    def grad_log_prior(self):
        """Return the negative gradient of log-prior wrt the parameters.
        :author: Sifan Jiang

        :return: The gradient of log-prior wrt the parameters.
        :rtype: class 'numpy.ndarray'
        """
        return self.hyper * self.kernel.get_params()

    def log_likelihood(self):
        """Return the negative log-likelihood.
        :author: Muhammed Memedi, Sifan Jiang
        
        :param params: The kernel parameters.
        :type params: class 'numpy.ndarray'
        :return: The negative log-likelihood without the constant term.
        :rtype: float
        """
        try:
            L = self.D * self.N / 2 * log(2 * pi) \
                + self.D / 2 * log(det(self.K)+epsilon) \
                + 1 / 2 * np.trace(self.KinvYYT)
        except:
            return np.inf

        return L

    def grad_log_likelihood(self):
        """Calculate the negative gradient of the kernel wrt the parameters.
        :author: Sifan Jiang, Muhammed Memedi
        
        :return: The gradients of the log-likelihood wrt the kernel parameters.
        :rtype: class 'numpy.ndarray'
        """
        try:
            dK_dP = self.kernel.dK_dP(self.X)
            dL_dP = []
            for d in dK_dP:
                dL_dP.append(np.trace(np.dot(self.dL_dK, d)))
        except:
            return np.full(self.kernel.num_params, np.nan)

        return -np.array(dL_dP)

    def opt_kernel_params(self):
        """Optimization of the kernel parameters.
        :author: Sifan Jiang, Muhammed Memedi
        
        :return: Optimized kernel parameters.
        :rtype: class 'numpy.ndarray'
        """
        params = fmin_cg(
            f=self.f,
            x0=self.kernel.get_params(),
            fprime=self.fprime
            )
        self.kernel.set_params(params)
        
        return params


if __name__ == "__main__":
    from numpy.linalg import eig
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris
    import pandas as pd

    def PCA(Y):
        """Optimization of the latent variable X = ULV'
        """
        N, D = Y.shape
        beta = 1.0
        q = 2
        _, v = eig(np.dot(Y, Y.T))
        w, _ = eig(np.dot(Y, Y.T) / D)
        v = np.real(v)
        w = np.real(w)

        U = v[:,0:q]
        L = np.diag((w[0:q] - 1/beta)**(-1/2))
        V = np.eye(q) # Zero rotation angle

        return np.dot(np.dot(U, L), V.T)

    dataset_name = "digits"
    if dataset_name is "breast_cancer":
        dataset = load_breast_cancer() # 2 classes, 569 samples, 30 dimensions
    elif dataset_name is "digits":
        dataset = load_digits() # 10 classes, 1797 samples, 64 dimensions
    elif dataset_name is "iris":
        dataset = load_iris() # 3 classes, 150 samples, 4 dimensions

    Y = dataset.data
    T = dataset.target
    Y = StandardScaler().fit_transform(Y)
    X = PCA(Y)
    targets = list(set(T))
    target_names = list(dataset.target_names)

    gp = GP(X, Y)
    print(gp.opt_kernel_params())
    print(gp.kernel.get_params())