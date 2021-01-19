from math import exp, log, pi
from numpy.linalg import det, inv, pinv
from scipy.optimize import fmin_cg

import numpy as np
import sys
epsilon = sys.float_info.epsilon


class Linear():
    """The class of Linear kernel.
    $$
        k(x_i,x_j) = \theta * x_i * x_j + \beta
    $$
    :author: Sifan Jiang
    """
    
    def __init__(self, params):
        """Initialization of the kernel.
        :author: Sifan Jiang
        
        :param params: Parameters of the kernel.
        :type params: class 'numpy.ndarray'
        """       
        self.theta, self.bias = np.log(1.0 + np.exp(params.copy()))
        self.num_params = params.size

    def __call__(self, X, Y=None):
        """Call funtion of the kernel, which calculates the kernel matrix of two matrix.
        If Y is None, the kernel matrix K(X, X) is calculated.
        :author: Sifan Jiang
        
        :param X: Latent variable matrix.
        :type X: class 'numpy.ndarray'
        :param Y: Latent variable matrix, should have same dimension D with X, defaults to None
        :type Y: class 'numpy.ndarray', optional
        :return: The kernel matrix.
        :rtype: class 'numpy.ndarray'
        """   
        Nx, Dx = X.shape
        
        if Y is None:
            Y = X
        Ny, Dy = Y.shape
        
        K = X.reshape(Nx,1,Dx) * Y.reshape(1,Ny,Dy)
        K = self.theta * np.sum(K,-1) + self.bias
        return K
   
    def dK_dP(self, X):
        """Calculate the gradient of the kernel wrt the parameters.
        :author: Sifan Jiang

        :param X: The latent variable matrix [N x q].
        :type X: class 'numpy.ndarray'
        :return: The gradient of the kernel wrt the parameters.
        :rtype: list
        """   
        N, D = X.shape

        d_theta = X.reshape(N,1,D) * X.reshape(1,N,D)
        d_theta = np.sum(d_theta,-1)
        d_bias = np.ones((N, N))
        return d_theta, d_bias

    def dK_dX(self, X, n=None, j=None):
        """Calculate the gradient of the kernel with respect to the latent variable X
        :author: Sifan Jiang

        :param X: The latent variables.
        :type X: class 'numpy.ndarray'
        :param n: The index of the latent variable to be calculated for the gradient, defaults to None
        :type n: int, optional
        :param j: The dimension of the latent vaiable to be calculated for the graident, defaults to None
        :type j: int, optional
        :return: The gradient of the kernel wrt a certain latent variable.
                 Or a list of the gradient of the kernel wrt all the latent vairables.
        :rtype: class 'numpy.ndarray' / list
        """      
        N, D = X.shape

        if n is None and j is None:
            gradList = []
            for n in range(N):
                for j in range(D):
                    grad = np.zeros((N,N))
                    grad[n,:] = self.theta * np.ones(N) + self.bias
                    grad[:,n] = grad[n,:]
                    gradList.append(grad.copy())
            return gradList

        grad = np.zeros((N,N))
        grad[n,:] = self.theta * np.ones(N) + self.bias
        grad[:,n] = grad[n,:]
        return grad

    def set_params(self, params):
        """Set the parameters of the kernel.
        :author: Sifan Jiang
        
        :param params: The parameters of the kernel to be set.
        :type params: class 'numpy.ndarray'
        """     
        assert params.size == self.num_params, "[Error]: Number of parameters is wrong when setting the parameters of kernel."
        self.theta, self.bias = np.log(1.0 + np.exp(params.copy()))

    def get_params(self):
        """Get the parameters of the kernel.
        :author: Sifan Jiang
        
        :return: The parameters of the kernel.
        :rtype: class 'numpy.ndarray'
        """       
        return np.array([self.theta, self.bias])


class RBF():
    """The class of RBF kernel with bias and white.
    $$
        k(x_i,x_j) = \theta_{\text{rbf}} * \exp\left[- \frac{\gamma}{2} \|x_i-x_j\|^2 \right] + \theta_{\text{bias}} + \theta_{\text{white}}\delta_{ij}
    $$
    :author: Sifan Jiang
    """
    
    def __init__(self, params):
        """Initialization of the kernel.
        :author: Sifan Jiang
        
        :param params: Parameters of the kernel.
        :type params: class 'numpy.ndarray'
        """
        self.theta, self.gamma, self.bias, self.white = np.log(1.0 + np.exp(params.copy()))
        self.num_params = params.size

    def __call__(self, X, Y=None, white=True):
        """Call funtion of the kernel, which calculates the kernel matrix of two matrix.
        If Y is None, the kernel matrix K(X, X) is calculated.
        :author: Sifan Jiang
        
        :param X: Latent variable matrix.
        :type X: class 'numpy.ndarray'
        :param Y: Latent variable matrix, should have same dimension D with X, defaults to None
        :type Y: class 'numpy.ndarray', optional
        :return: The kernel matrix.
        :rtype: class 'numpy.ndarray'
        """
        Nx, Dx = X.shape
        
        if Y is None:
            Y = X
        Ny, Dy = Y.shape

        dist = X.reshape(Nx,1,Dx) - Y.reshape(1,Ny,Dy)
        dist = np.sum(np.square(dist),-1)
        K = self.theta * np.exp(- self.gamma / 2 * dist) + self.bias
        if white is True:
            K += self.white * np.eye(Nx, Ny)

        return K

    def dK_dP(self, X):
        """Calculate the gradient of the kernel wrt the parameters.
        :author: Sifan Jiang

        :param X: The latent variable matrix [N x q].
        :type X: class 'numpy.ndarray'
        :return: The gradient of the kernel wrt the parameters.
        :rtype: list
        """
        # K = self(X)
        N, D = X.shape

        dist = X.reshape(N,1,D) - X.reshape(1,N,D)
        dist = np.sum(np.square(dist),-1)

        d_theta = np.exp(- self.gamma / 2 * dist)
        d_gamma = - 1 / 2 * dist * self.theta * np.exp(- self.gamma / 2 * dist)
        d_bias = np.ones((N, N))
        d_white = np.eye(N)
        # d_theta = self.theta * np.exp(- self.gamma / 2 * dist) # exp(params)
        # d_gamma = - 1 / 2 * self.theta * self.gamma * dist  * np.exp(- self.gamma / 2 * dist)

        return [d_theta, d_gamma, d_bias, d_white]

    def dK_dX(self, X, n=None, j=None):
        """Calculate the gradient of the kernel with respect to the latent variable X
        :author: Sifan Jiang

        :param X: The latent variables.
        :type X: class 'numpy.ndarray'
        :param n: The index of the latent variable to be calculated for the gradient, defaults to None
        :type n: int, optional
        :param j: The dimension of the latent vaiable to be calculated for the graident, defaults to None
        :type j: int, optional
        :return: The gradient of the kernel wrt a certain latent variable.
                 Or a list of the gradient of the kernel wrt all the latent vairables.
        :rtype: class 'numpy.ndarray' / list
        """
        K = self(X)
        N, D = X.shape

        if n is None and j is None:
            gradList = []
            for n in range(N):
                for j in range(D):
                    grad = np.zeros((N,N))
                    grad[n,:] = - self.gamma * (X[n,j] - X[:,j]) * K[n,:]
                    grad[:,n] = grad[n,:]
                    gradList.append(grad.copy())
            return gradList

        grad = np.zeros((N,N))
        grad[n,:] = - self.gamma * (X[n,j] - X[:,j]) * K[n,:]
        grad[:,n] = grad[n,:]
        return grad

    def set_params(self, params):
        """Set the parameters of the kernel.
        :author: Sifan Jiang
        
        :param params: The parameters of the kernel to be set.
        :type params: class 'numpy.ndarray'
        """
        assert params.size == self.num_params, "[Error]: Number of parameters is wrong when setting the parameters of kernel."
        self.theta, self.gamma, self.bias, self.white = np.log(1.0 + np.exp(params.copy()))
        # self.theta, self.gamma = params
        # self.theta, self.gamma = np.exp(params)

    def get_params(self):
        """Get the parameters of the kernel.
        :author: Sifan Jiang
        
        :return: The parameters of the kernel.
        :rtype: class 'numpy.ndarray'
        """        
        return np.array([self.theta, self.gamma, self.bias, self.white])
        # return np.log(np.array([self.theta, self.gamma]))


class MLP():
    """The class of MLP kernel.
    $$
        k(x_i,x_j) = \theta * 'sin^{-1} \left(\frac{w x_i^T x_j + b}{\sqrt{(w x_i^T x_i + b + 1)(w x_j^T x_j + b + 1)}}\right)
    $$
    :author: Sifan Jiang
    """
    def __init__(self, params):
        """Initialization of the kernel.
        :author: Sifan Jiang
        
        :param params: Parameters of the kernel.
        :type params: class 'numpy.ndarray'
        """
        self.theta, self.w, self.b = np.log(1.0 + np.exp(params.copy()))
        self.num_params = params.size

    def __call__(self, X, Y=None):
        """Call funtion of the kernel, which calculates the kernel matrix of two matrix.
        If Y is None, the kernel matrix K(X, X) is calculated.
        :author: Sifan Jiang
        
        :param X: Latent variable matrix.
        :type X: class 'numpy.ndarray'
        :param Y: Latent variable matrix, should have same dimension D with X, defaults to None
        :type Y: class 'numpy.ndarray', optional
        :return: The kernel matrix.
        :rtype: class 'numpy.ndarray'
        """
        Nx, Dx = X.shape
        
        if Y is None:
            Y = X
        Ny, Dy = Y.shape

        prod_xy = X.reshape(Nx,1,Dx) * Y.reshape(1,Ny,Dy)
        prod_xx = X.reshape(Nx,1,Dx) * X.reshape(1,Nx,Dx)
        prod_yy = Y.reshape(Ny,1,Dy) * Y.reshape(1,Ny,Dy)

        linear_xy = self.w * np.sum(prod_xy,-1) + self.b
        linear_xx = self.w * np.sum(prod_xx,-1) + self.b + 1
        linear_yy = self.w * np.sum(prod_yy,-1) + self.b + 1

        return self.theta * np.arcsin(linear_xy / np.sqrt(linear_xx * linear_yy))
    
    def dK_dP(self, X):
        """Calculate the gradient of the kernel wrt the parameters.
        :author: Sifan Jiang

        :param X: The latent variable matrix [N x q].
        :type X: class 'numpy.ndarray'
        :return: The gradient of the kernel wrt the parameters.
        :rtype: list
        """
        K = self(X)
        N, D = X.shape

        xx = X.reshape(N,1,D) * X.reshape(1,N,D)
        xx = np.sum(xx,-1)

        numer = self.w * xx + self.b
        denom = self.w * xx + self.b + 1

        frac = numer / denom

        d_theta = K / self.theta
        d_w = self.theta / np.sqrt(1 - frac ** 2) * (xx  / denom - numer * xx * np.sqrt(denom))
        d_b = self.theta / np.sqrt(1 - frac ** 2) * (1 / denom - numer * np.sqrt(denom))

        return [d_theta, d_w, d_b]

    def dK_dX(self, X):
        pass

    def set_params(self, params):
        """Set the parameters of the kernel.
        :author: Sifan Jiang
        
        :param params: The parameters of the kernel to be set.
        :type params: class 'numpy.ndarray'
        """      
        assert params.size == self.num_params, "[Error]: Number of parameters is wrong when setting the parameters of kernel."
        self.theta, self.w, self.b = np.log(1.0 + np.exp(params.copy()))

    def get_params(self):
        """Get the parameters of the kernel.
        :author: Sifan Jiang
        
        :return: The parameters of the kernel.
        :rtype: class 'numpy.ndarray'
        """          
        return np.array([self.theta, self.w, self.b])

