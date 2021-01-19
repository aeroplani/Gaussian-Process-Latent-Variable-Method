from math import exp, pi
from numpy.linalg import eig
from scipy.optimize import fmin_cg
from scipy.spatial.transform import Rotation
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from scipy import stats

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import GP
import KernelPCA
import kernels


class GPLVM():
    
    def __init__(self, Y, q, d, beta, ITER, kernel=None):
        """Initialization
        :author: Sifan Jiang, Hongsheng Chang, Muhammed Memedi

        :param Y: Data
        :type Y: class 'numpy.ndarray'
        :param q: The dimension of each latent variable 
        :type q: int
        :param d: The length of active set.
        :type d: int
        :param beta: Bias in linear kernel, would be overwritten when using linear kernel
        :type beta: float
        :param ITER: The number of iterations.
        :param kernel: The kernel used, defaults to None
        :type kernel: class 'kernels.*', optional
        """
        self.scaler = StandardScaler()
        self.Y = Y.copy()
        self.N, self.D = self.Y.shape # N: len of data-set; D: dimension of date-set
        self.q = q
        self.d = d if d < self.N else self.N # The length of active-set
        self.beta = beta
        self.I = None
        self.X = self.PCA() # Initialize the latent variable X with PCA
        # pca = PCA(n_components=self.q)
        # self.X = pca.fit_transform(self.Y)
        self.ITER = ITER

        if kernel is None:
            self.kernel = kernels.RBF(np.array([1.0, 1.0, exp(-1), exp(-1)]))
        else:
            self.kernel = kernel

        self.GP = GP.GP(self.X, self.Y, self.kernel)
        self.K = self.kernel(self.X)

    def PCA(self):
        """Initial optimization of the latent variable X = ULV'.
        Linear PCA.
        :author: Sifan Jiang
        
        :return: Initialized latent variables.
        :rtype: class 'numpy.ndarray'
        """
        _, v = eig(np.dot(self.Y, self.Y.T))
        w, _ = eig(np.dot(self.Y, self.Y.T) / self.D)
        v = np.real(v)
        w = np.real(w)

        U = v[:,0:self.q]
        L = np.diag((w[0:self.q] - 1/self.beta)**(-1/2))
        V = np.eye(self.q) # Zero rotation angle
        # if self.q is 2:
        #     theta = np.random.uniform(0, 2*pi)
        #     c, s = np.cos(theta), np.sin(theta)
        #     V = np.array(((c,-s), (s, c)))
        # elif self.q is 3:
        #     V = Rotation.from_rotvec(np.random.uniform(0, 2*pi, size=self.q)).as_matrix()

        X = np.dot(np.dot(U, L), V.T)
        X = self.scaler.fit_transform(X)

        return X

    def opt_kernel_params(self):
        """Optimize and set the kernel parameters using the active set.
        Update the kernel matrix after optimization. 
        :author: Sifan Jiang
        """
        self.GP.set_active(self.X_I, self.Y_I)
        params = self.GP.opt_kernel_params()
        self.kernel.set_params(params)
        self.K = self.kernel(self.X)
        print("The kernel parameter: "+str(self.kernel.get_params()))

    # def full_opt_latent_var(self):
    #     X = fmin_cg(
    #         f=self.f,
    #         x0=self.X,
    #         fprime=self.fprime
    #     )
    #     self.X = X

    # def f(self):
    #     self.GP.set_active(self.X, self.Y)
    #     self.GP.update()
    #     return self.GP.log_likelihood() + self.GP.log_prior()

    # def fprime(self):
    #     self.GP.set_active(self.X, self.Y)
    #     self.GP.update()
    #     dK_dX = self.GP.kernel.dK_dX(self.X)
    #     dL_dX = []
    #     for d in dK_dX:
    #         pass

    def RBF(self, x_i, x_j, params):
        """Calculate the RBF of two latent variable vectors
        :author: Hongsheng Chang
        
        :param x_i: A latent variable vector.
        :type x_i: class 'numpy.ndarray'
        :param x_j: A latent variable vector.
        :type x_j: class 'numpy.ndarray'
        :param params: The kernel parameters.
        :type params: class 'numpy.ndarray'
        :return: The RBF of two latent variable vectors
        :rtype: class 'numpy.ndarray'
        """
        theta, gamma, bias, _ = params
        return theta*np.exp(-(gamma/2) * np.linalg.norm(x_i - x_j) ** 2) + bias

    def k_Ij(self, X_j, ker_para):
        '''update k_Ij made up of rows in I from the jth column of K
        :author: Hongsheng Chang
        
        :param X_j: latent variable to be optimized
        :type X_j: ndarray shape(q,)
        :param ker_para: optmimized kernel parameters
        :type ker_para: list ie,[theta, gamma] for RBF
        :return: rows in I from the jth column of K
        :rtype: ndarray shape(d,)
        '''
        kIj = []
        for i in range(self.d):
            k = self.RBF(self.X_I[i], X_j, ker_para)
            kIj.append(k)
        return np.array(kIj)

        # return np.squeeze(self.kernel(self.X_I, np.atleast_2d(X_j), white=False))

    def P_y_x(self, X_j, *args):
        '''Function for log-likelohood ln[p(y_j|x_j)], eq(12) in Lawrance(2005)
        :author: Hongsheng Chang

        :param X_j: latent variable to be optimized
        :type X_j: ndarray shape(q,)
        :param args: Parameter values passed to f  
        :type args: turple, (ker_para, Y_j)
        :return: p(y_j|x_j) given parameters and x_j
        :rtype: float
        Note: 
        :param ker_para: optmimized kernel parameters
        :type ker_para: list ie,[theta, gamma] for RBF
        :param j: index of datapoint in insctive date-set J
        :type j: int
        '''
        ker_para, j = args # optmimized kernel parameters
        Y_j = self.Y_J[j]
        K_II = self.K[self.I][:,self.I] # K_II keeps same
        k_Ij = self.k_Ij(X_j, ker_para) # update k_Ij
        mu_j = np.matmul(np.transpose(self.Y_I), np.matmul(np.linalg.pinv(K_II),k_Ij))
        sigma_sq_j = self.RBF(X_j, X_j,ker_para) - np.matmul(np.transpose(k_Ij),  np.matmul(np.linalg.pinv(K_II),k_Ij))
        ll = stats.multivariate_normal(mean=mu_j, cov=abs(sigma_sq_j)*np.eye(self.D)).logpdf(Y_j)
        #print('lat',X_j)
        #print('log-likelihood',ll)
        return - ll  / len(self.Y_J)
    
    def grad_wrt_lat(self, X_j, *args):
        '''Function for gradient of log-likelihood, ln[p(y_j|x_j)] eq(12) in Lawrance(2005)
        :author: Hongsheng Chang
        
        :param X_j: latent variable to be optimized
        :type X_j: ndarray shape(q,)
        :param args: Parameter values passed to f  
        :type args: turple, (ker_para, Y_j)
        :return: grad of p(y_j|x_j) given parameters and x_j
        :rtype: nparray shape(q,)
        Note: 
        :param ker_para: optmimized kernel parameters
        :type ker_para: list ie,[theta, gamma] for RBF
        :param j: index of datapoint in insctive date-set J
        :type j: int
        '''
        ker_para, j= args
        Y_j = self.Y_J[j]
        # [theta, gamma] = ker_para
        K_II = self.K[self.I][:,self.I] # K_II keeps same
        k_Ij = self.k_Ij(X_j, ker_para) # update k_Ij
        mu_j = np.matmul(np.transpose(self.Y_I), np.matmul(np.linalg.pinv(K_II),k_Ij))
        sigma_sq_j = abs(ker_para[0] - np.matmul(np.transpose(k_Ij),  np.matmul(np.linalg.pinv(K_II),k_Ij))) # self.RBF(X_j, X_j,ker_para) = theta

        dKIj_dx = np.zeros((self.d, self.q))
        for i in range(self.d):
            dKIj_dx[i] = self.RBF(self.X_I[i], X_j, ker_para) * ker_para[1]*(self.X_I[i]-X_j)
        dmu_dx = np.dot(np.dot(self.Y_I.T, K_II), dKIj_dx) # Dxq matrix
        dsigma_dx = - np.dot(np.dot(k_Ij.T, np.linalg.pinv(K_II)),  dKIj_dx)
        dnorm_dx = -2*np.dot((Y_j-mu_j), dmu_dx)

        grad = -dsigma_dx /2/sigma_sq_j - (dnorm_dx*2*sigma_sq_j - 2*dsigma_dx*np.linalg.norm(Y_j-mu_j)**2)/4/sigma_sq_j**2
        #print('grad',grad)
        return -grad / len(self.Y_J)

    def opt_latent_var(self):
        '''Function to potimize lantent variables
        :author: Muhammed Memedi; modified by Hongsheng Chang

        :param ker_para: kernel parameters optimized from step1
        :type ker_para: list[], (ie. [theta, gamma] for RBF)
        :return: Inactive set of latent variables after ortimization
        :rtype: list[ndarray] shape(n-d,q)
        '''
        N_J = self.N - self.d # len of inactive set J
        params = self.kernel.get_params()
        X = []
        for j in range(N_J):
            args = (params, j) # turple for arguments in f and f'
            x_j = fmin_cg(f=self.P_y_x, x0=self.X_J[j], args=args)
            # x_j = fmin_cg(f=self.P_y_x, x0=self.X_J[j], fprime=self.grad_wrt_lat, args=args)
            X.append(x_j)
        self.X_J = np.array(X)
        self.X[self.J] = np.array(X)
        self.X = self.scaler.fit_transform(self.X)

    def opt_latent_var_full(self):
        params = self.kernel.get_params()
        X = []
        for j in range(self.N):
            args = (params, j) # turple for arguments in f and f'
            x_j = fmin_cg(f=self.logl, x0=self.X[j], args=args)
            X.append(x_j)
        self.X = np.array(X)
    
    def logl(self, X_j, *args):
        ker_para, j= args
        kIj = []
        for i in range(self.N):
            k = self.RBF(self.X[i], X_j, ker_para)
            kIj.append(k)
        kIj = np.array(kIj)
        K = np.copy(self.K)
        K[:,j] = kIj
        K[j,:] = kIj
        ll = -self.D*self.N/2*np.log(2*math.pi) - self.D/2*np.linalg.norm(K) - np.trace(np.dot(np.linalg.pinv(K), np.dot(self.Y,self.Y.T)))
        return -ll/self.N

    def IVM(self):
        '''Sparse Gaussian processes by informative vector machine
        divide data-set into active-set,I and inactive-set J
        :author: Hongsheng Chang
        
        :return I_index: index of active set
        :rtype I_index: ndarray shape(d,)
        :return J_index: index of inactive set
        :rtype J_index: ndarray shape(N-d,)
        '''
        # Note d: length of active-set; N: length of whole data-set
        # Init
        b = 2 # Bias parameter of noise model(1~+inf)
        if self.I is None:
            y = np.ones(self.N) # class of observed data(all set as 1 now) TODO correct?
        else:
            y = np.ones(self.N)
            y[self.I] = -1 
        h = np.zeros(self.N)
        K = self.kernel(self.X) # kernal matrix
        A = np.copy(np.diag(K)) # The initial diagonal matrix of kernel
        M = np.zeros(self.N).reshape((1,self.N)) # TODO correct?
        I_index= [] # list of index of active-set
        J_index = np.arange(0,self.N,1) # list of index of inactive set
        # Iteration to choose active-set
        for i in range(self.d):
            delta = []
            # go through all inactive set for max entropy
            for j in J_index:
                z = y[j] * (h[j]+b)/math.sqrt(abs(1+A[j]))
                alpha = y[j]*stats.norm(0,1).pdf(z)/stats.norm(0,1).cdf(z)/math.sqrt(abs(1+A[j]))
                nu = alpha*(alpha+(h[j]+b)/(1+A[j]))
                delta_j = math.log(abs(1-A[j]*nu) )/2
                delta.append(delta_j)
            act_i = delta.index(max(delta)) # choose the index with max entropy as active set
            I_index.append(J_index[act_i]) # add to active set
            J_index = np.delete(J_index, act_i) # delete from inactive set index
            #update pi,mi
            z = y[act_i] * (h[act_i]+b)/math.sqrt(abs(1+A[act_i]))
            alpha = y[act_i]*stats.norm(0,1).pdf(z)/stats.norm(0,1).cdf(z)/math.sqrt(abs(1+A[act_i]))
            nu = alpha*(alpha+(h[act_i]+b)/(1+A[act_i]))
            p = nu/(1-A[act_i]*nu)
            #update matrix
            l_vector = math.sqrt(abs(p)) * M[:,i]
            l = math.sqrt(abs(1+p*K[act_i,act_i]-np.dot(l_vector.T,l_vector)))
            mu_vector = (math.sqrt(abs(p))*K[:,act_i] - np.dot(M.T,l_vector))/l
            M = np.vstack((M,mu_vector))
            h += alpha*l/math.sqrt(abs(p))*mu_vector
            A = A - mu_vector**2 # TODO correct?
        if self.I is not None:
            inters = np.intersect1d(self.I, np.array(I_index))
            print("IVM repeart",len(inters))
        # update actice-set and inactice set
        self.Y_I = self.Y[I_index]
        self.X_I = self.X[I_index]
        self.Y_J = self.Y[J_index]
        self.X_J = self.X[J_index]
        self.I = np.array(I_index)
        self.J = J_index

        print("IVM finished.")

def gaussian2d(x, mu):
    Sigma = np.eye(2)* 0.02
    n = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n)
    fac = np.einsum('...k,kl,...l->...',x-mu,Sigma_inv,x-mu)
    return np.exp(-fac/2)/N

def plot_figure(X, T, alg, target_names, dataset_name, colors):
    '''plot result and distribuction
    :author Hongsheng Chang

    :param X: [latent variable]
    :type X: [ndarray]
    :param T: [target class]
    :type T: [ndarray]
    :param alg: [algorithm used to get result eg PCA, GPLVM + iter]
    :type alg: [str]
    :param target_names: [description]
    :type target_names: [type]
    :param dataset_name: [description]
    :type dataset_name: [type]
    '''
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X, T)
    score = KNN.score(X, T)*100
    targets = list(set(T))
    fig=plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    if alg[:5] == 'GPLVM':
        i = alg[5:]
        alg = alg[:5]
        # compute distribuction, img size 480*640
        xmin = min(X[:,0])-0.1
        xmax = max(X[:,0])+0.1
        ymin = min(X[:,1])-0.1
        ymax = max(X[:,1])+0.1
        x_axis = np.linspace(xmin, xmax, num = 640)
        y_axis = np.linspace(ymin, ymax, num = 480)
        x_axis,y_axis = np.meshgrid(x_axis,y_axis)
        pos = np.empty(x_axis.shape+(2,))
        pos[:,:,0] = x_axis
        pos[:,:,1] = y_axis
        gray = np.zeros((480,640))
        for x in X:
            gray += np.flip(gaussian2d(pos, x),0)
        thresh = 0.7
        gray[np.where(gray>thresh)] = thresh
        axprops=dict(xticks=[],yticks=[])
        ax0=fig.add_axes(rect,label='ax0',**axprops)
        ax0.imshow(-gray, cmap='gray_r')
        ax1=fig.add_axes(rect,label='ax1',frameon=False)
        ax1.set_xlim((xmin, xmax))
        ax1.set_ylim((ymin, ymax))
    else:
        ax1=fig.add_axes(rect,label='ax1')
        i=''
    ax1.set_title(alg+" KNN score =%.2f"% score)
    for t, c in zip(targets, colors):
        index = np.where(T == t)[0]
        ax1.scatter(X[index,0], X[index,1], c=c, s=3)
    plt.legend(target_names)
    plt.savefig(dataset_name+alg+i)

if __name__ == "__main__":
    def read_oil_data():
        """Read the oil flow data from "oil.txt"
        :author: Sifan Jiang
        
        :return: Targets and datapoints.
        :rtype: class 'numpy.ndarray'
        """        
        import csv
        T = []
        Y = []
        with open('oil.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            csv_list = list(csv_reader)
            for line, datapoint in enumerate(csv_list):
                y = []
                try:
                    for count, value in enumerate(datapoint):
                        if count is 0:
                            t= int(value)
                        elif count > 0 and count < 13:
                            y.append(float(value))
                        else:
                            break
                except:
                    pass
                if len(y) is 12:
                    T.append(t)
                    Y.append(y)

        return np.array(T), np.array(Y)


    dataset_name = "oil"
    if dataset_name is "oil":
        T, Y = read_oil_data()
        target_names = ['1', '2', '3']
    else:
        if dataset_name is "breast_cancer":
            dataset = load_breast_cancer() # 2 classes, 569 samples, 30 dimensions
        elif dataset_name is "digits":
            dataset = load_digits() # 10 classes, 1797 samples, 64 dimensions
        elif dataset_name is "iris":
            dataset = load_iris() # 3 classes, 150 samples, 4 dimensions
        Y = dataset.data
        T = dataset.target
        target_names = list(dataset.target_names)

    Y = StandardScaler().fit_transform(Y) # Standardize the dataset
    targets = list(set(T))

    KNN = KNeighborsClassifier(n_neighbors=3)

    # Select colors for visualization
    target_len = len(target_names)
    all_colors = ['yellowgreen', 'springgreen', 'teal', 'navy', 'darkviolet',
        'dimgray', 'salmon', 'sienna', 'darkorange', 'gold']
    color_index = np.linspace(0, len(all_colors)-1, num=target_len).tolist()
    colors = []
    for index in color_index:
        colors.append(all_colors[int(index)])

    # Set parameters for GPLVM
    q = 2
    d = 100
    beta = 1.0
    iterations = 100

    gplvm = GPLVM(Y, q, d, beta, iterations)
    plot_figure(gplvm.X, T, 'PCA', target_names, dataset_name, colors)


    for iter in range(15):
        gplvm.IVM()
        gplvm.opt_kernel_params()
        #gplvm.IVM()
        gplvm.opt_latent_var()


        # Visualize result
        plot_figure(gplvm.X, T, 'GPLVM'+str(iter), target_names, dataset_name, colors)
