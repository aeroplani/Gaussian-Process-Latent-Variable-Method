import numpy as np
import math

# RBF kernel
def RBF(x_i,x_j,theta,gamma):
    return  theta*np.exp(-(gamma/2) * np.linalg.norm(x_i - x_j) ** 2)


def K(x, kernel):
    # N is number of samples
    N = np.shape(x)[1]

    K = np.zeros(N,N)
    for i in range(N):
        for j in range(N):
            K[i][j] = kernel(x[i],x[j], theta, gamma)
    return K

def K_II(x_I,kernel):
    return K(x_I,kernel)
    

# choose active set
def Y_I(Y,I):
    Y_II = [Y[:,i] for i in I]
    return Y_II
        

def Pr_Y_I(K_II, Y_I,D):
    p1= -(D/2)* np.log(2*math.pi)
    p2= -(1/2) * np.log(np.linalg.det(K_II))
    Y_trans = np.transpose(Y_I)
    Y_product =  np.matmul(Y_trans, Y_I)
    p3= -(1/2)* np.trace(np.matmul(K_II, Y_product))
    return p1+p2+p3

def grad_wrt_par(x_I, K_II,Y_i, kernel,theta,gamma):
    dKII_dtheta = (1/theta) * K_II
    #dKII_dgamma
    N = np.shape(x_I)[1]
    dKII_dgamma = np.zeros(N,N)
    for i in range(N):
        for j in range(N):
            dKII_dgamma[i][j] = kernel(x_I[i],x_I[j],theta, gamma) * -(1/2)* np.linalg.norm(x_I[i] - x_I[j]) ** 2


    dlndetKII_dtheta = np.trace(np.matmul(np.inverse(KII), dKII_dtheta))
    dlndetKII_dgamma = np.trace(np.matmul(np.inverse(KII), dKII_gamma))

    YY= np.matmul(Y_I, np.transpose(Y_I))
    dtransKIIYY_dtheta = np.trace(np.matmul(YY,dKII_dtheta))
    dtransKIIYY_gamma = np.trace(np.matmul(YY,dKII_dgamma))

    dF_dtheta= -(1/2)* dlndetKII_dtheta - (1/2)* dtransKIIYY_dtheta
    dF_dgamma= -(1/2)* dlndetKII_gamma - (1/2)* dtransKIIYY_dgamma
    return [dF_dtheta, dF_dgamma]

def opt_kernel_par(iter):
    initial = 0
    new_kernel_par = fmin_cg(Pr_Y_I, initial,fprime = grad_wrt_par, maxIter=iter)
    return new_kernel_par


print('k')