import torch
import numpy as np




def pdist(X,Y, p = 2):
    dif = X[:,None] - Y
    dists = torch.linalg.norm(dif, axis = -1)
    return torch.square(dists)


class RBF():
    def __init__(self, length_scale = 1.0543, A = 1):
        self.length_scale = length_scale
        self.A = A
    def __call__(self, X, Y=None):
        if Y is None:
            dists = pdist(X, X) / self.length_scale ** 2
            K = self.A * torch.exp(-.5 * dists)
            return K
        else:
            dists = pdist(X, Y) / self.length_scale ** 2
            K = self.A * torch.exp(-.5*dists)
            return K 
    
    

class Matern():
    def __init__(self, length_scale = 1, nu = 1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, X, Y = None):
        if Y is None:
            dists = pdist(X, X) / self.length_scale ** 2
        else:
            dists = pdist(X, Y)/ self.length_scale ** 2
        
        if self.nu  == 0.5:
            K = torch.exp(-dists)
        elif self.nu == 1.5:
            K = dists * torch.sqrt(torch.tensor([3.0]))
            K = (1. + K) * torch.exp(-K)
        
        else:
            raise ValueError( 'Computation is expensive')
        
        return K
            



def swap_i_j(A, i,j=-1):
    B = A.clone()
    r_i = A[i,:].clone()
    c_i = A[:,i].clone()
    a_ij = A[i,j].clone()
    a_ji = A[j,i].clone()
    a_ii = A[i,i].clone()
    a_jj = A[j,j].clone()
    B[i,:] = B[j,:]
    B[:,i] = B[:,j]
    B[j,:] = r_i
    B[:,j] = c_i
    B[i,j] = a_ji
    B[j,i] = a_ij
    B[j,j] = a_ii
    B[i,i] = a_jj
    return B

