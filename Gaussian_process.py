import numpy as np
import torch
from kernel import RBF, Matern, swap_i_j


class gp:
    
    def __init__(self , kernel, x, y, sigma, alpha = 1e-5, prior = 0):
        
        self.kernel = kernel
        self.prior = prior
        self.x = x
        self.y = y
        self.var_sigma = sigma
        self.alpha = alpha
        self.kxx_no_var_info = self.kernel(x)
        #self.k_xx = self.kernel(x)           # originaly = self.k_xx = kenrel(X) 
        #self.k_xx[np.diag_indices_from(self.k_xx)] += self.alpha
        
        #self.k_xx[i][np.diag_indices_from(self.k_xx[i])] += self.alpha * np.diag(var[:,i])
        self.k_xx = self.kxx_no_var_info + self.alpha * torch.diag(self.var_sigma ** 2) 
        
        try:
            self.L = torch.cholesky(self.k_xx) 
            # self.L_ changed, self._K_inv needs to be recomputed
             
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel,) + exc.args
            raise
        self.alpha = torch.cholesky_solve(self.y.reshape(-1,1) - self.prior, self.L) 
      
        #self.L = cholesky(self.k_xx )#+ np.diag(self.var))     #line 2 (Algorithm)
        #self.alpha = cho_solve((self.L, True), self.y)       # line 3  (Algorithm)
    
    def predict(self, X, return_std = 0 ):
        
        
        k_xt  = self.kernel(self.x , X.reshape(-1,2))
        y_out = self.prior + torch.einsum('ij,ik->jk', k_xt, self.alpha)         #line 4 (Algorithm)
        #Y = np.matmul(np.matmul(k_xt, np.linalg.inv(self.k_xx + np.diag(self.var))), self.y)
        
        if return_std:
        
            L_inv = solve_triangular(self.L.T,
                                             np.eye(self.L.shape))
            K_inv = L_inv.dot(L_inv.T)

            y_var = np.diag(self.k_xx)
            y_var -= np.einsum("ij,ij->i",
                                   np.dot(k_xt, K_inv), k_xt)

            return y_out, np.sqrt(y_var)

        
        return y_out


    def log_likelihood(self, Position, estimate):
        k_xt  = self.kernel(self.x , Position)
        y_out = self.prior + torch.einsum('ij,ik->jk', k_xt, self.alpha)       # Line 4 (Algorithm)
        
        v = torch.cholesky_solve(k_xt, self.L)   # Line 5
        y_cov = self.kernel(Position) - torch.einsum('ij,ij->j', k_xt, v)  + torch.diag(torch.ones(len(Position))) * 1e-5 # Line 6        

        G = torch.distributions.normal.Normal(y_out.flatten(), torch.sqrt(y_cov)) 
        p = G.log_prob(estimate) 
        return p

    def log_likelihood2(self, Position, estimate):
        k_xt  = self.kernel(self.x , Position)
        y_out = self.prior + torch.einsum('ij,ik->jk', k_xt, self.alpha)       # Line 4 (Algorithm)
        
        v = torch.cholesky_solve(k_xt, self.L)   # Line 5
        y_cov = 1 - torch.einsum('ij,ij->j', k_xt, v)  + torch.diag(torch.ones(len(Position))) * 1e-5 # Line 6        

        G = torch.distributions.normal.Normal(y_out.flatten(), torch.sqrt(y_cov)) 
        p = G.log_prob(estimate) 
        return p



    
    def log_marginal_likelihood(self):
        d_fit = -1/2 * (self.y - self.prior ).dot(self.alpha.squeeze())
        reg = - torch.sum(torch.log(torch.diag(self.L)))
        LML = d_fit + reg              #line 8 (Algorithm)
        return LML
    
    
    def LOO_CV(self,cl_idx = None):
        if cl_idx is None:
            cl_idx = np.arange(len(self.y))
        


        p_mll = 0               #psudo mariginal log_likelihood
        data_fit = 0
        
        k_xx_inv = torch.cholesky_inverse(self.L)
        
        
        for i in cl_idx:
            k_xx_inv_i_last =  swap_i_j(k_xx_inv, i)                    #putting the ith element as the last element (to be ommited)
            k_xx_inv_i =   k_xx_inv_i_last[0:-1, 0:-1] - \
                torch.mm(k_xx_inv_i_last[:-1,-1].reshape(-1,1), k_xx_inv_i_last[-1,:-1].reshape(1,-1)) \
                     / k_xx_inv_i_last[-1,-1]                           #finding the inverse of covariance matrix w/out ith data
            
            
            y_i = self.y.clone()
            y_i[i] = y_i[-1]
            y_i = y_i[:-1]
            k_xt_i = self.k_xx[i].clone()
            #print(k_xt_i)
            k_xt_i[i] = k_xt_i[-1]
            k_xt_i = k_xt_i[:-1]

            v = torch.mm(k_xt_i.reshape(1,-1), k_xx_inv_i).squeeze()    #self.k_xx[i, idx[idx!= i]].reshape(1,-1), k_xx_inv_i).squeeze()
            
            mu_i = torch.dot(v, y_i)           #self.y[idx[idx!=i]])
            sigma2_i = self.kernel(self.x[i].reshape(1,-1), self.x[i].reshape(1,-1)) + self.var_sigma[i] ** 2-  torch.dot(v, k_xt_i) 
            #print(mu_i,sigma2_i, self.y[i]) 
            data_fit = data_fit - (self.y[i] - mu_i) ** 2
            p_mll += -1/2 * torch.log(sigma2_i) - (self.y[i] - mu_i) ** 2/ (2* sigma2_i)

        return  p_mll

   


