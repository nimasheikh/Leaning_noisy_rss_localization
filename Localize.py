######## one parameter of noise per each access point and location ########
########          Training papameters using autograd                ########

import torch
from gaussian_process import gp, RBF



locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('sameh_data_clean_np/Locations.npy'), np.load('sameh_data_clean_np/Radio_map_full.npy'), \
    np.load('sameh_data_clean_np/Radio_map_mean.npy'), np.load('sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('sameh_data_clean_np/Radio_map_var.npy')


rp_mean_5 = cl_rp_mean_5
rp_var_5 = rp_var = rp_var[:, 0:5]
rp_full_5 = rp_full[:, :, 0:5]




kernel = RBF(1)

def localize(signal , init_point, radio_map, location, var = None,return_likelihood = False,
                                                    return_signal = False,  kernel = RBF(1)):
    if (var is None): 
        var = torch.ones(radio_map.shape)
    
    G = []
    W_ = radio_map
    l_ = location
    signal_ = signal
    init_point_ = init_point
    active_aps = torch.where( signal_ != -110)[0]
    ap_signal_pair = [] 
    est_signal = torch.empty([len(active_aps)])
    
    n_i = 0
    for i in active_aps:
        ap_i_neighbours = torch.where(W_[:, i] != -110)[0]
        ap_i_signal = W_[ap_i_neighbours, i]
        ap_i_var = var[ap_i_neighbours, i]
        ap_i_neighbours_locations = l_[ap_i_neighbours]
        ap_signal_pair.append( signal_[i] )
        G.append( gp(kernel , ap_i_neighbours_locations, ap_i_signal , ap_i_var ))
        est_signal[n_i] = G[-1].predict(init_point_.reshape(-1,2))
        n_i +=1
    if return_signal == True: 
        return est_signal

    def f(x):
        s = 0 
        for i in range(len(G)):
            s += -G[i].log_likelihood( x.reshape(-1, 2), ap_signal_pair[i] )
        return s

    
    
    if return_likelihood:
        return f(init_point)
    
    




class Localize:
    
    def __init__(self, radio_map,  location, var = None, prior = 0 , kernel = RBF(), alpha = 1e-5, cl_radio_idx = None):
        
        if (cl_radio_idx is None):
            cl_radio_idx = np.arange(len(radio_map))
        if (var is None): 
            var = torch.ones(radio_map.shape)
        
        self.alpha = alpha
        self.prior = prior 
        self.kernel = kernel
        self.var = var
        self.W_ = radio_map
        self.l_ = location
        self.G = []
        self.cl_radio_idx = []
        for i in range(self.W_.shape[1]):
            idx = torch.where(self.W_[:,i] != -110)[0]
            if len(idx) == 0 :
                self.cl_radio_idx.append(None)
                self.G.append(None)
                continue
            self.G.append(gp(self.kernel, self.l_[idx], self.W_[idx,i], self.var[idx,i],alpha = self.alpha, prior = self.prior))
            cl_in_idx = np.in1d(idx, cl_radio_idx)
            new_cl_in_idx = np.arange(len(idx))[cl_in_idx]
            self.cl_radio_idx.append(new_cl_in_idx)            







    def update_var(self, var):
        
        self.var = var
        if self.var.shape != self.W_.shape:
            self.var = torch.multiply(self.var, torch.ones(self.W_.shape))
        self.G = []
        for i in range(self.W_.shape[1]):
            idx = torch.where(self.W_[:,i] != -110)[0]
            if len(idx) == 0 :
                self.G.append(None)
                continue
            self.G.append(gp(self.kernel, self.l_[idx], self.W_[idx,i], self.var[idx,i],alpha = self.alpha, prior = self.prior))
    
   
    def updat_var_2(self, var):
        """
        using the matrix identity: (A + D)^-1 = (I + A^-1D)^-1 A^-1
        and the fact that D is a diagonal matrix we don't 
        need to build the whole model from scratch!!!
        

        each element of the diagonal should be added one at a time

        thus more efficient than udpat_var 2
        """
        self.var = var
        if self.var.shape != self.W_.shape:
            self.var = torch.multiply(self.var, torch.ones(self.W_.shape))
   
        for i in range(self.W_.sape[1]):
            idx = torch.where(self.W[:, i] != -110)[0]
            if len(idx) == 0:
                self.G.append(None)
                continue
            self.G[i].update_var(self.var[:,i])


      

   
   
    def update_kernel(self, kernel):
        self.kernel = kernel
        self.G = []
        for i in range(self.W_.shape[1]):
            idx = torch.where(self.W_[:,i] != -110)[0]
            if len(idx) == 0 :
                self.G.append(None)
                continue
            self.G.append(gp(self.kernel, self.l_[idx], self.W_[idx,i], self.var[idx,i],alpha =self.alpha, prior = self.prior))
        
    
    def signal(self, signal_location, signal):
        signal_ap = torch.where(signal != -110)
        signal_ = -110 * torch.ones(signal.shape)
        for i in torch.unique(signal_ap[1]):
            idx_i = signal_ap[0][signal_ap[1] == i]
            signal_[idx_i, i] = torch.tensor(self.G[i].predict(signal_location[idx_i].reshape(-1,2)), \
                dtype = torch.float).squeeze()

        return signal_

    def n_marginal_log_likelihood(self):
        nmll = 0
        for i in range(len(self.G)):
            if self.G[i] is None:
                continue
            nmll = nmll - self.G[i].log_marginal_likelihood()
        return nmll

    
    def signal_est_error(self):
        see_1 = 0
        see_2 = 0 
        for i in range(len(self.G)):
            if self.G[i] is None:
                continue
            dif1, dif2 = self.G[i].signal_est_error()
            see_1 = see_1 + dif1
            see_2 = see_2 + dif2
        
        return see_1, see_2
    
    
    
    def n_log_likelihood(self, signal, signal_location):
        signal_ap = torch.where(signal != -110)[0]
       
        nll = 0 
        for i in signal_ap:
            nll = nll - self.G[i].log_likelihood( signal_location.reshape(-1,2), signal[i])
        
        return nll

    def n_log_likelihood2(self, signal, signal_location):
        signal_ap = torch.where(signal != -110)[0]
       
        nll = np.zeros(len(signal))
        for i in signal_ap:
            nll = nll - self.G[i].log_likelihood( signal_location.reshape(-1,2), signal[i])
        
        return nll



    
    def n_LOO_CV(self):
        pll= 0 
        for i in range(len(self.G)):
            if self.G[i] is None:
                continue
            pll -= self.G[i].LOO_CV(self.cl_radio_idx[i])
        return pll






        

def location_est(signal, radio_map, location, var = None, lr=1e-7):
    if var is None:
        var = torch.ones(radio_map.shape, dtype = torch.float)



    model = Localize(radio_map, location, var)
    n_log_likelihood_floor_plan = []
    for i in location:
        n_log_likelihood_floor_plan.append(model.n_log_likelihood(signal.squeeze(), i))
    init_idx = np.argsort(n_log_likelihood_floor_plan)[:4]
    init_point = torch.mean(location[init_idx], axis = 0)
    print(init_point)
    l = torch.tensor(init_point,dtype = torch.float, device = torch.device("cpu"), requires_grad = True)
    #model = lambda x: localize(signal , x, radio_map, location, var, return_likelihood = True)    SHAME!!!
    
    step = torch.tensor([1, 1], dtype = torch.float)
    n_iter = 0
    
    while (torch.linalg.norm(step).item() > 1e-5):
        loss = model.n_log_likelihood(signal, l)
        l.retain_grad()
        print(l)
        loss.backward()
        step = lr * l.grad
        l = l - lr * l.grad
        n_iter += 1
        if n_iter >= 1500:
            print('Exceeded number of iterations')
            break
    return l.detach() ,loss.item()
    




























