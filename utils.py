import torch
from kernel import RBF
import numpy as np


locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('sameh_data_clean_np/Locations.npy'),\
                np.load('sameh_data_clean_np/Radio_map_full.npy'), np.load('sameh_data_clean_np/Radio_map_mean.npy'),\
                np.load('sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('sameh_data_clean_np/Radio_map_var.npy')
        

R = torch.from_numpy(cl_rp_mean_5)
L = torch.from_numpy(locations)





def soft_knn(Radio, Loc, x, kernel = RBF()):
    x = x.reshape(-1, 5)
    sim = kernel(x, Radio)
    w_ = sim / torch.sum(sim, axis = -1).reshape(-1,1)

    l_hat = torch.einsum('hi,ij->hj', w_, Loc)

    return l_hat



def add_noise_RSS(Radio_map, cr_idx, noise_scale = 1):
    noise_scale_vec = np.zeros(Radio_map.shape)
    noise_scale_vec[cr_idx, :] = noise_scale
    Noise = torch.from_numpy(np.random.normal(loc =0 , scale = noise_scale_vec))
    
    no_signal_ap = torch.where(Radio_map == -110)
    Radio_noisy = Radio_map + Noise
    Radio_noisy[no_signal_ap] = -110

    return Radio_noisy




def split_data(Data_x, Data_y, len_test, cr_data_portion):
    
    idx = np.arange(len(Data_x))
    idx_test = np.random.choice(idx, len_test, replace = False)

    idx_radio_map = idx[ ~np.in1d(idx, idx_test)]
    n_radio_map_idx = np.arange(len(idx_radio_map))
    len_cr_data = int( len(n_radio_map_idx) * cr_data_portion)
    cr_idx = np.random.choice(n_radio_map_idx, len_cr_data, replace = False)
    cl_idx = n_radio_map_idx[ ~np.in1d(n_radio_map_idx, cr_idx)]

    radio_map = Data_x[idx_radio_map]
    radio_map_loc = Data_y[idx_radio_map]

    test = Data_x[idx_test]
    test_loc = Data_y[idx_test]

    return radio_map, radio_map_loc, test, test_loc, cr_idx, cl_idx
