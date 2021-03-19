######## one parameter of noise per each access point and location ########
########          Training papameters using autograd                ########
import numpy as np
import torch
from kernel import RBF



locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('sameh_data_clean_np/Locations.npy'), np.load('sameh_data_clean_np/Radio_map_full.npy'), \
    np.load('sameh_data_clean_np/Radio_map_mean.npy'), np.load('sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('sameh_data_clean_np/Radio_map_var.npy')


rp_mean_5 = cl_rp_mean_5
rp_var_5 = rp_var = rp_var[:, 0:5]
rp_full_5 = rp_full[:, :, 0:5]




kernel = RBF(1)
def sknn(Radio_rss, Radio_loc, rss):
    rss_ = rss.reshape(-1, 1)
    w = kernel(Radio_rss, rss)
    normal_w = w/ torch.sum(w, axis = -1)
    return torch.mm(normal_w, Radio_loc)
