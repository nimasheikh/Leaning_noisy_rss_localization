######## one parameter of noise per each access point and location ########
########          Training papameters using autograd                ########
import numpy as np
import torch
from kernel import RBF



locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('Sameh_data_clean_np/Locations.npy'), np.load('Sameh_data_clean_np/Radio_map_full.npy'), \
    np.load('Sameh_data_clean_np/Radio_map_mean.npy'), np.load('Sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('Sameh_data_clean_np/Radio_map_var.npy')

L = torch.from_numpy(locations)
R = torch.from_numpy(cl_rp_mean_5)



kernel = RBF(1)
def data_split(D, L, radio_size_portion = .8):
    radio_size = int(len(D) * radio_size_portion)
    test_size = len(D) - radio_size


    per = torch.randperm(len(D))
    radio_idx , test_idx = per[:radio_size], per[radio_size:]


    return (R[radio_idx], L[radio_idx]), (R[test_idx], L[test_idx])






def sknn(Radio_rss, Radio_loc, rss, kernel_):
    rss_ = rss.reshape(-1, 5)
    w = kernel_(rss_, Radio_rss)
    normal_w = w/ torch.sum(w, axis = -1)
    return torch.mm(normal_w, Radio_loc)



class sknn_model(torch.nn.Module):
    def __init__(self, Radio, Loc, sigma):
        super().__init__()
        self.Radio = Radio
        self.Loc = Loc
        sigma.requires_grad = True
        self.sigma = torch.nn.Parameter(sigma)



    def forward(self, rss):
        kernel_ = RBF(self.sigma)
        return sknn(self.Radio, self.Loc, rss, kernel_)







class sknn_cv(torch.nn.Module):
    def __init__(self, Radio, Loc, sigma):
        super().__init__()
        self.Radio = Radio
        self.Loc = Loc
        sigma.requires_grad = True
        self.sigma = torch.nn.Parameter(sigma)
        self.idx = torch.arange(len(Radio))

    def forward(self):
        loss = 0
        kernel_ = RBF(self.sigma)
        for i in range(len(self.Radio)):
            radio_idx = self.idx[ self.idx != i]
            test_rss, test_loc, radio, loc = self.Radio[i], self.Loc[i],\
                                            self.Radio[radio_idx], self.Loc[radio_idx]
            loss += torch.linalg.norm( sknn(radio, loc, test_rss, kernel_) - test_loc)

        return loss / len(self.Radio)
