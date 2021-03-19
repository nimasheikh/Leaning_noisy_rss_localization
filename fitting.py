import numpy as np
import torch
from kernel import RBF
kernel = RBF(1)


def noisy_knn(Radio, Loc, x, kernel_):
    x = x.reshape(-1, 5)
    sim = kernel_(x, Radio)
    w_ = sim / torch.sum(sim, axis = -1).reshape(-1,1)

    l_hat = torch.einsum('hi,ij->hj', w_, Loc)

    return l_hat



class sknn(torch.nn.Module):
    def __init__(self, Radio, Loc):
        super().__init__()
        self.Radio = Radio
        self.Loc = Loc
        self.sigma = torch.nn.parameter(torch.tensor[1.0])



    def forward(self, rss):
        kernel_ = RBF(self.sigma)
        return noisy_knn(self.Radio, self.Loc, kernel_)



if __name__ == "__main__":



    ### Loading data and puting it into tensor form

    locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('Sameh_data_clean_np/Locations.npy'),\
                np.load('Sameh_data_clean_np/Radio_map_full.npy'), np.load('Sameh_data_clean_np/Radio_map_mean.npy'),\
                np.load('Sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('Sameh_data_clean_np/Radio_map_var.npy')


    R = torch.from_numpy(cl_rp_mean_5)
    L = torch.from_numpy(locations)

    # number of times to average over (number of runs)
    num_experiment = 20

    # various noise stds for the noise to be added to cr data
    Noise_scale = np.linspace(0, 45, 46)










pass
