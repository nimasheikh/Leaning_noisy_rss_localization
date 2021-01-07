import torch
from utils import soft_knn
from kernel import RBF
import numpy as np
import numpy


class noisy_soft_knn(torch.nn.Module):
    def __init__(self, Radio, Loc, g, kernel = RBF(length_scale=5)):
        super().__init__()
        self.Radio = Radio
        self.Loc = Loc
        self.g = torch.nn.Parameter(g)
        self.kernel = kernel
        self.softmax = torch.nn.Softmax(dim = 0)
    
    
    def forward(self, x):
        X = x.reshape(-1,5)
        sim = self.kernel(X, self.Radio)
        w = sim / torch.sum(sim, axis = -1).reshape(-1,1)
        w_ = torch.mm(self.softmax(self.g), w.T).T
        l_hat = torch.einsum('hi,ij->hj', w_, self.Loc)
        return l_hat



def train_model(Radio, Radio_loc, val_data, val_data_loc, g_init = None, learning_rate = 1 ):
    if g_init is None:
        g_init = torch.ones([len(Radio), len(Radio)], dtype = torch.double)
        g_init[np.arange(len(Radio)), np.arange(len(Radio))] = 10.0
     
    g = g_init
    

    localizer = noisy_soft_knn(Radio, Radio_loc, g)
    optimizer = torch.optim.SGD(localizer.parameters(), lr = learning_rate, momentum = 0.9)
    for i in range(1500):

        error = torch.mean(torch.linalg.norm(localizer(val_data) - val_data_loc, axis = -1))
        print(i, error.item())

        error.backward()
        optimizer.step()
        optimizer.zero_grad()
    

    
    return localizer







