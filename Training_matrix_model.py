import torch
from utils import soft_knn
from kernel import RBF
import numpy as np
import numpy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        g_init = torch.ones([len(Radio), len(Radio)], dtype = torch.double).to(device)
        g_init[torch.arange(len(Radio)), torch.arange(len(Radio))] = 10.0
     
    g = g_init
    

    localizer = noisy_soft_knn(Radio, Radio_loc, g)
    optimizer = torch.optim.SGD(localizer.parameters(), lr = learning_rate, momentum = 0.9)
    val1, val1_loc = val_data[:len(val_data) // 2], val_data_loc[: len(val_data) // 2]
    val2, val2_loc = val_data[len(val_data) // 2:], val_data_loc[len(val_data)//2 :]
    localization_accuracy_val = []
    localization_accuracy_test_ = []
    for i in range(1500):

        with torch.no_grad():
            err_test = torch.mean(torch.linalg.norm(localizer(val2) - val2_loc, axis = -1))

        error = torch.mean(torch.linalg.norm(localizer(val1) - val1_loc, axis = -1))
        localization_accuracy_val.append(error.item())
        localization_accuracy_test_.append(err_test)

        if i == 1499:
            print(i, error.item())

        error.backward()
        optimizer.step()
        optimizer.zero_grad()
    

    
    return localizer, localization_accuracy_val, localization_accuracy_test_







