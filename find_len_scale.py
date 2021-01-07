import torch
import numpy
from kernel import RBF
from utils import soft_knn, split_data, R, L


if __name__ == "__main__":
    radio, radio_loc, test, test_loc, _, _ = split_data(R, L, 154, 0)
    
    length_scale = torch.tensor(1,dtype = torch.float,  requires_grad = True)
    learning_rate = 1e-1


    loc = lambda x: soft_knn(radio, radio_loc, test, RBF(x))


    for i in range(500):
        loss = torch.mean(torch.linalg.norm(loc(length_scale) - test_loc, axis = -1))
        #length_scale.requires_grad = True #requires changing 
        length_scale.retain_grad()
        print(i, loss.item())
        loss.backward()
        length_scale = length_scale - learning_rate * length_scale.grad




    
    pass