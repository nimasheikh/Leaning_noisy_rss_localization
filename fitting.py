import numpy as np
import torch
from kernel import RBF
from Localize import data_split, R, L, sknn, sknn_cv
kernel = RBF(1)


def noisy_knn(Radio, Loc, x, kernel_):
    x = x.reshape(-1, 5)
    sim = kernel_(x, Radio)
    w_ = sim / torch.sum(sim, axis = -1).reshape(-1,1)

    l_hat = torch.einsum('hi,ij->hj', w_, Loc)

    return l_hat




if __name__ == "__main__":




    sigma = []
    init_sigma = torch.tensor([10.0])
    Loss = []

    S = sknn_cv(R, L, init_sigma)


    optim = torch.optim.SGD(S.parameters(), lr = 1e-1, momentum = 0.9)
    for i in range(100):
        print(i)
        loss = S()
        Loss.append(loss.item())
        sigma.append(S.sigma.item())
        loss.backward()

        optim.step()
        optim.zero_grad()









pass
