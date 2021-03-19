import numpy as np
import torch
from kernel import RBF
from Localize import data_split, R, L
kernel = RBF(1)


def noisy_knn(Radio, Loc, x, kernel_):
    x = x.reshape(-1, 5)
    sim = kernel_(x, Radio)
    w_ = sim / torch.sum(sim, axis = -1).reshape(-1,1)

    l_hat = torch.einsum('hi,ij->hj', w_, Loc)

    return l_hat



class sknn(torch.nn.Module):
    def __init__(self, Radio, Loc, sigma):
        super().__init__()
        self.Radio = Radio
        self.Loc = Loc
        sigma.requires_grad = True
        self.sigma = torch.nn.Parameter(sigma)



    def forward(self, rss):
        kernel_ = RBF(self.sigma)
        return noisy_knn(self.Radio, self.Loc, rss, kernel_)



if __name__ == "__main__":




    sigma = []
    init_sigma = torch.tensor([10.0])
    Loss = []

    for j in range(10):

        (Radio, Loc), (test_rss, test_loc) = data_split(R, L, radio_size_portion = .99)
        S = sknn(Radio, Loc, init_sigma)
        optim = torch.optim.SGD(S.parameters(), lr = 1e-1)
        for i in range(200):
            print(i)
            loss = torch.sum(torch.linalg.norm((S(test_rss) - test_loc), axis = -1))
            Loss.append(loss.item() / len(test_rss))
            sigma.append(S.sigma.item())
            loss.backward()
            optim.step()
            optim.zero_grad()









pass
