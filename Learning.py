import torch
from utils import soft_knn,split_data, R, L, add_noise_Loc
from Training_matrix_model import train_model
from kernel import RBF
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kernel = RBF(5)



if __name__ == "__main__":
    n_exp = 100
    noisy_portion = [.1, .3 ,.5, .6,.8,.9,.95,.98]
    V,T = torch.zeros([len(noisy_portion), n_exp, 1500]).to(device), torch.zeros([len(noisy_portion), n_exp, 1500]).to(device)


    for j in range(len(noisy_portion)):
        V.append([])
        T.append([])
        for i in range(100):
            print(j,i)
            radio_map, radio_map_loc, test, test_loc, cr_idx, cl_idx = split_data(R, L, 104, noisy_portion[j])
            radio_map_loc_noisy = add_noise_Loc(radio_map_loc, cr_idx, 8)
            _, val_acc, test_acc = train_model(radio_map, radio_map_loc_noisy, test, test_loc)
            V[j, i] = val_acc.detach()
            T[j, i] = test_acc.detach()
        
    Learning_result = torch.tensor(V, T)
    torh.save(Learning_rsult, 'results/learning_result_01_12')

    


pass