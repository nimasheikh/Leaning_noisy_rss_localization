import torch
from utils import soft_knn,split_data, R, L, add_noise_Loc
from Training_matrix_model import train_model
from kernel import RBF
import matplotlib.pyplot as plt
kernel = RBF(5)



if __name__ == "__main__":
    V , T = [], []
    noisy_portion = [.5, .6,.8,.9,.95,.98]
    
    for j in range(len(noisy_portion)):
        V.append([])
        T.append([])
        for i in range(100):
            print(j,i)
            radio_map, radio_map_loc, test, test_loc, cr_idx, cl_idx = split_data(R, L, 104, noisy_portion[j])
            radio_map_loc_noisy = add_noise_Loc(radio_map_loc, cr_idx, 8)
            model, val_acc, test_acc = train_model(radio_map, radio_map_loc_noisy, test, test_loc)
            V[j].append(val_acc)
            T[j].append(test_acc)

    v, t = np.array(V), np.array(T)





pass