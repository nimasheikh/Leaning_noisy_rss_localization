from utils import add_noise_Loc, soft_knn, split_data, R, L
import torch
from Training_matrix_model import train_model
from kernel import RBF
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_availabe() else 'cpu')
device_2 = torch.device('cpu')


if __name__ == "__main__":
    idx = np.arange(len(R)) 
    

    num_exp = 1000

    n_cr_data_portion_cases = 3
    Noise_scale = np.linspace(0, 15, 16)

    length_scale = 5   
 
    d_simple_average = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_ignore_noisy_data = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_perfect_data = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_train_model = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])



    Cr_data_portion = np.linspace(0.9, .98, n_cr_data_portion_cases)
    n_test = 44

    for i in range(num_exp):
  
        for c in range(n_cr_data_portion_cases):    
            print(i, c)
            cr_data_portion = Cr_data_portion[c]  
            radio, radio_loc, test, test_loc, cr_idx, cl_idx = split_data(R, L, n_test, cr_data_portion)
            val_data, val_loc = test[: n_test // 2], test_loc[: n_test // 2]
            test, test_loc = test[n_test // 2 :], test_loc[n_test // 2 :]



            for j in range(len(Noise_scale)):

                noise_scale = Noise_scale[j]
                radio_loc_noisy = add_noise_Loc(radio_loc, cr_idx, noise_scale)

                matrix_model = train_model(radio, radio_loc_noisy, val_data, val_loc).to(decice)
                
                loc_hat_test_simple_avg = soft_knn(radio, radio_loc_noisy, test, RBF(length_scale))
                loc_hat_test_ignore_noisy_data = soft_knn(radio[cl_idx], radio_loc[cl_idx], test, RBF(length_scale))
                loc_hat_test_perfect_data = soft_knn(radio, radio_loc, test, RBF(length_scale))
                loc_hat_test_train_model = matrix_model(test).detach()
	
                d_simple_average[i,c,j] = torch.mean(torch.linalg.norm(loc_hat_test_simple_avg - test_loc, axis = -1))
                d_train_model[i,c,j] = torch.mean(torch.linalg.norm(loc_hat_test_train_model - test_loc, axis = -1))
                if j == 0:
                    d_ignore_noisy_data[i,c,:] = torch.mean(torch.linalg.norm(loc_hat_test_ignore_noisy_data- test_loc, axis = -1))
                    d_perfect_data[i,c,:] = torch.mean(torch.linalg.norm(loc_hat_test_perfect_data - test_loc, axis = -1))
                
        
    D_ = [d_simple_average..detach().to(device_2), d_train_model.detach().to(device_2), \
            d_ignore_noisy_data.detach().to(device_2), d_perfect_data.detach().to(device_2)]
        
    
    np.save('result/Loc_sensitivity_01_09', D_)
    print('result:[simple_average, train_model, ignore_noisy_data, perfect_data]')
    print('Using the trained model along side other strategies, learning_momentum =0.9, lr = 1')
    print('cr_data_portion = np.linspace(.9, .98, 3)')
    print('noise_loc = np.linspace(0,10, 11)')

    pass

