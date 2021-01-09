from utils import add_noise_Loc, soft_knn, split_data, R, L
import torch
from Training_matrix_model import train_model
from kernel import RBF
import numpy as np 




if __name__ == "__main__":
    idx = np.arange(len(R)) 
    

    num_exp = 1000

    n_cr_data_portion_cases = 5
    Noise_scale = np.linspace(0, 10, 11)

    length_scale = 5   
 
    d_simple_average = np.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_ignore_noisy_data = np.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_perfect_data = np.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_train_model = np.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])



    Cr_data_portion = np.linspace(0.5, .9, n_cr_data_portion_cases)
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

                matrix_model = train_model(radio, radio_loc_noisy, val_data, val_loc)
                
                loc_hat_test_simple_avg = soft_knn(radio, radio_loc_noisy, test, RBF(length_scale))
                loc_hat_test_ignore_noisy_data = soft_knn(radio[cl_idx], radio_loc[cl_idx], test, RBF(length_scale))
                loc_hat_test_perfect_data = soft_knn(radio, radio_loc, test, RBF(length_scale))
                loc_hat_test_train_model = matrix_model(test).detach()
	
                d_simple_average[i,c,j] = np.mean(np.linalg.norm(loc_hat_test_simple_avg - test_loc, axis = -1))
                d_train_model[i,c,j] = np.mean(np.linalg.norm(loc_hat_test_train_model - test_loc, axis = -1))
                if j == 0:
                    d_ignore_noisy_data[i,c,:] = np.mean(np.linalg.norm(loc_hat_test_ignore_noisy_data- test_loc, axis = -1))
                    d_perfect_data[i,c,:] = np.mean(np.linalg.norm(loc_hat_test_perfect_data - test_loc, axis = -1))
                
        
    D_ = [d_simple_average, d_train_model, d_ignore_noisy_data, d_perfect_data]
        
    
    np.save('result/Loc_sensitivity_01_07', D_)
    print('Using the trained model along side other strategies, learning_momentum =0.9, lr = 1')
    print('cr_data_portion = np.linspace(.5, .9, 5)')
    print('noise_loc = np.linspace(0,10, 11)')

    pass

