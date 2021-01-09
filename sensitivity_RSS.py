from utils import add_noise_RSS, soft_knn, split_data, R, L
import torch
from Training_matrix_model import train_model
from kernel import RBF
import numpy as np 




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    R, L = R.to(device), L.to(device)
    idx = torch.arange(len(R)).to(device) 
      

    num_exp = 1000
    n_cr_data_portion_cases = 5
    Noise_scale = torch.linspace(0, 45, 19).to(device)
    length_scale = 5

    d_simple_average = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_ignore_noisy_data = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_perfect_data = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])
    d_train_model = torch.zeros([num_exp, n_cr_data_portion_cases, len(Noise_scale)])

    n_test = 44
    Cr_data_portion = torch.linspace(.5, .9, n_cr_data_portion_cases).to(device)
        


    
    for i in range(num_exp):
       
        for c in range(n_cr_data_portion_cases):
            print(i, c)
            cr_data_portion = Cr_data_portion[c]
            radio, radio_loc, test, test_loc, cr_idx, cl_idx = split_data(R, L, n_test, cr_data_portion)
            val_data, val_loc = test[: n_test //2], test_loc[: n_test // 2]
            test, test_loc = test[n_test//2:], test_loc[n_test//2:]

        
            for j in range(len(Noise_scale)):

                noise_scale = Noise_scale[j]
                radio_noisy = add_noise_RSS(radio, cr_idx, noise_scale)

                matrix_model = train_model(radio_noisy, radio_loc, val_data, val_loc).to(device) 
                
                loc_hat_test_simple_avg = soft_knn(radio_noisy, radio_loc, test, RBF(length_scale))
                loc_hat_test_ignore_noisy_data = soft_knn(radio[cl_idx], radio_loc[cl_idx], test, RBF(length_scale))
                loc_hat_test_perfect_data = soft_knn(radio, radio_loc, test, RBF(length_scale))
                loc_hat_test_train_model = matrix_model(test).detach()
        
                d_simple_average[i,c ,j] = torch.mean(torch.linalg.norm(loc_hat_test_simple_avg - test_loc, axis = -1))
                d_train_model[i,c,j] = torch.mean(torch.linalg.norm(loc_hat_test_train_model - test_loc, axis = -1)) 
                if j == 0 :
                    d_ignore_noisy_data[i,c,:] = torch.mean(torch.linalg.norm(loc_hat_test_ignore_noisy_data- test_loc, axis = -1))
                    d_perfect_data[i,c,:] = torch.mean(torch.linalg.norm(loc_hat_test_perfect_data - test_loc, axis = -1))
                
        
        D = [d_simple_average.detach(), d_train_model.detach(), d_ignore_noisy_data.detach(), d_perfect_data.detach()]
        
    torch.save(D,'result/RSS_sensitivity_01_07')
        
    
    pass

