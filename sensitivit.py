from Learning import R, L
from utils import add_noise_RSS, soft_knn, split_data
import torch
import numpy 
import numpy 




if __name__ == "__main__":
    idx = np.arange(len(R)) 
    

    num_exp = 20
    Noise_scale = np.linspace(0, 45, 19)
    
    d_simple_average = np.zeros([num_exp, len(Noise_scale)])
    d_ignore_noisy_data = np.zeros([num_exp, len(Noise_scale)])
    d_perfect_data = np.zeros([num_exp, len(Noise_scale)])


    n_test, cr_data_portion = 44, 0.5
    radio, radio_loc, test, test_loc, cr_idx, cl_idx = split_data(R, L, n_test, cr_data_portion)
    


    length_scale = 5
    for i in range(num_exp):
        for j in range(len(Noise_scale)):

            noise_scale = Noise_scale[j]
            radio_noisy = add_noise_RSS(radio, cr_idx, noise_scale)
            
            loc_hat_test_simple_avg = soft_knn(radio_noisy, radio_loc, test, RBF(length_scale)).numpy()
            loc_hat_test_ignore_noisy_data = soft_knn(radio[cl_idx], radio_loc[cl_idx], test).numpy()
            loc_hat_test_perfect_data = soft_knn(radio, radio_loc, test, RBF(length_scale)).numpy()
    
            d_simple_average[i,j] = np.mean(np.linalg.norm(loc_hat_test_simple_avg - test_loc, axis = -1))
            d_ignore_noisy_data[i,j] = np.mean(np.linalg.norm(loc_hat_test_ignore_noisy_data- test_loc, axis = -1))
            d_perfect_data[i,j] = np.mean(np.linalg.norm(loc_hat_test_perfect_data - test_loc, axis = -1))
            
    
    
    
    
    
    pass

