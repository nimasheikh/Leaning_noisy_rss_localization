import numpy as np
import torch
from Gaussian_process import gp, RBF
from Localize import Localize, localize, cl_rp_mean_5, locations, location_est
kernel = RBF(5)

def smooth_radio(Radio, cr_idx, cr_pred ,alpha):
    smooth_radio_ = Radio.clone()
    smooth_radio_[cr_idx] = Radio[cr_idx] * (1 - torch.abs(alpha)) + cr_pred * torch.abs(alpha)
    return smooth_radio_


def noisy_knn(Radio, Loc, x):
    x = x.reshape(-1, 5)
    sim = kernel(x, Radio)
    w_ = sim / torch.sum(sim, axis = -1).reshape(-1,1)

    l_hat = torch.einsum('hi,ij->hj', w_, Loc)

    return l_hat

def loss(Radio, Loc, cr_pred, cr_idx, alpha):
    
    smooth_radio_ = smooth_radio(Radio, cr_idx, cr_pred, alpha)

    # seting cr_data as test points
    idx_radio = np.arange(len(Radio))
    idx_cl = idx_radio[~ np.in1d( idx_radio, cr_idx)]
    
    error = 0 
    
    for i in idx_cl:
        idx_ = idx_radio[idx_radio != i]
        smooth_radio_i = smooth_radio_[idx_]
        Loc_ = Loc[idx_]
        test_ = Radio[i]
        test_loc = Loc[i]
        test_loc_hat = noisy_knn(smooth_radio_i, Loc_, test_)
        error += torch.linalg.norm(test_loc - test_loc_hat, axis = -1)
        

    return error/ len(idx_cl)




if __name__ == "__main__":



    ### Loading data and puting it into tensor form

    locations, rp_full, rp_mean , cl_rp_mean_5, rp_var = np.load('sameh_data_clean_np/Locations.npy'),\
                np.load('sameh_data_clean_np/Radio_map_full.npy'), np.load('sameh_data_clean_np/Radio_map_mean.npy'),\
                np.load('sameh_data_clean_np/cl_rp_mean_5.npy'), np.load('sameh_data_clean_np/Radio_map_var.npy')
        

    R = torch.from_numpy(cl_rp_mean_5)
    L = torch.from_numpy(locations)

    # number of times to average over (number of runs)
    num_experiment = 20 
    
    # various noise stds for the noise to be added to cr data
    Noise_scale = np.linspace(0, 45, 46)


    ### The data to be saved
    d_simple_average = np.zeros([num_experiment, len(Noise_scale)])         
    d_ignore_noisy_data = np.zeros([num_experiment, len(Noise_scale)])      
    d_learning_init_1 = np.zeros([num_experiment, len(Noise_scale)])
    d_learning_init_2 = np.zeros([num_experiment, len(Noise_scale)])
    D = np.array([d_simple_average, d_ignore_noisy_data, d_learning_init_1, d_learning_init_2])


    for i in range(num_experiment):
            
        #### defining test data, validation data and radio map
        n_test = 54
        idx = np.arange(224)

        test_idx = np.random.choice(idx, n_test, replace = False)
        radio_idx = idx[~np.in1d(idx, test_idx)]
        test = R[test_idx]
        test_loc = L[test_idx]
        radio = R[radio_idx]
        radio_loc = L[radio_idx]
        n_radio_idx = np.arange(len(radio_idx))

        
        ## Spliting the radio map into clear and crowd sourced (cr) part
        """
        It may happen that the data is splited such that the clean data does not have signals from an specific access point
        in that case our model can predict the signal and an error raises

        In order to prevent the estimation is inside a try except box, if it is not possible to estimate the 
        crowd signal's output based on the clean data, it is left unchanged(no estimation is used)
        """
        cl_idx = np.random.choice(n_radio_idx, len(radio_idx)// 2, replace=False)
        cr_idx = n_radio_idx[~ np.in1d(n_radio_idx, cl_idx)]


        
        for j in range(len(Noise_scale)):

            ## Defining noise ,  setting the elements corresponding to cr with noise
            noise_scale = Noise_scale[j]
            noise_scale_v = np.zeros(radio.shape)
            noise_scale_v[cr_idx, :] = noise_scale
            noise = torch.from_numpy(np.random.normal(loc = 0 , scale = noise_scale_v))

            ## Adding noise to crowd sourced (cr) part of the data
            
            radio_noisy = radio + noise


            ## estimation of CR part of data based on the cl part of the data with GP
            L_ = Localize(radio_noisy[cl_idx], radio_loc[cl_idx])
            cr_predict = L_.signal(radio_loc[cr_idx], radio_noisy[cr_idx])

            ## defining two models _1, _2 ,_3 for three intial values
            alpha_1 = torch.from_numpy(1.0 *  np.ones(radio[cr_idx].shape))
            alpha_1.requires_grad = True
            
            alpha_2 = torch.from_numpy(1.0 *  np.random.uniform(0,1 , size =radio[cr_idx].shape))
            alpha_2.requires_grad = True
            
            alpha_3 = torch.from_numpy(0.1e-2 * np.ones(radio[cr_idx].shape))
            alpha_3.requires_grad = True
            
            
            
            
            
            
            l_1 = lambda x: loss(radio_noisy, radio_loc, cr_predict, cr_idx, x)
            l_2 = lambda x: loss(radio_noisy, radio_loc, cr_predict, cr_idx, x)
            l_3 = lambda x: loss(radio_noisy, radio_loc, cr_predict, cr_idx, x)



            #model update starts from here
            learning_rate = 100
            for iteration in range(1000):
                if iteration > 400:
                    learning_rate = 5
                elif iteration > 600:
                    learning_rate = 5e-1
                elif iteration > 800:
                    learning_rate = 5e-2

                alpha_1.requires_grad = True
                alpha_2.requires_grad = True
                alpha_3.requires_grad = True

                error_1 = l_1(alpha_1)
                error_2 = l_2(alpha_2)
                error_3 = l_3(alpha_3)

                print(i, j , iteration, error_1.item(), error_2.item(), error_3.item())
                
                
                alpha_1.retain_grad()
                alpha_2.retain_grad()
                alpha_3.retain_grad()
                
                error_1.backward()
                error_2.backward()
                error_3.backward()

                with torch.no_grad():
                    alpha_1 = alpha_1 - learning_rate * alpha_1.grad
                    alpha_2 = alpha_2 - learning_rate * alpha_2.grad
                    alpha_3 = alpha_3 - learning_rate * alpha_3.grad

                    alpha_1.grad = None
                    alpha_2.grad = None
                    alpha_3.grad = None


            D[0, i, j] = torch.mean(torch.linalg.norm(noisy_knn(radio_noisy, radio_loc, test)\
                                                     - test_loc, axis = -1)).detach().numpy()       #simple average
            D[1, i, j] = torch.mean(torch.linalg.norm(noisy_knn(radio_noisy[cl_idx], radio_loc[cl_idx], test)\
                                                     - test_loc, axis = -1)).detach().numpy()       #ignore (just cl data)

            smooth_radio_1, smooth_radio_2 = smooth_radio(radio_noisy, cr_idx, cr_predict, alpha_1)\
                                            , smooth_radio(radio_noisy, cr_idx, cr_predict, alpha_1)

            D[2, i, j] = torch.mean(torch.linalg.norm(noisy_knn(smooth_radio_1, radio_loc, test)\
                                                     - test_loc, axis = -1)).detach().numpy()
            D[3, i, j] = torch.mean(torch.linalg.norm(noisy_knn(smooth_radio_2, radio_loc, test)\
                                                     - test_loc, axis = -1)).detach().numpy()
            
            
            
            
            
            








pass