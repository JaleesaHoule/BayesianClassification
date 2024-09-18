import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import BayesianClassifierHW2 as BC


class ML_Estimator(BC.SampleData):
    def __init__(self, mu=None, cov=None, n_samples=None, priors=None, seed=1):
        BC.SampleData.__init__(self, mu, cov, n_samples, priors=None, seed=1)
    
    def get_ML_param_estimates(self, pdf='normal', corr = True, round_estimates=False, training_size='all', print_params=False):
        # TODO: add sympy functionality for solving ML estimate using any pdf/pmf 
        if pdf=='normal':
            mu_hat = []
            cov_hat= []
            if training_size =='all':
                samples= self.samples
            else: 
                samples = self.get_training_sample(training_size)
            
            var_name = 'training_samples_' + str(training_size).replace(".", "_")
            setattr(self, var_name, samples)
            
            n_classes = len(self.samples)
            for i in range(n_classes):
                n = samples[i].shape[0] # length of sample
                n_features= samples[i].shape[1] 
                mu_hat.append([1/n * np.sum(samples[i][:,j]) for j in range(n_features)])
                if corr == True: #no assumptions made about feature correlation
                    cov_hat.append(1/n* np.sum([(samples[i][j,:].reshape(n_features,1) - np.array(mu_hat[i]).reshape(n_features,1)) @ (samples[i][j,:].reshape(n_features,1) - np.array(mu_hat[i]).reshape(n_features,1)).T for j in range(n)], axis=0))
                if corr==False: #assuming features are not correlated
                    _cov_ = 1/n* np.sum([(samples[i][j,:].reshape(n_features,1) - np.array(mu_hat[i]).reshape(n_features,1)) @ (samples[i][j,:].reshape(n_features,1) - np.array(mu_hat[i]).reshape(n_features,1)).T for j in range(n)], axis=0)
                    cov_hat.append(_cov_ * np.eye(n_features)) # make off-diagonal elements zeros
                
            if round_estimates == True:
                mu_hat= np.round(mu_hat,1)
                cov_hat=np.round(cov_hat,1)
            
            if print_params== True:
                print('\n ****************************************************************** \n')
                print('ML Param Estimates of Randomly Generated Samples: \n')
                [print('Number of samples used from class',(i+1),':', len(getattr(self, var_name)[i])) for i in range(len(getattr(self, var_name)))]
                print('\n ML Estimated Means: \n') 
                [print('Sample',i+1,':', mu_hat[i]) for i in range(len(mu_hat))]
                print('\n ML Estimated Covariances: \n')
                [print('Sample',i+1,':', cov_hat[i]) for i in range(len(cov_hat))]
                print('\n ****************************************************************** \n')
            
            
            setattr(self, 'ML_mu_estimate_'+ str(training_size).replace(".", "_"), np.array(mu_hat))
            setattr(self, 'ML_cov_estimate_'+ str(training_size).replace(".", "_"), cov_hat)
            

    def get_training_sample(self,training_size):
        sample_subset= []
        for i in range(len(self.samples)):
            n_total_samples= self.samples[i].shape[0]
            np.random.seed(self.seed)
            idx = np.random.choice(n_total_samples, int(n_total_samples*training_size))
            sample_subset.append(self.samples[i][idx])
        return sample_subset 
    
    def classify_using_ML_estimates(self, corr=True, round_estimates = False, print_params=False, training_size='all'):
        self.generate_samples()
        self.convert_to_pandas_df()
        self.get_ML_param_estimates(corr=corr, round_estimates = round_estimates, print_params =print_params, training_size=training_size)
        #if self.n_samples is not None:
        self.determine_case_and_params(param_estimates='ML', training_size=training_size)
        self.get_g_equations()
        self.classify_2d_sample()
        self.get_true_error()
        self.Bhattacharyya_error_bound()
        self.classify_by_euclidean_distance()
        self.get_euclidean_error()

