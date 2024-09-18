import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import scipy
import BayesianClassifierHW2 as BC
import MaxLikelihoodEstimator as ML


class ImageClassifier(ML.ML_Estimator):
    def __init__(self):
        ML.ML_Estimator.__init__(self)
    
    def convert_img_color_space(self, training_img, color_space='chromatic'):
            ## make default color space of training image is RGB
        self.training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
        R = self.training_img[:,:,0]
        G = self.training_img[:,:,1]
        B = self.training_img[:,:,2]
        np.seterr(divide='ignore', invalid='ignore')
        if color_space=='chromatic':
            r = R / self.training_img.sum(axis=2)
            g = G / self.training_img.sum(axis=2)
            new_layer= np.zeros((R.shape[0], R.shape[1])) #create all black channel so that image can be plotted
            self.rg_img = np.dstack((r, g, new_layer))
        if color_space=='YCbCr':
            Y  = (0.299 * R) + (0.587 * G) + (0.114 *B)
            Cb = (-0.169 *R) - (0.332*G) + (0.500 *B)
            Cr = (0.500*R) - (0.419*G) - (0.081*B)
            self.YCbCr_img = np.dstack((Y, Cb, Cr)) 
            
    def categorize_features(self, ref_img, color_space='chromatic'):
        black_pixels = np.where((ref_img[:, :, 0] == 0) & (ref_img[:, :, 1] == 0) & (ref_img[:, :, 2] == 0))
        non_black_pixels = np.where((ref_img[:, :, 0] != 0) & (ref_img[:, :, 1] != 0) & (ref_img[:, :, 2] != 0))
        self.ref_img = ref_img
        self.ref_img[non_black_pixels] = [255, 255, 255]
        self.masked_img_RGB = cv2.bitwise_and(self.training_img, self.ref_img)
        
        if color_space=='chromatic':
            samples = self.rg_img[non_black_pixels][:,0:2]
            non_skin_samples = self.rg_img[black_pixels][:,0:2]
        if color_space=='YCbCr':
            samples = self.YCbCr_img[non_black_pixels][:,1:3] 
            non_skin_samples = self.YCbCr_img[black_pixels][:,1:3]
            
        self.n_features= samples.shape[1]
        self.samples = samples.reshape(1,samples.shape[0], samples.shape[1])
        self.convert_to_pandas_df()
        #assign all skin data and get x,y coordinates for each point
        self.df.true_class = 'skin'
        self.df['pixel_x'] = non_black_pixels[1]
        self.df['pixel_y'] = non_black_pixels[0]
        #assign all non-skin data and get x,y coordinates for each point
        non_skin_df = pd.DataFrame(non_skin_samples).rename(columns={0:'x1',1:'x2'})
        non_skin_df['pixel_x'] = black_pixels[1]
        non_skin_df['pixel_y'] = black_pixels[0]
        non_skin_df['true_class'] = 'non_skin'
        #merge skin and non-skin data
        self.df = pd.concat([self.df, non_skin_df])
        
    def calculate_decision_boundary(self,color_space='chromatic',params=None, training_size='all',):
        if params is None:
            self.get_ML_param_estimates(training_size=training_size)
            self.mu = getattr(self, 'ML_mu_estimate_'+ str(training_size).replace(".", "_"))
            self.cov = getattr(self, 'ML_cov_estimate_'+ str(training_size).replace(".", "_")) 
        else: #params should be mu, cov in list type
            self.mu = params[0]
            self.cov= params[1]
        
        self.c = 1/(((2*np.pi)**(self.n_features/2))* (np.linalg.det(self.cov[0]))**0.5 )
        multivariate_gaussian = scipy.stats.multivariate_normal(mean=self.mu[0], cov=self.cov[0])
        self.df['g_x']= multivariate_gaussian.pdf(self.df[['x1','x2']])

    def get_ROC_curves(self):
        t = np.arange(0,21)*self.c/20
        accuracy_rates = []
        for i in t:
            t_bool = self.df.g_x > i
            TP = len(self.df[(t_bool==True) & (self.df['true_class']=='skin')])
            FP = len(self.df[(t_bool==True) & (self.df['true_class']=='non_skin')])
            TN = len(self.df[(t_bool==False) & (self.df['true_class']=='non_skin')])
            FN = len(self.df[(t_bool==False) & (self.df['true_class']=='skin')])
            FPR = FP/(FP+TN)
            FRR = FN/(FN+TP)
            accuracy_rates.append([i,TP,FP,TN,FN, FPR,FRR])
            self.ROC_df = pd.DataFrame(accuracy_rates).rename(columns={0:'t', 1:'TP', 2:'FP', 3:'TN', 4:'FN', 5:'FPR', 6:'FRR'})
    
    def get_EER_thresh(self, x0=0):
        FPR_function = scipy.interpolate.interp1d(self.ROC_df.t, self.ROC_df.FPR)
        FRR_function  = scipy.interpolate.interp1d(self.ROC_df.t, self.ROC_df.FRR)
        def diff(t):
            return np.abs(FPR_function(t) - FRR_function(t))
        self.EER_thresh = scipy.optimize.fsolve(diff, x0 = x0)
        self.EER_value =  FRR_function(self.EER_thresh)
    
    def filter_img_with_EER_thresh(self):
        self.df['EER_thresh_bool'] = self.df.g_x > self.EER_thresh[0]
        non_skin_df = self.df[self.df['EER_thresh_bool'] ==False]
        non_skin_locs = (np.array(non_skin_df.pixel_y), np.array(non_skin_df.pixel_x))
        self.EER_filtered_image=  self.training_img.copy()
        self.EER_filtered_image[non_skin_locs] = [255,255,255]
    
    def classify_image(self, training_img, ref_img, color_space, analysis_type='train', params=None, x0=0):
        self.convert_img_color_space(training_img = training_img, color_space=color_space)
        self.categorize_features(ref_img=ref_img, color_space=color_space)
        
        if analysis_type=='train':
            self.calculate_decision_boundary() 
        elif analysis_type=='test':
            self.calculate_decision_boundary(params=params)    
            
        self.get_ROC_curves()
        self.get_EER_thresh(x0=0)
        self.filter_img_with_EER_thresh()
        
 
