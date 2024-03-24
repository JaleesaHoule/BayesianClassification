import numpy as np
import pandas as pd
import sympy as sym
from sympy.calculus.util import continuous_domain
import matplotlib.pyplot as plt
import seaborn as sns

x1,x2 = sym.symbols('x1, x2')

class SampleData:
    '''
    sample inputs:
    mu1 = np.array([1,1])
    cov1 = np.array([[1,0],[0,1]])
    mu2 = np.array([4,4])
    cov2 = np.array([[1,0],[0,1]])

    mu= [mu1,mu2 ]
    cov =[cov1, cov2]
    
    '''

    def __init__(self, mu, cov, n_samples, priors=None, seed=1):
        self.mu = mu
        self.cov = cov
        self.n_samples= n_samples
        self.priors = priors
        self.seed = seed

#**************************************************************************************#        
    # Part A
    #data generation and processing
    def generate_samples(self, print_sample_params=False):
        self.samples = []
        self.sample_means = []
        self.sample_covs = []
        for i in range(len(self.mu)):
            self.samples.append(np.random.default_rng(seed=self.seed).multivariate_normal(self.mu[i], self.cov[i], self.n_samples[i]))
            self.sample_means.append(self.samples[i].mean(axis=0))
            self.sample_covs.append(np.cov(self.samples[i].T))
        if self.priors is None:
            self.priors=np.array([self.n_samples[i]/np.sum(self.n_samples) for i in range(len(self.n_samples))])
        
        if print_sample_params== True:
            print('\n ****************************************************************** \n')
            print('Params of Randomly Generated Samples: \n')
            print('\n Sample Means: \n') 
            [print('Sample',i+1,':', self.sample_means[i]) for i in range(len(self.sample_means))]
            print('\n Sample Covariances: \n')
            [print('Sample',i+1,':', self.sample_covs[i]) for i in range(len(self.sample_covs))]
            print('\n ****************************************************************** \n')
    def convert_to_pandas_df(self):
        self.df = []
        # reshape into dataframe; assign values for their true sample distribution
        dum = []
        for i in range(len(self.samples)):
            d = pd.DataFrame(self.samples[i])
            d.columns = [f'x{i+1}' for i in range(len(d.columns))]
            d['true_class'] = i+1  
            dum.append(d)            
        self.df = pd.concat(dum)

        
#**************************************************************************************#     
     # determine case I,II, or III and get parameters for discriminant
    def determine_case_and_params(self, print_params=False):    
        '''
        Uses Bayes Decision Theory

        looks at cov matrices, assigns case I, II, or III, then finds g_i(x) for each class w_i  

        inputs: mu: nested list of mean vectors with length 1xn (one for each w_i) 
                
                covariances: nested list of cov matrices with length nxn (one for each w_i)
                
                priors: 1D list of prior probilities for each w_i 

        '''
        covariances = self.cov
        n= len(self.mu[0]) # number of features
        mu = [self.mu[i].reshape(n,1) for i in range(len(self.mu))] #reshape the given mu inputs into vectors
        
        w_i = []
        w_i0=[]
        W_i = []
        
        if all(np.array_equal(covariances[i],covariances[i+1]) for i in range(len(covariances)-1)) == True:
            if all([np.array_equal(covariances[i]/np.array(covariances[i])[0][0],np.eye(len(covariances[i]))) for i in range(len(covariances))]) == True:
                # case 1: same cov matrix for all w_i, only 1 sigma^2 
                w_i.append([(mu[i]*1)/np.array(covariances[i])[0][0] for i in range(len(covariances))])
                w_i0.append([-1/(2*np.array(covariances[i])[0][0])*(np.transpose(mu[i])@mu[i]) + np.log(self.priors[i]) for i in range(len(covariances))])
                self.case_num = 1
                self.classifier_params = [w_i, w_i0]

            else: 
                # case 2: same cov matrices, != sigma^2 I 
                w_i.append([np.linalg.inv(covariances[i])@mu[i] for i in range(len(covariances))])
                w_i0.append([-0.5*np.transpose(mu[i])@np.linalg.inv(covariances[i])@mu[i] + np.log(self.priors[i]) for i in range(len(covariances))])
                self.case_num = 2
                self.classifier_params = [w_i, w_i0]
        else:
            # case 3: different cov matrices
            W_i.append([-0.5*np.linalg.inv(covariances[i]) for i in range(len(covariances))])
            w_i.append([np.linalg.inv(covariances[i])@mu[i] for i in range(len(covariances))])
            w_i0.append([-0.5*np.transpose(mu[i])@np.linalg.inv(covariances[i])@mu[i] - 0.5*np.log(np.linalg.det(covariances[i])) + np.log(self.priors[i]) for i in range(len(covariances))])
            self.case_num = 3
            self.classifier_params = [W_i, w_i, w_i0]

        if print_params==True:
            print('\n ****************************************************************** \n')
            print('\n Case number:', self.case_num)
            print('\b Classifier Params: \n', self.classifier_params)
            print('\n ****************************************************************** \n')
    
    # get discriminant equations       
    def get_g_equations(self, print_params=False): # takes the parameters and calculates the discriminant for case 1,2 or 3
        n_eqs = len(self.classifier_params[0][0])
        x = np.transpose([[sym.symbols('x%d' % i) for i in range(1,n_eqs+1)]])
        g = []
        if self.case_num==1:
            g.append([np.transpose(self.classifier_params[0][0][i])@x + self.classifier_params[1][0][i] for i in range(n_eqs)])
        if self.case_num==2:
            g.append([np.transpose(self.classifier_params[0][0][i])@x + self.classifier_params[1][0][i] for i in range(n_eqs)])
        if self.case_num==3:
            g.append([np.transpose(x)@np.transpose(self.classifier_params[0][0][i])@x + np.transpose(self.classifier_params[1][0][i])@x + self.classifier_params[2][0][i] for i in range(n_eqs)])

        self.g_equations = g
        
        if print_params == True:
            print('\n ****************************************************************** \n')
            print('\n Discriminant equations:')
            print('\n g1, g2 = \n', self.g_equations)
            print('\n ****************************************************************** \n')

    #converts the symbolic discriminant equations into usable lambda expressions
    def lambdify_equations(self): 
        # get function g = g1 - g2
        g_x_symbolic = sym.simplify(self.g_equations[0][0][0][0] - self.g_equations[0][1][0][0])
        self.g_x_lambda_2d = sym.lambdify([x1,x2], g_x_symbolic, "numpy")
        
        # calculate values of g_1 - g2 
        self.df['decision_boundary_2d'] = self.g_x_lambda_2d(self.df.x1,self.df.x2) 
        
        #set g1=g2 and solve for x2
        self.x2_symbolic = sym.solveset(self.g_equations[0][0][0][0] - self.g_equations[0][1][0][0] , x2)
        
        #check if there are multiple solutions for x2, if true make a function for each solution
        if len(self.x2_symbolic.args)> 1:
            f=[]
            for i in range(len(self.x2_symbolic.args)):
                f.append(sym.lambdify(x1, self.x2_symbolic.args[i], "numpy")) 
        else: #one function for x2
            f = sym.lambdify(x1, self.x2_symbolic.args[0], "numpy") 
        
        self.boundary_function = f
        #get the domain for the case with multiple x2 equations - not really needed but maybe useful 
        domain = continuous_domain(self.x2_symbolic, x1, sym.S.Reals)
        #domain = np.array([domain.args[i] for i in range(len(domain.args))])                 
        self.boundary_domain = domain
        
    # classifies samples into class 1 or 2 depending on decision boundary    
    def classify_2d_sample(self, print_params=False): 
        self.lambdify_equations() 
        # choose w_1 if g_i>0 or choose w_2 if g_i < 0
        self.df.loc[self.df.decision_boundary_2d>0, 'estimated_class'] =1
        self.df.loc[self.df.decision_boundary_2d<0, 'estimated_class'] =2   
        
        if print_params==True:
            print('\n ****************************************************************** \n')
            print('\n x2= \n', self.x2_symbolic)
            print('\n domain:', self.boundary_domain)
            print('\n ****************************************************************** \n')
#**************************************************************************************#  
    # Finds the actual amount of misclassified values by comparing true class with the estimated class

    def get_true_error(self, print_params=False):
        self.n_samples_misclassified = []
        for i in range(len(self.samples)):
            self.n_samples_misclassified.append(len(self.df[(self.df.true_class == i+1) & (self.df.estimated_class != i+1)]))
        self.misclassification_probability = np.array(self.n_samples_misclassified)/np.array(self.n_samples)
        self.total_error_probability = np.sum(self.n_samples_misclassified)/np.sum(self.n_samples)
        
        if print_params==True:
            print('\n ****************************************************************** \n')
            print('\n Empirical Error Stats: \n')
            print('N samples misclassified:', self.n_samples_misclassified)
            print('Misclassification rate by class:', self.misclassification_probability)
            print('Total misclassification rate:',self.total_error_probability)
            print('\n ****************************************************************** \n')

#**************************************************************************************#     

    def Bhattacharyya_error_bound(self, beta=0.5, print_params=False):
        self.beta = beta
        self.find_k()
        P = self.priors[0]**self.beta * self.priors[1]**(1-self.beta) * np.exp(-self.Bhattacharyya_k_value) 
        self.Bhattacharyya_error_probability = P
        if print_params==True:
            print('\n ****************************************************************** \n')
            print('\n Bhattacharyya Error Bound Stats: \n')
            print('k value:', self.Bhattacharyya_k_value)
            print('Probability of error <=', self.Bhattacharyya_error_probability)
            print('\n ****************************************************************** \n')
            
    def find_k(self): #for Bhattacharyya error
        num = np.linalg.det((1-self.beta)*self.cov[0]+ self.beta*self.cov[1])
        denom = np.linalg.det(self.cov[0])**(1-self.beta) * np.linalg.det(self.cov[1])**self.beta
        k = 0.5*self.beta*(1-self.beta) * np.transpose(self.mu[0]-self.mu[1]) @ np.linalg.inv((1-self.beta)*self.cov[0] + self.beta*self.cov[1]) @ (self.mu[0]-self.mu[1])+ 0.5*np.log(num/denom)
        self.Bhattacharyya_k_value = k
        
#**************************************************************************************#          
    # Euclidean distance classifer and error calculations
    
    def classify_by_euclidean_distance(self):
        euc_funcs = []
        n_eqs = len(self.mu)
        mu = [self.mu[i].reshape(n_eqs,1) for i in range(len(self.mu))]
        x = np.transpose([[sym.symbols('x%d' % i) for i in range(1,n_eqs+1)]])
        euc_funcs.append([- (x- mu[i]).T @ (x- mu[i])  for i in range(n_eqs)])
        self.euclidean_eqs_symbolic = euc_funcs
        self.euclidean_g = sym.simplify(euc_funcs[0][0][0][0] - euc_funcs[0][1][0][0])
        self.euclidean_function = sym.lambdify([x1,x2], self.euclidean_g, "numpy")
        self.euclidean_x2_symbolic = sym.solveset(euc_funcs[0][0][0][0] - euc_funcs[0][1][0][0] , x2)
        try:
            self.euclidean_x2_function = sym.lambdify(x1, self.euclidean_x2_symbolic.args[0], "numpy") 
        except IndexError:
            print('Warning: Unable to get function in terms of x2. Euclidean plot utility will be affected')
        self.df['euclidean_decision_boundary_2d'] = self.euclidean_function(self.df.x1, self.df.x2)
        self.df.loc[self.df.euclidean_decision_boundary_2d >0, 'estimated_class_euclidean'] =1
        self.df.loc[self.df.euclidean_decision_boundary_2d <0, 'estimated_class_euclidean'] =2  
        
    def get_euclidean_error(self, print_params=False):
        self.n_samples_misclassified_euclidean = []
        for i in range(len(self.samples)):
            self.n_samples_misclassified_euclidean.append(len(self.df[(self.df.true_class == i+1) & (self.df.estimated_class_euclidean != i+1)]))
        self.misclassification_probability_euclidean = np.array(self.n_samples_misclassified_euclidean)/np.array(self.n_samples)
        self.total_error_probability_euclidean = np.sum(self.n_samples_misclassified_euclidean)/np.sum(self.n_samples)
        
        if print_params==True:
            print('\n ****************************************************************** \n')
            print('\n Discriminant equations: \n')
            print('g1, g2 = ', self.euclidean_eqs_symbolic)
            print('\n x2= \n', self.euclidean_x2_symbolic)
            print('\n ****************************************************************** \n')
            print('\n Euclidean Distance Error Stats: \n')
            print('N samples misclassified:', self.n_samples_misclassified_euclidean)
            print('Probability of misclassification by class:', self.misclassification_probability_euclidean)
            print('Total error probability:',self.total_error_probability_euclidean)
            print('\n ****************************************************************** \n')
            
            
#**************************************************************************************#          
    # Utilities to run code 
    
    def run_all_analysis(self): 
        self.generate_samples()
        self.convert_to_pandas_df()
        self.determine_case_and_params()
        self.get_g_equations()
        self.classify_2d_sample()
        self.get_true_error()
        self.Bhattacharyya_error_bound()
        self.classify_by_euclidean_distance()
        self.get_euclidean_error()

    def summary_stats(self):
        print('\n ****************************************************************** \n')
        print('Params of Randomly Generated Samples: \n')
        print('\n Sample Means: \n') 
        [print('Sample',i+1,':', self.sample_means[i]) for i in range(len(self.sample_means))]
        print('\n Sample Covariances: \n')
        [print('Sample',i+1,':', self.sample_covs[i]) for i in range(len(self.sample_covs))]
        print('\n ****************************************************************** \n')
        print('\n Case type:', self.case_num)
        print('\n Discriminant equations:')
        print('\n g1, g2 = \n', self.g_equations)
        print('\n x2= \n', self.x2_symbolic)
        print('\n ****************************************************************** \n')
        print('\n Empirical Error Stats: \n')
        print('N samples misclassified:', self.n_samples_misclassified)
        print('Misclassification rate by class:', self.misclassification_probability)
        print('Total misclassification rate:',self.total_error_probability)
        print('\n ****************************************************************** \n')
        print('\n Bhattacharyya Error Bound Stats: \n')
        print('k value:', self.Bhattacharyya_k_value)
        print('Probability of error <=', self.Bhattacharyya_error_probability)
        print('\n ****************************************************************** \n')
        print('\n Euclidean Distance Error Stats: \n')
        print('N samples misclassified:', self.n_samples_misclassified_euclidean)
        print('Probability of misclassification by class:', self.misclassification_probability_euclidean)
        print('Total error probability:',self.total_error_probability_euclidean)
        print('\n ****************************************************************** \n')
        
        
    def make_plot(self, plot_type='matplotlib', title='Randomly Generated Data', sort_type='raw', mean_lines=False, show_boundary=False):
        
        
        ## seaborn plots
        if plot_type == 'seaborn': # seaborn plot
            if sort_type == 'raw':
                self.df['True Class'] = self.df['true_class'].map({1:'Class 1', 2:'Class 2'})
                p = sns.jointplot(x = 'x1', y = 'x2', hue='True Class', data = self.df, palette=['red', 'blue'], s=0.5)
            if sort_type == 'sorted':
                self.df['Estimated Class'] = self.df['estimated_class'].map({1:'Class 1', 2:'Class 2'})
                self.df = self.df.sort_values(by='estimated_class')
                p = sns.jointplot(x = 'x1', y = 'x2', hue='Estimated Class', data = self.df, palette=['red', 'blue'], s=0.5)
            if sort_type == 'euclidean':
                show_boundary=False
                self.df['Estimated Class (Euclidean)'] = self.df['estimated_class_euclidean'].map({1:'Class 1', 2:'Class 2'})
                self.df = self.df.sort_values(by='estimated_class_euclidean')
                p = sns.jointplot(x = 'x1', y = 'x2', hue='Estimated Class (Euclidean)', data = self.df, palette=['red', 'blue'], s=0.5)
                p.ax_joint.plot(self.df.x1, self.euclidean_x2_function(self.df.x1), lw=1, color='k', label='Decision Boundary')

            if mean_lines==True:
                # only one line may be specified; full height
                p.ax_joint.axvline(self.mu[0][0], ls='--', color='red')
                p.ax_joint.axhline(self.mu[0][1], ls='--', color='red')
                
                # only one line may be specified; full height
                p.ax_joint.axvline(self.mu[1][0],ls='--', color='blue')
                p.ax_joint.axhline(self.mu[1][1], ls='--', color='blue')

            if show_boundary==True:
                try:
                    if len(self.boundary_function)>1:
                        self.df = self.df.sort_values(by='x1')
                        labels=['Decision Boundary', '']
                        [p.ax_joint.plot(self.df.x1.sort_values(ascending=True), self.boundary_function[i](self.df.x1.sort_values(ascending=True)), lw=1, color='k', label=labels[i]) for i in range(len(self.boundary_function))]
                except TypeError:
                       p.ax_joint.plot(self.df.x1.sort_values(), self.boundary_function(self.df.x1.sort_values()), lw=1, color='k', label='Decision Boundary')
            
            for axes in p.ax_joint.get_shared_y_axes():
                for ax in axes:
                    p.ax_joint.get_shared_y_axes().remove(ax)
            p.ax_joint.set_aspect('equal', 'datalim') 
            p.ax_joint.legend(loc='upper right')
            p.fig.suptitle(title)
            p.fig.tight_layout()
            p.fig.subplots_adjust(top=0.95) 
        
        
        ## matplotlib plots
        elif plot_type=='matplotlib': 

            fig, ax = plt.subplots()

            if sort_type=='raw':
                ax.plot(self.samples[0][:, 0], self.samples[0][:, 1], '.',markersize=1, alpha=0.5, label='Class 1')
                ax.plot(self.samples[1][:, 0], self.samples[1][:, 1], '.',markersize=1, alpha=0.5, label='Class 2')

            if sort_type=='sorted':
                ax.plot(self.df.x1[(self.df.estimated_class==1)],self.df.x2[(self.df.estimated_class==1)], '.',markersize=1, alpha=0.5, label='Estimated Class 1')
                ax.plot(self.df.x1[(self.df.estimated_class==2)],self.df.x2[(self.df.estimated_class==2)],'.',markersize=1, alpha=0.5, label='Estimated Class 2')

            if sort_type=='euclidean':
                ax.plot(self.df.x1[(self.df.estimated_class_euclidean==1)],self.df.x2[(self.df.estimated_class_euclidean==1)], '.',markersize=1, alpha=0.5, label='Estimated Class 1')
                ax.plot(self.df.x1[(self.df.estimated_class_euclidean==2)],self.df.x2[(self.df.estimated_class_euclidean==2)],'.',markersize=1, alpha=0.5, label='Estimated Class 2')
                try:
                    ax.plot(self.df.x1, self.euclidean_x2_function(self.df.x1), lw=1, color='k', label='Euclidean Decision Boundary')
                except ValueError:
                    m = np.full(np.sum(self.n_samples), self.euclidean_x2_function(self.df.x1))
                    ax.plot(self.df.x1, m, lw=1, color='k', label='Euclidean Decision Boundary')
                except AttributeError:
                    print('euclidean_x2_function not defined. Unable to plot boundary line')

            if mean_lines==True:
                # only one line may be specified; full height
                ax.axvline(self.mu[0][0], ls='--', color='b')
                ax.axhline(self.mu[0][1], ls='--', color='b')

                # only one line may be specified; full height
                ax.axvline(self.mu[1][0],ls='--', color='orange')
                ax.axhline(self.mu[1][1], ls='--', color='orange')

            if show_boundary==True:
                try:
                    if len(self.boundary_function)>1:
                        self.df = self.df.sort_values(by='x1')
                        labels=['Decision Boundary', '']
                        [ax.plot(self.df.x1, self.boundary_function[i](self.df.x1), lw=1, color='k', label=labels[i]) for i in range(len(self.boundary_function))]
                except TypeError:
                       ax.plot(self.df.x1.sort_values(), self.boundary_function(self.df.x1.sort_values()), lw=1, color='k', label='Decision Boundary')

            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.set_ylabel('x2')
            ax.set_xlabel('x1')
            ax.set_title(title)    
            ax.legend(loc='upper right',markerscale=10)



