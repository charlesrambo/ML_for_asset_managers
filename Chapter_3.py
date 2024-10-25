# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 08:44:47 2024

@author: charlesr
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

path = ???
plt.style.use("seaborn-v0_8")

#------------------------------------------------------------------------------
def get_num_bins(N, corr = None):
    # Optimal number of bins for discretization
    
    # Univariate case
    if corr is None:
        
        z = np.cbrt(8 + 324 * N + 12 * np.sqrt(36 * N + 729 * N**2))
        
        b = np.round(z/6 + 2/(3 * z) + 1/3)
        
    # Bivariate case
    else:
        
        b = np.round(np.sqrt(1/2) * np.sqrt(1 + np.sqrt(1 + 24 * N/(1 - corr**2))))
        
    return int(b)


#------------------------------------------------------------------------------
def shannon_entropy(x, y = None, bins = None):
    # Function to get Shannon entropy
    
    # Univariate case...
    if y is None:
         
        # If the number of bins is none...
        if bins is None:
            
            # ... use the get_num_bins function to get the optimal number of bins
            bins = get_num_bins(len(x), corr = None)
         
        # Use numpy to calculate histogram
        hist = np.histogram(x, bins)[0]
    
    # Multivariate case...        
    else:
        
        # If the lengths of x and y aren't the same...
        if len(x) != len(y):
            
            # ... we have a problem so throw an error
           raise Exception('The arrays x and y must be the same length!')
         
        # If the number of bins is none...           
        if bins is None:
            
            # ... calculate the correlation between x and y
            corr = np.corrcoef(x, y)[0, 1]
            
            # ... then use the result to obtain the optimal number of bins
            bins = get_num_bins(len(x), corr = corr)
        
        # Use numpy to calculate histogram 
        hist = np.histogram2d(x, y, bins)[0]
        
    # Use histogram to calculate entropy 
    return stats.entropy(hist)


#------------------------------------------------------------------------------
def mutual_info(x, y, bins = None, normalize = False):
    
    # If the lengths of x and y aren't the same...
    if len(x) != len(y):
        
        # ... we have a problem so throw an error
       raise Exception('The arrays x and y must be the same length!')
           
    # If the number of bins is none...
    if bins is None:
        
        # ... use the get_num_bins function to get the optimal number of bins
        bins = get_num_bins(len(x), corr = np.corrcoef(x, y)[0, 1])
     
    # Calculate 'contingency' between x and y
    C_XY = np.histogram2d(x, y, bins)[0]
    
    # Use contingency to calculate mutual information
    I_XY = mutual_info_score(None, None, contingency = C_XY)
    
    # If normalize is True...
    if normalize:
        
        #  ... calculate the univariate entropies
        H_X = shannon_entropy(x, bins = bins)
        H_Y = shannon_entropy(y, bins = bins)
        
        # ... then divide the mutual information by the smaller of the two
        I_XY /= np.min([H_X, H_Y])
        
    return I_XY


#------------------------------------------------------------------------------
def varation_of_info(x, y, bins = None, normalize = False):
    
    # If the number of bins is none...
    if bins is None:
        
        # ... use the get_num_bins function to get the optimal number of bins
        bins = get_num_bins(len(x), corr = np.corrcoef(x, y)[0, 1])
    
    # Calculate the mutual information
    I_XY = mutual_info(x, y, bins = bins)
    
    # Calculate univariate entropies
    H_X = shannon_entropy(x, bins = bins)
    H_Y = shannon_entropy(y, bins = bins)
    
    # Use univariate results to calculation varation of information
    V_XY = H_X + H_Y - 2 * I_XY
    
    # If normalize is True...
    if normalize:
        
        # ... calculate multivariate entropy
        H_XY = H_X + H_Y - I_XY
        
        # ... divide varation of information by result
        V_XY /= H_XY
        
    return V_XY


if __name__ == '__main__':
    
    # Define variables
    size, seed = 10000, 0
    noise = 1
    
    # Set random seed
    np.random.seed(seed)
    
    # Create x-values for plot
    x = np.linspace(-3, 3, num = size)
    
    # The first four plots will be x raised to various degrees
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8))
    
    for degree in range(4):
        
        row, col = degree//2, degree % 2
        
        y = x**degree + noise * stats.norm.rvs(size = size)
        
        rho = np.corrcoef(x, y)[0, 1]
        info = mutual_info(x, y, normalize = True)
        
        ax[row, col].plot(x, y)

        ax[row, col].set_title(r'$d = $' + f'{degree}' + '\n'
                               + f'Correlation: {rho: .3f}' + '\n' + f'Mutual Information: {info: .3f}')
     
    plt.suptitle('$y = x^d + \epsilon$', fontsize = 15)
    plt.tight_layout()
    plt.savefig(path + 'fig3.1.png')
    plt.show()
    
    # The second four plots will be x multiplied by various coefficients
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8))
    
    for i, coef in enumerate(np.linspace(0, 10, 4)):
        
        row, col = i//2, i % 2
        
        y = coef * x + noise * stats.norm.rvs(size = size)
        
        rho = np.corrcoef(x, y)[0, 1]
        info = mutual_info(x, y, normalize = True)
        
        ax[row, col].plot(x, y)

        ax[row, col].set_title('$c = $' + f'{coef:.2f}' + '\n' 
                               + f'Correlation: {rho: .3f}' + '\n' + f'Mutual Information: {info: .3f}')
    
    plt.suptitle('$y = c x + \epsilon$', fontsize = 15)
    plt.tight_layout()
    plt.savefig(path + 'fig3.2.png')
    
    plt.show()
    
    # The thir four plots will be |x| multiplied by various coefficients    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8))
    
    for i, coef in enumerate(np.linspace(0, 10, 4)):
        
        row, col = i//2, i % 2
        
        y = coef * np.abs(x) + noise * stats.norm.rvs(size = size)
        
        rho = np.corrcoef(x, y)[0, 1]
        info = mutual_info(x, y, normalize = True)
        
        ax[row, col].plot(x, y)
        ax[row, col].set_title('$c = $' + f'{coef:.2f}' + '\n' 
                               + f'Correlation: {rho: .3f}' + '\n' + f'Mutual Information: {info: .3f}')
        
    plt.suptitle('$y = c |x| + \epsilon$', fontsize = 15)
    plt.tight_layout()
    plt.savefig(path + 'fig3.3.png')
    plt.show()       