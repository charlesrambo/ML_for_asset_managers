# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 07:25:40 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import matplotlib.pyplot as plt

path = r'G:/USERS/CharlesR/Python/ML_for_asset_managers/'
plt.style.use("seaborn-v0_8")

#------------------------------------------------------------------------------
class marchenkoPastur:
    
    def __init__(self, sigma2, q):
        
        self.sigma2 = sigma2
        self.q = q
        
        lam_minus = sigma2 * (1 - np.sqrt(1/q))**2
        lam_plus = sigma2 * (1 + np.sqrt(1/q))**2 
        
        self.lam_minus = lam_minus
        self.lam_plus = lam_plus
        
        # Vectorize pd
        self.pdf = np.vectorize(self.pdf)
        
    def pdf(self, x):
        
        # Support of Marchenko–Pastur distribution
        if self.lam_minus <= x <= self.lam_plus:
            
            coef = self.q/(2 * np.pi * self.sigma2)
        
            return coef * np.sqrt((self.lam_plus - x) * (x - self.lam_minus))/x
                    
        else:
            
            return 0  
        
    def support(self):
            
        return self.lam_plus, self.lam_minus
        
    
#------------------------------------------------------------------------------
def marchenko_pastur_pdf(var, q, num_points):
    
    #Marcenk-Pastur pdf
    # q = T/N
    
    # Initialize instance of Marchenk-Pastur
    mp = marchenkoPastur(var, q)
    
    # Generate x-values for pdf
    eigvals = np.linspace(mp.lam_minus, mp.lam_plus, num_points)
    
    # Calculate y-values
    pdf = pd.Series(mp.pdf(eigvals), index = eigvals)
    
    return pdf

#------------------------------------------------------------------------------
def get_PCA(matrix):
    
    # Get eigenvalues and eigenvectors from Hermitian matrix
    eigvals, eigvecs = np.linalg.eigh(matrix)
    
    # Arguments for sorting eigenvalue decending order
    indices = eigvals.argsort()[::-1]
    
    # Rearange entries
    eigvals, eigvecs = eigvals[indices], eigvecs[:, indices]
    
    # Convert eigenvalues into matrix
    eigvals = np.diagflat(eigvals)
    
    return eigvals, eigvecs

#------------------------------------------------------------------------------
def KDE_pdf(samples, bandwidth = 0.25, kernel = 'gaussian', x = None):
    """
    Fit kernel to a series of samples, and derive the prob of samples.
    The array x is the array of values on which the fit KDE will be evaluated

    Parameters
    ----------
    samples : np.array
        Array of samples
    bandwidth : float or string
        The bandwidth of the kernel. If bandwidth is a float, it defines the 
        bandwidth of the kernel. If bandwidth is a string, one of the estimation 
        methods is implemented.. The default is 0.25.
    kernel : ‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’
        The kernel to use. The default is 'gaussian'.
    x : np.array, optional
        Values to evaluate the pdf at. The default is None. If the default is 
        used, then the pdf will be evaluate at the unique observations of the
        array samples.

    Returns
    -------
    pdf : pd.Series
        Pandas series of empirical pdf values. The index is x.

    """
    
    # Convert samples into a column vector
    if len(samples.shape) == 1: samples = samples.reshape((-1, 1))
    
    # Fit the kernel desnity estimate of the pdf
    kde = KernelDensity(kernel = kernel, bandwidth = bandwidth).fit(samples)
    
    # If no x-values were provided, use the unique observations of samples
    if x is None: x = np.unique(samples).reshape((-1, 1))
    
    # Make sure that x is a column vector
    if len(x.shape) == 1: x = x.reshape((-1, 1))
    
    # Compute the log-likelihood of each sample under the model.
    logProb = kde.score_samples(x)
    
    # Values the pdf values
    pdf = pd.Series(np.exp(logProb), index = x.flatten())
    
    return pdf

#------------------------------------------------------------------------------
def get_random_cov(ncols, nfactors):
    
    # Rendomly generate  normal numparray
    w = np.random.normal(size = (ncols, nfactors))
    
    # Random cov matrix, which is not full rank
    cov = w @ w.T
    
    # Make full rank by adding uniform random to diagonal
    cov += np.diag(np.random.uniform(size = ncols))
    
    return cov

#------------------------------------------------------------------------------
def cov2corr(cov, return_std = False):
    """
    Convert covariance matrix to correlation matrix.
    Copied from https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/moment_helpers.py

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that
    division is defined elementwise. np.ma.array and np.matrix are allowed.
    """
    cov_ = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov_))
    corr = cov_ / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr

#------------------------------------------------------------------------------
def corr2cov(corr, std):
    """
    Convert correlation matrix to covariance matrix given standard deviation.
    Copied from https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/moment_helpers.py
    
    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.
    """
    corr_ = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr_ * np.outer(std_, std_)
    
    return cov

#------------------------------------------------------------------------------
def get_pdf_error(var, eigvals, q, bandwidth, num_points = 1000):
     # Fit error
     
     # Theoretical pdf
    pdf0 = marchenko_pastur_pdf(var[0], q, num_points)
    
    # Empirical pdf
    pdf1 = KDE_pdf(eigvals, bandwidth, x = pdf0.index.values)

    # Calculate sum of square errors
    sse = np.sum((pdf1 - pdf0)**2)

    return sse    

#------------------------------------------------------------------------------
def find_marchenko_pastur_params(eigvals, q, bandwidth):
    
    # Find max random eigenvalue by fitting Marcenko-Pastur distribution
    out = minimize(lambda *x: get_pdf_error(*x), x0 = 0.5, 
                   args = (eigvals, q, bandwidth),
                   bounds = ((1e-5, 1 - 1e-5),))
    
    # Get the result in cases where we have convergence
    sigma2 = out['x'][0] if out['success'] else 1.0
   
    # Calculate the support per Marcenk-Pastur theory
    lam_plus, lam_minus = marchenkoPastur(sigma2, q).support()
    
    return lam_plus, lam_minus, sigma2

#------------------------------------------------------------------------------
def get_clipped_corr(eigvals, eigvecs, nfactors):
    # Remove noise from corr by fixing random eigenvalues
    
    # Convert to 1D array
    evals_ = np.diag(eigvals).copy()
    
    # Replace noise eigenvalues with average
    evals_[nfactors:] = evals_[nfactors:].sum()/(evals_.shape[0] - nfactors)
    
    # Convert to matrix
    evals_= np.diag(evals_)
    
    # Calculate correlation matrix
    corr1 = eigvecs @ evals_ @ eigvecs.T
    
    # Make sure observations on main diagonal are 1
    corr1 = cov2corr(corr1)
    
    return corr1

#------------------------------------------------------------------------------
def get_shrunk_corr(eigvals, eigvecs, nfactors, alpha = 0):   
    # Remove noise from corr through targeted shrinkage
    
    # Get the signal part
    evals_signal = eigvals[:nfactors, :nfactors] 
    evecs_signal = eigvecs[:, :nfactors]
    
    # Calculate signal correlation matrix
    corr_signal = evecs_signal @ evals_signal @ evecs_signal.T
    
    # Get the noise part
    evals_noise = eigvals[nfactors:, nfactors:]
    evecs_noise = eigvecs[:, nfactors:]
    
    # Calculate noise correlation matrix
    corr_noise = evecs_noise @ evals_noise @ evecs_noise.T
    
    # Shrink the noise part of the correlation matrix but not the signal part
    corr = corr_signal + alpha * corr_noise + (1 - alpha) * np.diag(np.diag(corr_noise))
    
    return corr

#------------------------------------------------------------------------------
def detoned_corr(eigvals, eigvecs, indicies):
    
    # Calculate correlation matrix
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Start with correlation matrix
    C_detoned = C 
    
    # Remove principal components specified in indicies
    C_detoned -= np.outer(eigvecs[:, indicies] @ eigvals[indicies], 
                          eigvecs[:, indicies].T)
    
    # Make sure main diagonal only has ones
    C_detoned = cov2corr(C_detoned)
    
    return C_detoned

#------------------------------------------------------------------------------
def form_block_diag_matrix(num_blocks, block_size, block_corr):
    
    # Create clock with given correlation
    block = np.ones((block_size, block_size)) * block_corr
    
    # Make sure the diagonal contains ones
    np.fill_diagonal(block, 1.0)

    # Create larger correlation matrix
    corr = block_diag(*([block] * num_blocks))
    
    return corr

#------------------------------------------------------------------------------
def form_true_matrix(num_blocks, block_size, block_corr):
    
    # Get block diagonal matrix
    block_diag_mat = form_block_diag_matrix(num_blocks, block_size, block_corr)
    
    # It the columns
    indices = np.arange(block_diag_mat.shape[0])
    
    # Shuffle the columns
    np.random.shuffle(indices)
    
    # Rearrange the shows and columns
    shuffled_corr = block_diag_mat[indices, :][:, indices]
    
    # Randomly generate standard deviations
    std = np.random.uniform(0.05, 0.20, shuffled_corr.shape[0])
    
    # Convert correlation matrix to covariance matrix
    cov = corr2cov(shuffled_corr, std)
    
    # Randomly generate expected returns
    mu = np.random.normal(std, std, cov.shape[0]).reshape((-1, 1))
    
    return mu, cov

#------------------------------------------------------------------------------
def sim_cov_mu(mu, cov, nobs, shrink = False):
    
    # Generate the data
    X = np.random.multivariate_normal(mu.flatten(), cov, size = nobs)
    
    # Take the sample mean
    mu1 = X.mean(axis = 0).reshape((-1, 1))
    
    # If shrink is true...
    if shrink:
        
        # ... calculate the shrunken covariance matrix
        cov1 = LedoitWolf().fit(X).covariance_
    
    # Otherwise...
    else:
        
        # ... calculate the sample covariance matrix
        cov1 = np.cov(X, rowvar = False)
        
    return mu1, cov1

#------------------------------------------------------------------------------
def denoise_cov(cov, q, bandwidth):
    
    # Get the corresponding correlation matrix
    corr, std = cov2corr(cov, return_std = True)
    
    # Get the eigenvalues and eigenvectors
    eigvals, eigvecs = get_PCA(corr)
    
    # Get support for best fit Marchenko-Pastur distribution
    lam_plus, lam_minus, sigma2 = find_marchenko_pastur_params(np.diag(eigvals), 
                                                               q, bandwidth)
    
    # Get the number of signal factors
    nfactors = eigvals.shape[0] - np.diag(eigvals)[::-1].searchsorted(lam_plus)
    
    # Use results to clip the correlation matrix
    corr_clip = get_clipped_corr(eigvals, eigvecs, nfactors)
    
    # Use results to get new covariance matrix
    cov_clip = corr2cov(corr_clip, std)
    
    return cov_clip

#------------------------------------------------------------------------------
def optimize_portfolio(cov, mu = None):
    
    # Calculate the inverse covariance matrix
    cov_inv = np.linalg.inv(cov)
    
    # If mu is None...
    if mu is None: 
        
        # ... make it a vector of ones
        mu = np.ones(shape = (cov_inv.shape[0], 1))
     
    # Calculate the weight
    wt = cov_inv @ mu
    
    # Make sure it sums to 1
    wt /= np.sum(wt)
    
    return wt


if __name__ == '__main__':
    
    # Set random seed
    np.random.seed(0)
    
    #--------------------------------------------------------------------------
    # Generate samples
    X = np.random.normal(size = (10_000, 1000))
    
    # Calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = get_PCA(np.corrcoef(X, rowvar = False))

    # Get the theoretical pdf
    pdf_theo = marchenko_pastur_pdf(1., q = X.shape[0]/X.shape[1], 
                                num_points = 1000)

    # Get the empirical pdf
    pdf_emp =  KDE_pdf(np.diag(eigvals), bandwidth = 0.01)

    # Plot results
    ax = pdf_theo.plot(label = 'Mercenko-Pastur')
    pdf_emp.plot(ax = ax, label = 'Empirical:KDE', linestyle = 'dashed',
              xlabel = r'$\lambda$', ylabel = r'prob[$\lambda$]')

    ax.legend()

    plt.savefig(path + 'fig2.1.png')
    
    plt.show()
    
    #-------------------------------------------------------------------------- 
    # Initialize parameters
    alpha, ncols, nfactors, q = 0.995, 1000, 100, 10
    
    # Get sample correlation matrix
    cov = np.cov(np.random.normal(size = (ncols * q, ncols)), rowvar = False)

    # Noise + signal
    cov = alpha * cov + (1 - alpha) * get_random_cov(ncols, nfactors)

    # Get corresponding correlation matrix
    corr = cov2corr(cov)

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = get_PCA(corr)
    
    # Get the support of the best fit Marchenko-Pastur distribution
    lam_plus_hat, _, sigma2 = find_marchenko_pastur_params(np.diag(eigvals), 
                                                           q, bandwidth = 0.01)
    
    # Get the estimated number of factors
    nfactors_hat = ncols - np.diag(eigvals)[::-1].searchsorted(lam_plus_hat)

    # Get the pdf evaluated at 1000 points
    pdf_theo = marchenko_pastur_pdf(sigma2, q, num_points = 1000)

    # Plot the results
    ax = pdf_theo.plot(xlabel = r'$\lambda$', ylabel = r'prob[$\lambda$]',
                   label = 'Marcenko-Pastur Distribution')

    # Plot a histogram of the eigenvalues on the same plot
    ax.hist(np.diag(eigvals), bins = 150, density = True,
            label = 'Empirical Distribution')

    # Add a legend
    ax.legend()

    plt.savefig(path + 'fig2.2.png')
    
    plt.show()
    
    # Clip the correlation matrix
    corr_clip = get_clipped_corr(eigvals, eigvecs, nfactors_hat)
    
    # Get the eigenvalues and eigenvectors for the clipped matrix
    eigvals_clip, eigvecs_clip = get_PCA(corr_clip)

    # Get the axis which will be shared between two plots
    ax = plt.gca()

    # First plot is the original eigenfunction
    ax.plot(1 + np.arange(ncols),  np.diag(eigvals), 
            label = 'Original Eigenfunction')
    
    # The next plot is the eigenfunction of the clipped correlation matrix
    ax.plot(1 + np.arange(ncols), np.diag(eigvals_clip), 
            linestyle = 'dashed', 
            label = 'Clipped Eigenfunction')

    # Add a legend
    plt.legend()

    # Set the y-scale to log
    ax.set_yscale('log')

    # Add labels to the x- and y- axes
    ax.set_xlabel('Eigenvalue Number')
    ax.set_ylabel('Eigenvalues (Log-Scale)')

    # Save the plot
    plt.savefig(path + 'fig2.3.png')
    
    # Show the results
    plt.show()
    
    # Calculate the shrunken correlation matrix
    corr_shrunk = get_shrunk_corr(eigvals, eigvecs, nfactors_hat, 
                                  alpha = 0.5)
    
    # Get the eigenvalues and eigenvectors of the shrunken correlation matrix
    eigvals_shrunk, eigvecs_shrunk = get_PCA(corr_shrunk)

    # Get the axis which will be shared between two plots
    ax = plt.gca()

    # First plot is the original eigenfunction
    ax.plot(1 + np.arange(ncols),  np.diag(eigvals), 
            label = 'Original Eigenfunction')
    
    # Then plot the shrunken eigenfunction values
    ax.plot(1 + np.arange(ncols), np.diag(eigvals_shrunk), 
            linestyle = 'dashed', 
            label = 'Shrunken Eigenfunction')

    # Add a legend
    plt.legend()

    # Put the y-axis on a log scale
    ax.set_yscale('log')

    # Add labels to the x- and y- axes
    ax.set_xlabel('Eigenvalue Number')
    ax.set_ylabel('Eigenvalues (Log-Scale)')
    
    # Save the figure
    plt.savefig(path + 'fig2.4.png')
    
    # Show figure
    plt.show()
    
    #--------------------------------------------------------------------------
    # Define parameters for Monte-Carlo
    num_blocks, block_size, block_corr = 10, 50, 0.5
    nobs, ntrials, bandwidth = 1000, 1000, 0.01
    shrink, min_var_port = False, True
    
    # Generate true mu and covariance matrix
    mu_true, cov_true = form_true_matrix(num_blocks, block_size, block_corr)
    

    # Create a data frame to hold the weights
    wt = pd.DataFrame(columns = range(cov_true.shape[0]), 
                      index = range(ntrials), 
                      dtype = float)
    
    # Make a deep copy to hold the weights calculated using denoised metrics
    wt_denoised = wt.copy()
    
    # Set random seed
    np.random.seed(0)
    
    # Run simulations
    for i in range(ntrials):
        
        # Calculate metrics obtained from simulated data
        mu1, cov1 = sim_cov_mu(mu_true, cov_true, nobs, shrink = shrink)
        
        # If we are calculating a minimum variance portfolio set mu to None
        mu1 = None if min_var_port else mu1

        # Calculate denoised results
        cov_denoised = denoise_cov(cov1, q = nobs/cov1.shape[1], 
                                   bandwidth = bandwidth)
        
        # Save the weights using the original metrics
        wt.loc[i, :] = optimize_portfolio(cov1, mu1).flatten()
        
        # Save the weights using the denoised covariance and original mu
        wt_denoised.loc[i, :] = optimize_portfolio(cov_denoised, mu1).flatten()
     
    # Get the correct weight
    wt_true = optimize_portfolio(cov_true, None if min_var_port else mu_true).flatten()
    
    # Calculate sum of squared error
    error = wt.sub(wt_true, axis = 1)
    error_denoised = wt_denoised.sub(wt_true, axis = 1)
    
    # Create two subplot showing histograms of errors
    plt.hist(np.abs(error).mean(axis = 1), bins = int(np.sqrt(ntrials)),
             label = 'Original')
    plt.hist(np.abs(error_denoised).mean(axis = 1), 
               bins = int(np.sqrt(ntrials)), label = 'Denoised')
    
    # Add a legend
    plt.legend()
    
    # Add a title
    plt.title('Portfolio Weight Absolute Error')
    
    # Save the plot
    plt.savefig(path + 'fig2.5.png')
    
    # Show the results
    plt.show()
    
    # Calculate the root mean squared error
    rmsd = np.sqrt(np.mean(error**2))
    rmsd_denoised = np.sqrt(np.mean(error_denoised**2))
    
    if shrink:
        
        print('We used a shrunken covariance matrix in all of the calculations.\n')
        
    print(f'Sample: {rmsd:.5f}', f'Denoised: {rmsd_denoised:.5f}')
