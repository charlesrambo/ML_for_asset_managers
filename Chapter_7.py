# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:23:22 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import datetime as dt
from yfinance import download
import sys

path = r'G:/USERS/CharlesR/Python/ML_for_asset_managers/'

# Make the directory Python because that's where cal_signals is located
sys.path.insert(0, path)

# Import the calc_signals Python script
import Chapter_2 as two, Chapter_3 as three, Chapter_4 as four

# See https://stackoverflow.com/questions/69596239/how-to-avoid-memory-leak-when-dealing-with-kmeans-for-example-in-this-code-i-am
import warnings
warnings.filterwarnings('ignore')

plt.style.use("seaborn-v0_8")

#------------------------------------------------------------------------------
def get_condition_num(mat):
    
    # Get the eigenvalues; assume Hermitian
    eigvals, _ = np.linalg.eigh(mat)
    
    # Get the maximum and minimum of the absolute value of the eigenvalues
    max_val = np.max(np.abs(eigvals))
    min_val = np.min(np.abs(eigvals))
    
    return max_val/min_val


#------------------------------------------------------------------------------
# Define a function to get the data from yahoo finance
def get_data(tickers, start, end):

    # Get the return data
    data = download(tickers, 
                    start = (start - dt.timedelta(weeks = 2)).strftime("%Y-%m-%d"), 
                    end = (end + dt.timedelta(days = 1)).strftime("%Y-%m-%d")
                    )['Adj Close']
    
    # Remove rows where 95% or more of observations are missing
    data = data.loc[data.isna().mean(axis = 1) < 0.95, :]
    
    # Remove observations where 95% or more of results are missing
    data = data.loc[:, data.isna().mean(axis = 0) < 0.95]
    
    # Resample data
    data = data.resample('W').last()
    
    # Calculate return
    data = data.pct_change()
    
    # Fill missing values with mean
    data = data.fillna(data.mean(axis = 1))
    
    # Fill other na with 0
    data = data.fillna(0)
    
    # Subset to stuff no earlier than start
    data = data.loc[data.index >= str(start), :]
    
    return data


#------------------------------------------------------------------------------
def get_spx_tickers():
        
    # Get the first table on the S&P 500 company page
    spx_ticks = pd.read_html(r'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
      
    # Convert to a list
    spx_ticks = spx_ticks['Symbol'].unique().tolist()
        
    return spx_ticks


#------------------------------------------------------------------------------
def calc_mut_info_mat(df, normalize = True):
    
    # Initialize data frame to hold mutial informations
    mut_info = pd.DataFrame(index = df.columns, columns = df.columns, dtype = float)

    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns[i:], i):
                
                # Calculate mutual information
                I = 1.0 if i == j else three.mutual_info(df[col_i], df[col_j], 
                                                         normalize = normalize )
                
                # Save values
                mut_info.loc[col_i, col_j], mut_info.loc[col_j, col_i] = I, I
                
    return mut_info
    

#------------------------------------------------------------------------------
def optimal_portfolio_nco(cov, mu = None, max_clusters = None, method = 'k-means', 
                          detone = False, **kwargs):
    
    # Make covariance matrix a data frame
    cov = pd.DataFrame(cov)
    
    # Convert mu to a series
    if mu is not None: mu = pd.Series(mu[:, 0])
    
    # Calculate correlation matrix and make it a data frame
    corr = pd.DataFrame(two.cov2corr(cov), columns = cov.columns, index = cov.index)
    
    if detone:
        
        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(corr)
        
        # Detone the correlation matrix
        corr = two.detoned_corr(eigvals, eigvecs, [np.argmax(eigvals)])

    # Perform clustering
    if method == 'k-means':
    
        corr, clusters_dict, _ = four.cluster_kmeans_base(corr, max_clusters = max_clusters, **kwargs)
        
    else:
        
        corr, clusters_dict, _ = four.cluster_agglomerative_base(corr, max_clusters = max_clusters, **kwargs)       

    # Initialize data frame of weights
    wt_intra = pd.DataFrame(0.0, index = cov.index, 
                            columns = clusters_dict.keys())

    # Loop over dictionary
    for cluster in clusters_dict:
        
        # Subset covariance matrix to just stuff in cluster
        cov_intra = cov.loc[clusters_dict[cluster], clusters_dict[cluster]].values
        
        if mu is None:
            
            mu_intra = None 
            
        else:
            
            # If we're using expected; subset to just stuff in the cluster
            mu_intra = mu.loc[clusters_dict[cluster]].values.reshape(-1, 1)
            
        # Compute the optimum
        wt_intra.loc[clusters_dict[cluster], cluster] = two.optimize_portfolio(cov_intra, mu_intra).flatten()
        
    # Calculate covariance matrix of each optimized cluster
    cov_inter = wt_intra.T @ cov @ wt_intra
    
    # If we're using the expected returns, calculate expected returns of each optimized cluster
    mu_inter = None if mu is None else wt_intra.T @ mu
    
    # Perform optimization again
    wt_inter = pd.Series(two.optimize_portfolio(cov_inter, mu_inter).flatten(), 
                       index = cov_inter.index)
    
    # Calculate final weights
    wt = wt_intra.mul(wt_inter, axis = 1).sum(axis = 1).values.reshape(-1, 1)
    
    return wt


if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    corr = two.form_block_diag_matrix(num_blocks = 2, block_size = 2,
                                      block_corr = 0.5)

    print(f'The condition number is {get_condition_num(corr):.2f}.')

    #--------------------------------------------------------------------------
    corr = block_diag(two.form_block_diag_matrix(1, 3, 0.5), 
                       two.form_block_diag_matrix(1, 1, 0.0))

    print(f'The condition number is {get_condition_num(corr):.2f}.')

    #--------------------------------------------------------------------------
    corr = two.form_block_diag_matrix(2, 2, 0.75)

    print(f'The condition number is {get_condition_num(corr):.2f}.')
    
    #--------------------------------------------------------------------------
    
    # Get S&P 500 constituent returns from Yahoo finance
    data = get_data(get_spx_tickers(), start = dt.datetime(2021, 1, 1), 
                    end = dt.datetime(2023, 12, 31))

    # Calculate the covariance matrix
    cov0 = data.cov()

    # Denoise covariance matrix
    cov1 = two.denoise_cov(cov0, q = data.shape[0]/data.shape[1], bandwidth = 0.1)
    
    # Convert back into data frame
    cov1 = pd.DataFrame(cov1, index = cov0.columns, columns = cov0.columns)
    
    # Calculate correlation matrix and convert into data frame
    corr1 = pd.DataFrame(two.cov2corr(cov1), index = cov0.columns, 
                         columns = cov0.columns)
    
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(corr1)
    
    # Detone the correlation matrix
    corr_detoned = two.detoned_corr(eigvals, eigvecs, np.argsort(eigvals)[-3:])
    
    # Convert correlation matrix to data frame
    corr_detoned = pd.DataFrame(corr_detoned, index = cov0.columns, 
                         columns = cov0.columns)
    
    # Perform clustering using k-means clustering
    corr2, cluster_kmeans_dict, silh_kmeans = four.cluster_kmeans_base(corr_detoned, 
                                    max_clusters = int(0.5 * corr1.shape[0]),
                                    min_clusters = 2)
    
    # Initialize intra cluster weights
    wt_intra = pd.DataFrame(0.0, index = cov1.index, 
                            columns = cluster_kmeans_dict.keys())

    # For each cluster...
    for cluster in cluster_kmeans_dict:
        
        # ... perform optimization
        wt_intra.loc[cluster_kmeans_dict[cluster], cluster] = two.optimize_portfolio(
                                            cov1.loc[cluster_kmeans_dict[cluster], cluster_kmeans_dict[cluster]]).flatten()
        
    # Reduce covariance matrix
    cov2 = wt_intra.T @ cov1 @ wt_intra

    # Perform optimization of optimized clusters
    wt_inter = pd.Series(two.optimize_portfolio(cov2).flatten(), index = cov2.index)
    
    # Combine the results
    wt_kmeans = wt_intra.mul(wt_inter, axis = 1).sum(axis = 1).sort_index()
    
    # Perform clustering using agglomerative clustering
    corr2, cluster_agglo_dict, silh_agglo = four.cluster_agglomerative_base(corr_detoned, 
                                    max_clusters = int(0.5 * corr1.shape[0]),
                                    min_clusters = 2,
                                    metric = 'euclidean')
    
    print(f'There are {len(cluster_kmeans_dict)} clusters.')
    print(f'There are {len(cluster_agglo_dict)} clusters.')
    
    # Initialize intra cluster weights
    wt_intra = pd.DataFrame(0.0, index = cov1.index, 
                            columns = cluster_agglo_dict.keys())

    # For each cluster...
    for cluster in cluster_agglo_dict:
        
        # ... perform optimization
        wt_intra.loc[cluster_agglo_dict[cluster], cluster] = two.optimize_portfolio(
                                            cov1.loc[cluster_agglo_dict[cluster], cluster_agglo_dict[cluster]]).flatten()
        
    # Reduce covariance matrix
    cov2 = wt_intra.T @ cov1 @ wt_intra

    # Perform optimization of optimized clusters
    wt_inter = pd.Series(two.optimize_portfolio(cov2).flatten(), index = cov2.index)
    
    # Combine the results
    wt_agglo = wt_intra.mul(wt_inter, axis = 1).sum(axis = 1).sort_index()

    # Calculate direct results
    wt_markowitz = two.optimize_portfolio(cov0).flatten()
    
    print(f"The NCO algorithm and k-means the portfolio concentration is {np.sum(wt_kmeans**2):.3f}.")
    print(f"The NCO algorithm and agglomerative clustering portfolio concentration is {np.sum(wt_agglo**2):.3f}.")
    print(f"The Markowitz's portfolio concentration is {np.sum(wt_markowitz**2):.2f}.")
       
    #--------------------------------------------------------------------------
    
    # Set the random seed
    np.random.seed(0)
    
    # Initialize values
    num_blocks, block_size, block_corr = 10, 5, 0.5
    
    # Create true values
    mu_true, cov_true = two.form_true_matrix(num_blocks, block_size, block_corr)
    
    num_obs, num_sims, shrink, min_var_port = 1000, 1000, False, True
    
    np.random.seed(0)
    
    # Initialize data frames to hold weights
    wt_markowitz = pd.DataFrame(index = range(num_sims), 
                                columns = range(num_blocks * block_size))
    
    wt_kmeans_nco = pd.DataFrame(index = range(num_sims), 
                                 columns = range(num_blocks * block_size))
    
    wt_agglo_nco = pd.DataFrame(index = range(num_sims), 
                                columns = range(num_blocks * block_size))
    
    for i in range(num_sims):
        
        # Calculate values using randomly generated data
        mu_sim, cov_sim = two.sim_cov_mu(mu_true, cov_true, num_obs, shrink = shrink)
        
        # Change mu_sim to None if we're just doing low vol portfolio
        if min_var_port: mu_sim = None
        
        # Calculate minimum variance portfolio
        wt_markowitz.loc[i, :] = two.optimize_portfolio(cov_sim, mu_sim).flatten()
        
        # Calculate NCO weights using k-means
        wt_kmeans_nco.loc[i] = optimal_portfolio_nco(cov_sim, mu_sim, 
                    max_clusters = int(0.5 * num_blocks * block_size)).flatten()

        wt_agglo_nco.loc[i] = optimal_portfolio_nco(cov_sim, mu_sim, 
                    max_clusters = int(0.5 * num_blocks * block_size), 
                    method = 'agglo',
                    metric = 'euclidean').flatten()
        
    # Calculate the true optimal
    wt_true = two.optimize_portfolio(cov_true, None if min_var_port else mu_true)
    
    # Repeat true optimal simply to make it easy to compare to experimental results
    wt_true = np.repeat(wt_true.T, num_sims, axis = 0)
    
    # Calculate the root mean squared errors
    rmsd_markowitz = np.sqrt(np.mean((wt_markowitz - wt_true).values.flatten()**2))
    rmsd_kmeans_nco = np.sqrt(np.mean((wt_kmeans_nco - wt_true).values.flatten()**2))
    rmsd_agglo_nco = np.sqrt(np.mean((wt_agglo_nco - wt_true).values.flatten()**2))
    
    print(f"The NCO algorithm using k-means clustering has an RMSE of {rmsd_kmeans_nco:.10f}.")
    print(f"The NCO algorithm using agglomerative clusteringhas an RMSE of {rmsd_agglo_nco:.6f}.")
    print(f"The Markowitz's RMSE is {rmsd_markowitz:.6f}.")
    