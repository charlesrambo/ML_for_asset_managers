# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:19:16 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

path = r'G:/USERS/CharlesR/Python/ML_for_asset_managers/'
plt.style.use("seaborn-v0_8")



#------------------------------------------------------------------------------
def cluster_kmeans_base(corr0, max_clusters = 10, n_init = 10):
    
    # Calculate distance matrix
    x = np.sqrt(0.5 * (1 - corr0.fillna(0)).clip(lower = 0.0, upper = 2.0))
    
    # Initialize pandas series to hold silhouette scores
    silh = pd.Series()
    
    # max_clusters can't be more than samples minus 1
    max_clusters = int(np.min([max_clusters, corr0.shape[0] - 1]))
    
    # Loop over possible number of clusters
    for i in range(2, max_clusters + 1):
        
        # Itialize k-means
        kmeans_ = KMeans(n_clusters = i, n_init = n_init)
        
        # Fit results
        kmeans_ = kmeans_.fit(x)
        
        if len(np.unique(kmeans_.labels_)) > 1:
            
            # Get silhouette scores
            silh_ = silhouette_samples(x, kmeans_.labels_)
            
            # Record results
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            
            # If improvement...
            if np.isnan(stat[1])|(stat[0] > stat[1]):
                
                # ... record results
                silh, kmeans = silh_, kmeans_
                
        else:
            
            continue
    
    # Get the index values
    idx_new = np.argsort(kmeans.labels_)
    
    # Rearrange correlation matrix
    corr1 = corr0.iloc[idx_new, idx_new]
    
    # Create dictionary to record the clusters
    clust_dict = {clst:corr0.columns[np.where(kmeans.labels_ == clst)[0]].tolist()
                  for clst in np.unique(kmeans.labels_)}
    
    # Change index of silhouette scores
    silh = pd.Series(silh, index = x.index)
    
    return corr1, clust_dict, silh


#------------------------------------------------------------------------------    
def make_new_outputs(corr0, clusters, clusters2):
    
    # Initialize clusters dictionary
    clusters_new = {}
    
    # Loop over the original cluster keys
    for c in clusters:
        
        # Save the results; key based on order in clusters
        clusters_new[len(clusters_new.keys())] = list(clusters[c])
    
    # Loop over the new cluster keys
    for c in clusters2:
        
        # Save results; key based on order in clusters2 + number in original clusters
        clusters_new[len(clusters_new.keys())] = list(clusters2[c])
    
    # Get the new indices 
    idx_new = [idx for clst in clusters_new for idx in clusters_new[clst]]
    
    # Reorder observations based on cluster order
    corr_new = corr0.loc[idx_new, idx_new]

    # Calculate distance matrix
    x = np.sqrt(0.5 * (1 - corr0.fillna(0)).clip(lower = 0.0, upper = 2.0))
    
    # Initialize labels
    kmeans_labels = np.zeros(x.shape[0])
    
    # Loop over new cluster keys
    for c in clusters_new:
        
        # Get the indacies in cluster clst
        c_idxs = [x.index.get_loc(k) for k in clusters_new[c]]
        
        # Save these
        kmeans_labels[c_idxs] = c
    
    # Record new Silhouette scores
    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index = x.index)
    
    return corr_new, clusters_new, silh_new


#------------------------------------------------------------------------------
def cluster_kmeans_top(corr0, max_clusters = None, n_init = 10):
    
    # If the maximum number of clusters is None, then set it equal to the number of columns minus 1
    if max_clusters is None: max_clusters = corr0.shape[0] - 1
    
    # Perform clustering
    corr1, clusters, silh = cluster_kmeans_base(corr0, 
                                                max_clusters = max_clusters, 
                                                n_init = n_init)
    
    # Calculate t-stat for each cluster
    t_stats = {c:np.mean(silh[clusters[c]])/np.std(silh[clusters[c]]) 
               for c in clusters}
    
    # Compute the mean t-stat
    t_stats_mean = np.mean(list(t_stats.values()))
    
    # Get list of clusters to redo
    redo = [c for c in t_stats if t_stats[c] < t_stats_mean]
    
    # If one terminate algorithm 
    if len(redo) <= 1:
        
        return corr1, clusters, silh
    
    # Otherwise
    else:
        
        # Get the keys for the observations to redo
        redo_keys = [key for c in redo for key in clusters[c]]
        
        # Subset the observation matrix
        corr_temp = corr0.loc[redo_keys, redo_keys]
        
        # Calculate the mean t-stat
        t_stats_mean = np.mean([t_stats[c] for c in redo])
        
        # Perform new clustering
        corr2, clusters2, silh2 = cluster_kmeans_top(corr_temp, 
                                                 max_clusters = max_clusters,
                                                 n_init = n_init)
        
        # Make new outputs if necessary
        corr_new, clusters_new, silh_new = make_new_outputs(corr0, 
                {c:clusters[c] for c in clusters if c not in redo}, 
                clusters2)
        
        # Calculate t-stats of newly generated clusters
        new_t_stats_mean = np.mean([np.mean(silh_new[clusters_new[c]])/np.std(silh_new[clusters_new[c]]) 
                                    for c in clusters_new])
        
        if new_t_stats_mean <= t_stats_mean:
            
            return corr1, clusters, silh 
        
        else:
            
            return corr_new, clusters_new, silh_new 
        

#------------------------------------------------------------------------------
def get_cov_sub(num_obs, num_cols, noise, random_state = None):
    
    # Sub correlation matrix
    random_gen = check_random_state(random_state)
    
    if num_cols == 1: return np.ones(shape = (1, 1))
    
    # Geneate random normals
    x = random_gen.normal(size = (num_obs, 1))
    
    # Repeat num_cols times and stack horizontally
    X = np.repeat(x, num_cols, axis = 1)
    
    # Add random noise 
    X += random_gen.normal(scale = noise, size = X.shape)
    
    # Calculate the covariance matrix
    cov = np.cov(X, rowvar = False)
    
    return cov


#------------------------------------------------------------------------------
def get_random_block_cov(num_cols, num_blocks, min_blocks = 1, noise = 1.0, 
                         random_state = None):
    
    # Generate a block random correlation matrix
    random_gen = check_random_state(random_state)
    
    # Generate random numebers without replacement
    parts = random_gen.choice(range(1, num_cols - (min_blocks - 1) * num_blocks), 
                       num_blocks - 1, replace = False)
    
    # Sort values
    parts.sort()
    
    # Add what's left as the last value
    parts = np.append(parts, num_cols - (min_blocks - 1) * num_blocks)
    
    # Take the difference, subtract 1 and add M to get the number in each block
    parts = np.append(parts[0], np.diff(parts)) - 1 + min_blocks
    
    # Initialize covariance matrix
    cov = None
    
    # Loop over each block
    for num_cols_ in parts:
        
        # Calculate the number of observations for each block
        num_obs_ = int(np.max([num_cols_ * (num_cols_ + 1)/2, 100]))
        
        # Get the covariance matrix for black
        cov_ = get_cov_sub(num_obs_, num_cols_, noise, random_state = random_gen)
        
        # Construct block covariance matrix
        cov = cov_.copy() if cov is None else block_diag(cov, cov_)
        
    return cov


#------------------------------------------------------------------------------
def random_block_corr(num_cols, num_blocks, min_blocks = 1, random_state = None):
    
    # Form block correlation
    random_gen = check_random_state(random_state)
    
    # Calculate signal covariance matrix
    cov = get_random_block_cov(num_cols, num_blocks, min_blocks = min_blocks, 
                          noise = 0.5, random_state = random_gen)
    
    # Create noise covariance matrix
    cov_noise = get_random_block_cov(num_cols, 1, min_blocks = min_blocks, 
                               noise = 1.0, random_state = random_gen)
    
    # Add noise covariance matrix to signal covariance matrix
    cov += cov_noise
    
    # Create diagonal matrix with entries 1 over std
    sd_inv = np.diag(1/np.sqrt(np.diag(cov)))
    
    # Convert covariance matrix to correlation matrix
    corr = sd_inv  @ cov @ sd_inv
    
    # Clip at -1 and 1 for numerical stability
    corr = pd.DataFrame(corr).clip(-1.0, 1.0)
    
    return corr