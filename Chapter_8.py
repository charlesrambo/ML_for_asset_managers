# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:04:22 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------------------------------
def get_expected_max_SR(num_trials, mean_SR, std_SR):
    
    # Expected max SR, controlling for selection bias under multiple testing (SBuMT)
    
    z = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/num_trials) + \
            np.euler_gamma * stats.norm.ppf(1 - 1/(num_trials * np.e))
            
    sharpe_ratio = mean_SR + std_SR * z
    
    return sharpe_ratio

#------------------------------------------------------------------------------
def get_max_SR_distribution(num_sims, trials_per_sim_list, mean_SR, std_SR, seed = None):
    
    # Monte Carlo of max{SR} on nTrials from nSims simulations
    
    np.random.seed(seed)
    
    out = pd.DataFrame()
    
    for num_trials in trials_per_sim_list:
        
        # 1) Simulated Sharpe ratios
        z = pd.DataFrame(stats.norm.rvs(size = (int(num_sims), int(num_trials))))
        z = z.sub(z.mean(axis = 1), axis = 0) # Center
        z = z.div(z.std(axis = 1), axis = 0) # scale
        sharpe_ratios = mean_SR + std_SR * z
        
        # 2) Store output
        out_ = sharpe_ratios.max(axis = 1).to_frame(r'max{SR}')
        out_['num_trials'] = num_trials
        
        out = pd.concat([out, out_], axis = 0, ignore_index = True)
        
    return out

#------------------------------------------------------------------------------
def get_mean_std_error(num_sims_per_run, num_runs, trials_per_sim_list, 
                       mean_SR = 0.0, std_SR = 1.0):
    
    # Compute standard deviation of errors per nTrials
    # nTrials: [number of SR used to derive max{SR}]
    # nSims0: number of max{SR} used to estimate E[max{SR}]
    # nSims1: number of errors on which std is computed
    
    sharpe_ratios = pd.Series(
        {num_trials:get_expected_max_SR(num_trials, mean_SR, std_SR) 
         for num_trials in trials_per_sim_list})
    
    sharpe_ratios = sharpe_ratios.to_frame('E[max{SR}]')
    
    sharpe_ratios.index.name = 'num_trials'
    
    error = pd.DataFrame()
    
    for _ in range(int(num_runs)):
        
        sharpe_ratio_sims = get_max_SR_distribution(num_sims = num_sims_per_run, 
                                      trials_per_sim_list = trials_per_sim_list, 
                                      mean_SR = 0, std_SR = 1.00)
        
        sharpe_ratio_sims = sharpe_ratio_sims.groupby('num_trials').mean()
        
        run_error = sharpe_ratios.join(sharpe_ratio_sims).reset_index()
        run_error['error'] = run_error['max{SR}']/run_error['E[max{SR}]'] - 1
        
        
        error = pd.concat([error, run_error], axis = 0, ignore_index = True)
        
    out = {'mean_error':error.groupby('num_trials')['error'].mean(), 
           'std_error':error.groupby('num_trials')['error'].std()} 
    
    out = pd.DataFrame.from_dict(out, orient = 'columns')
    
    return out

#------------------------------------------------------------------------------
def get_z_stat(SR, num_samples, SR_star = 0, skew = 0, kurt = 3):
    
    # Calculate top of ratio
    z = (SR - SR_star) * np.sqrt(num_samples - 1)  
    
    # Divide by bottom of ratio
    z /= np.sqrt(1 - skew * SR + (kurt - 1)/4 * SR**2)
    
    return z

#------------------------------------------------------------------------------
def get_type_1_error_prob(z, k = 1):
    
    # False positive rate
    alpha = stats.norm.cdf(-z)
    
    # Multiple-testing correction
    alpha_k = 1 - (1 - alpha)**2
    
    return alpha_k

#------------------------------------------------------------------------------
def get_theta(SR, num_samples, SR_star, skew = 0, kurt = 3):
    
    # Calculate top of ratio
    theta = SR_star * np.sqrt(num_samples - 1)
    
    # Divide by bottom of ratio
    theta /= np.sqrt(1 - skew * SR + (kurt - 1)/4 * SR**2)
    
    return theta

#------------------------------------------------------------------------------
def get_type_2_error_prob(alpha_k, k, theta):
    
    # False negative rate
    
    # Sidak's correction
    z = stats.norm.ppf(( 1- alpha_k)**(1/k))
    
    beta = stats.norm.cdf(z - theta)
    
    return beta


if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    #trials_per_sim_list = np.logspace(1, 6, 1000).astype(int) 
    trials_per_sim_list = np.logspace(1, 5, 100).astype(int)
    trials_per_sim_list.sort()
    
    sharpe_ratio_theoretical = pd.Series({num_trials:get_expected_max_SR(num_trials, mean_SR = 0, std_SR = 1) 
                                          for num_trials in trials_per_sim_list})
    sharpe_ratio_theoretical = pd.DataFrame(sharpe_ratio_theoretical, columns = ['max{SR}'])
    sharpe_ratio_theoretical.index.names = ['num_trials']
    
    sharpe_ratio_sims = get_max_SR_distribution(
                                    #num_sims = 1e3,
                                    num_sims = 100,
                                    trials_per_sim_list = trials_per_sim_list, 
                                    mean_SR = 0.0, 
                                    std_SR = 1.0)
    
    
    heatmap_df = sharpe_ratio_sims.copy()
    
    heatmap_df['count'] = 1
    heatmap_df['max{SR}'] = heatmap_df['max{SR}'].round(3)
    heatmap_df = heatmap_df.groupby(['num_trials', 'max{SR}']).count().reset_index()
    heatmap_df = heatmap_df.pivot(index = 'max{SR}', columns = 'num_trials', 
                                  values = 'count')
    heatmap_df = heatmap_df.fillna(0)
    
    heatmap_df = heatmap_df.sort_index(ascending = False)
    
    fig, ax = plt.subplots()
    
    sns.heatmap(heatmap_df, cmap = 'Blues', ax = ax)
    sns.lineplot(x = sharpe_ratio_theoretical.index, 
                 y = sharpe_ratio_theoretical['max{SR}'], 
                 linestyle = 'dashed', ax = ax)
    
    plt.show()
    
    mean_error = get_mean_std_error(
                                #num_sims_per_run = 1e3, 
                                num_sims_per_run = 100,
                                num_runs = 1e2, 
                                trials_per_sim_list = trials_per_sim_list, 
                                std_SR = 1.0)
    
    #--------------------------------------------------------------------------
    t, skew, kurt, k, freq = 1250, -3, 10, 10, 250
    sr = 1.25/np.sqrt(freq)
    sr_star = 1/np.sqrt(freq)
    
    z = get_z_stat(sr, t, 0, skew, kurt)
    alpha_k = get_type_1_error_prob(z, k = k)
    
    theta = get_theta(sr, t, sr_star, skew, kurt)
    beta = get_type_2_error_prob(alpha_k, k, theta)
    beta_k = beta**k
    
    
    
    
    
    
        
        
