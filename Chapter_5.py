# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:42:57 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = r'G:/USERS/CharlesR/Python/ML_for_asset_managers/'
plt.style.use("seaborn-v0_8")


#------------------------------------------------------------------------------
def calculate_t_value(close):
    
    # Get the number of observations
    num_obs = close.shape[0]
    
    # Create the x-value
    x = np.concatenate([np.ones((num_obs, 1)), 
                        np.arange(num_obs).reshape((-1, 1))], axis = 1)
    
    # Fit OLS
    ols = sm.OLS(close, x).fit()
    
    # Return the t-value
    return ols.tvalues[1]


#------------------------------------------------------------------------------
def get_bins_from_trend(date_index, close, start, stop, step = 1):
    """
    Derives labels (t1, t-value, and bin) based on the sign of the t-value of 
    the linear trend for each date in the date_index list within the close 
    price series.

    Parameters:
    ----------
    date_index : list or pandas.Index
        List of dates for which trends are to be analyzed.
    close : pandas.Series
        Series containing closing prices for the time period.
    start : int
        Starting offset for trend window (relative to date in date_index).
    stop : int
        Ending offset for trend window (relative to date in date_index).
    step : int, optional
        Step size for iterating through the trend window (default: 1).

    Returns:
    -------
    pandas.DataFrame
        Dataframe containing columns for 't1' (end date of identified trend),
        't-val' (t-value associated with the trend coefficient), and 'bin'
        (sign of the trend).
    """
    
    # Initialize data frame to hold results
    out = pd.DataFrame(index = date_index, columns = ['t1', 't-val', 'bin'])
    
    # Loop over index values
    for start_date in date_index:
        
        # Initialize pandas series
        trend_data = pd.Series()
        
        # To the location of start_date in the index
        i = close.index.get_loc(start_date)
        
        # Make sure we don't run out of room on the other side
        if i + stop >= close.shape[0]: continue
    
        for j in range(start, stop, step):
            
            # Get the end date of the range
            end_date = close.index[i + j]
            
            # Calculate the t-values
            trend_data.loc[end_date] = calculate_t_value(close.loc[start_date:end_date])
         
        # Get the location of the max t-value
        max_t_date = trend_data.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        
        # Record results
        out.loc[start_date, :] = (trend_data.index[-1], trend_data[max_t_date], 
                           np.sign(trend_data[max_t_date]))
     
    # Covert t1 to a datetime
    out['t1'] = pd.to_datetime(out['t1'])
    
    # Calculate bin
    out['bin'] = pd.to_numeric(out['bin'], downcast = 'signed')
    
    return out.dropna(subset = ['bin'])


if __name__ == '__main__':
    
    # Set random seet
    np.random.seed(0)
    
    # Set number of observations in series
    num_obs = 100
    
    # Generate series
    data = pd.Series(np.cumsum(np.random.normal(0, .1, num_obs))
                        + np.sin(np.linspace(0, 10, num_obs)))
    
    # Get results
    results = get_bins_from_trend(data.index, data, start = 3, stop = 10)
    
    # Subset data
    data = data.loc[results.index]
    
    # Draw scatter plot of series
    plt.scatter(x = results.index, y = data.values, 
                        c = results['bin'].values, cmap = 'viridis')
    
    # Give plot a title
    plt.title('Up and Down Trends')
    
    # Save figure
    plt.savefig(path + 'fig5.1.png')
    
    # Show figure
    plt.show()