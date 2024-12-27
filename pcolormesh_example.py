# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:40:32 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys



# Make the directory Python because that's where cal_signals is located
sys.path.insert(0, path)

# Get array of values to use
num_vals = np.logspace(0.5, 3.5, 100).astype(int)

# Define number of runs and number of runs per trial
num_runs = 10000
trials_per_run = 10

# Initialize array to hold exprimental values
exprimental_values = np.zeros(shape = (num_runs * len(num_vals), 2))


for i, n in enumerate(num_vals):
    
    for j in range(num_runs):
        
        # Generate normal values and take the average of row
        dist = stats.norm.rvs(size = (trials_per_run, n)).mean(axis = 1)
        
        # Recrod results
        exprimental_values[num_runs * i + j, 0] = n
        exprimental_values[num_runs * i + j, 1] = np.std(dist, ddof = 1)
        
# Convert to a pandas data frame
exprimental_values = pd.DataFrame(exprimental_values, columns = ['num', 'std'])

# Calculate theoretical values
theoretical_values = pd.Series(1/np.sqrt(num_vals), index = num_vals)
    
# Make a deep copy of the exprimental values         
freq_df = exprimental_values.copy()

# Rework so we get normalized frequencies
freq_df['count'] = 1

# Discretize standard deviations
freq_df['std'] = freq_df['std'].round(3)

# Count number of elements in each bin
freq_df = freq_df.groupby(['num', 'std'])['count'].sum().reset_index()

# Rows are the standard deviation, columns are the number of elements averaged, entry is the count
freq_df = freq_df.pivot(index = 'std', columns = 'num', values = 'count') 

# Fill empty values with zero
freq_df = freq_df.fillna(0)

# Divide by the sum of the column      
freq_df = freq_df.div(freq_df.sum(axis = 0), axis = 1)

# Sort values
freq_df = freq_df.sort_index(ascending = False)

# Get figure and axes
fig, ax = plt.subplots()

# Convert to grid
X, Y = np.meshgrid(freq_df.columns, freq_df.index)

# Get the frequency numbers
Z = freq_df.values[1:, 1:]

# Generate the pcolormesh
c = ax.pcolormesh(X, Y, Z, cmap = 'Blues', shading = 'flat', norm = 'log', 
                  label = 'simulated')

# Add a color bar
fig.colorbar(c, ax = ax)

# Place theoretical values on graph
theoretical_values.plot(ax = ax, color = 'red', label = '$y = 1/\sqrt{x}$')

# Add a title
ax.set_title('Experiment and Theoretical Results for SD of Mean')

# Add axis labels
ax.set_xlabel('Number of Samples Used to Calculate Mean')
ax.set_ylabel('Standard Deviation of Mean')

# Add legend
plt.legend()

plt.savefig(path + 'fig_pcolormesh_example.png')

# Show the figure
plt.show()