# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:52:00 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection._split import KFold
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LinearRegression
import sys




# Change director to path
sys.path.insert(0, path)

# Import the chapter four
import Chapter_4 as four

# See https://stackoverflow.com/questions/69596239/how-to-avoid-memory-leak-when-dealing-with-kmeans-for-example-in-this-code-i-am
import warnings
warnings.filterwarnings('ignore')

plt.style.use("seaborn-v0_8")

#------------------------------------------------------------------------------
def gen_test_data(n_features = 100, n_informative = 25, n_redundant = 25, 
                n_samples = 10000, random_state = 0, scale = 0, is_clf = True):
    
    
    # Generate a random dataset for a classification/regression problem
    np.random.seed(random_state)
    
    # Calculate the number of noise features
    n_noise = n_features - n_informative - n_redundant
    
    if is_clf:
        
        # Use make_classification to construct informative and noise features
        X, y = make_classification(n_samples = n_samples,
                                   n_features = n_features - n_redundant,
                                   n_informative = n_informative,
                                   n_redundant = 0,
                                   shuffle = False,
                                   random_state = random_state)
        
        # Think Lopez de Prado not using n_redundant because linear combinations of informative features
        # Too hard for clustering to untangle
        
    else:
        
        # Use make_regression to construct informative and noise features
        X, y = make_regression(n_samples = n_samples,
                               n_features = n_features - n_redundant,
                               n_informative = n_informative,
                               shuffle = False,
                               random_state = random_state)    
    
    # Add names for the informative features
    cols = [f'I_{i}' for i in range(n_informative)]
    
    # Add names for the noise features
    cols += [f'N_{i}' for i in range(n_noise)]
    
    # Convert results to a pandas data frame
    X, y = pd.DataFrame(X, columns = cols), pd.Series(y)
    
    # Randomly choose which features the redundant ones replicate
    rep = np.random.choice(range(n_informative), size = n_redundant)
    
    for j, k, in enumerate(rep):
        
        # Redundant feature j is informative feature k plus random noise
        X[f'R_{j}'] = X[f'I_{k}'] + np.random.normal(size = n_samples, 
                                                     scale = scale)
        
    return X, y


#------------------------------------------------------------------------------
def orthogonalize_clusters(df, clusters):
    
    # Make a deep copy of df
    df_copy = df.copy()
    
    # Get the values of the clusters
    vals = list(clusters.values())
    
    # Loop over the clusters
    for i, responses in enumerate(vals):
        
        if i != 0:
            
            # Get the explanitory columns
            exp_vars = [col for j in range(i) for col in vals[j]]
            
            # Loop over response columns
            for response in responses:
                
                # Get X and y
                X = df_copy[exp_vars].values                
                y = df_copy[response].values

                # Need to make sure X is a 2D matrix
                if len(exp_vars) == 1:
                    
                    X = X.reshape((-1, 1))
                 
                # Initialize and fit the linear regression
                reg = LinearRegression().fit(X, y)
                
                # Calculate the residuals
                df_copy[response] = y - reg.predict(X)
                
    return df_copy
        
        
#------------------------------------------------------------------------------
def feat_imp_MDI(clf, feat_names):
    
    # Feature importance based on in-sample mean impurity reduction
    df = {i:tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    
    # Convert from dictionary to data frame
    df = pd.DataFrame.from_dict(df, orient = 'index')
    
    # Name the columns
    df.columns = feat_names
    
    # Because max_features = 1
    df = df.replace(0, np.nan)
    
    # Calculate the mean and std of the samples
    imp = pd.concat({'mean':df.mean(), 'std':df.std()/np.sqrt(df.shape[0])}, 
                    axis = 1)
    
    # Rescale by dividing by mean
    imp /= imp['mean'].sum()
    
    return imp


#------------------------------------------------------------------------------
def feat_imp_MDA(clf, X, y, n_splits = None, cv = None):
    
    if cv is None:
        
        # Initialize k-folds constructor
        cv_gen = KFold(n_splits = n_splits).split(X = X)
        
    else:
        
        cv_gen = cv.split(X = X)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = X.columns)
    
    # Generate split
    for fold, (train_idx, test_idx) in enumerate(cv_gen):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using the training data
        clf_fit = clf.fit(X = X_train, y = y_train)
        
        # Use testing data to predict probabilities
        probs = clf_fit.predict_proba(X_test)
        
        # Record negative log-loss
        score_raw.loc[fold] = -log_loss(y_test, probs, labels = clf.classes_)
        
        for col in X.columns:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # Shuffle column
            np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities
            probs_shuff = clf_fit.predict_proba(X_shuff)
            
            # Calculate the score
            score_shuff.loc[fold, col] = -log_loss(y_test, probs_shuff, 
                                                   labels = clf.classes_)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by dividing by the shuffled score
    imp = imp/score_shuff
    
    # Compute the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Calculate t-stat
    imp.loc[imp['std'] != 0, 't-stat'] = imp.loc[imp['std'] != 0, 
                                    'mean']/imp.loc[imp['std'] != 0, 'std']
    
    return imp


#------------------------------------------------------------------------------
def reg_feat_imp_MDA(reg, X, y, n_splits = None, p = 2, cv = None):
    
    # Define penalty function
    penalty_fun = lambda e: np.sum(np.abs(e)**p)
    
    if cv is None:
        
        # Initialize k-folds constructor
        cv_gen = KFold(n_splits = n_splits).split(X = X)
        
    else:
        
        cv_gen = cv.split(X = X)
        
    # Initialize pandas objects to hold raw and shuffled scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = X.columns)
    
    # Generate split
    for fold, (train_idx, test_idx) in enumerate(cv_gen):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using the training data
        reg_fit = reg.fit(X = X_train, y = y_train)
        
        # Use testing data to predict probabilities
        y_pred = reg_fit.predict(X_test)
        
        # Record score
        score_raw.loc[fold] = -penalty_fun(y_test - y_pred)
        
        for col in X.columns:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # Shuffle the j-th column
            np.random.shuffle(X_shuff[col].values)
            
            # Predict values
            y_shuff = reg_fit.predict(X_shuff)
            
            # Calculate the score
            score_shuff.loc[fold, col] = -penalty_fun(y_test - y_shuff)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by dividing by the shuffled score
    imp = imp/score_shuff
    
    # Compute the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Calculate t-stat
    imp.loc[imp['std'] != 0, 't-stat'] = imp.loc[imp['std'] != 0, 
                                    'mean']/imp.loc[imp['std'] != 0, 'std']
    
    return imp
        

#------------------------------------------------------------------------------
def group_mean_std(df, clusters):
    
    # Initialize data frame for output
    out = pd.DataFrame(columns = ['mean', 'std'])
    
    # Loop over clusters
    for clst, col in clusters.items():
        
        # Take the sum of the values for each cluster
        temp = df[col].sum(axis = 1)
        
        # Compute the mean value
        out.loc[f'C_{clst}', 'mean'] = temp.mean()
        
        # Compute the standard deviation 
        out.loc[f'C_{clst}', 'std'] = temp.std()/np.sqrt(temp.shape[0])
        
        # Calculate t-stat
        if temp.std() != 0:
            
            out.loc[f'C_{clst}', 't-stat'] = out.loc[f'C_{clst}', 
                                    'mean']/out.loc[f'C_{clst}', 'std']
            
    return out


#------------------------------------------------------------------------------
def feat_imp_MDI_clustered(clf, feat_names, clusters):
    
    # Feature importance based on in-sample mean impurity reduction
    df = {i:tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    
    # Convert dictionary to data frame
    df = pd.DataFrame.from_dict(df, orient = 'index')
    
    # Rename columns
    df.columns = feat_names
    
    # Replace 0 with np.nan
    df = df.replace(0, np.nan)
    
    # Get impurity of each cluster
    imp = group_mean_std(df, clusters)
    
    # Divide by sum to normalize
    imp /= imp['mean'].sum()
    
    return imp


#------------------------------------------------------------------------------
def feat_imp_MDA_clustered(clf, X, y, clusters, n_splits = None, cv = None):
    
    if cv is None:
        
        # Initialize k-folds constructor
        cv_gen = KFold(n_splits = n_splits).split(X = X)
        
    else:
        
        cv_gen = cv.split(X = X)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = clusters.keys())
    
    # Generate splits
    for fold, (train_idx, test_idx) in enumerate(cv_gen):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using training data
        clf_fit = clf.fit(X = X_train, y = y_train)
        
        # Use the fitted model to predict probabilities
        probs = clf_fit.predict_proba(X_test)
        
        # Record log-loss
        score_raw.loc[fold] = -log_loss(y_test, probs, labels = clf.classes_)
        
        # Loop over clusters
        for clst in clusters:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # For each column in clst
            for col in clusters[clst]:
                
                # Shuffle col
                np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities with shuffled results
            probs_shuff = clf_fit.predict_proba(X_shuff)
            
            # Calcualte the score
            score_shuff.loc[fold, clst] = -log_loss(y_test, probs_shuff, 
                                                    labels = clf.classes_)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by the shuffled scores
    imp = imp/score_shuff
    
    # Calculate the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Calculate t-stat
    imp.loc[imp['std'] != 0, 't-stat'] = imp.loc[imp['std'] != 0, 
                                    'mean']/imp.loc[imp['std'] != 0, 'std']
    
    # Change the index name
    imp.index = [f'C_{i}' for i in imp.index]
        
    return imp


#------------------------------------------------------------------------------
def reg_feat_imp_MDA_clustered(reg, X, y, clusters, n_splits = 10, p = 2):
    
    # Define penalty function
    penalty_fun = lambda e: np.sum(np.abs(e)**p)
    
    # Initialize k-folds constructor
    cv_gen = KFold(n_splits = n_splits)
    
    # Initialize pandas objects to hold raw and shuffled scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = clusters.keys())
    
    # Generate splits
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using training data
        reg_fit = reg.fit(X = X_train, y = y_train)
        
        # Use the fitted model to predict probabilities
        y_pred = reg_fit.predict(X_test)
        
        # Record score
        score_raw.loc[fold] = -penalty_fun(y_test - y_pred)
        
        # Loop over clusters
        for clst in clusters:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # For each column in clst
            for col in clusters[clst]:
                
                # Shuffle col
                np.random.shuffle(X_shuff[col].values)
            
            # Predict the values with shuffled results
            y_shuff = reg_fit.predict(X_shuff)
            
            # Calculate the score
            score_shuff.loc[fold, clst] = -penalty_fun(y_test - y_shuff)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by the shuffled scores
    imp = imp/score_shuff
    
    # Calculate the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Calculate t-stat
    imp.loc[imp['std'] != 0, 't-stat'] = imp.loc[imp['std'] != 0, 
                                    'mean']/imp.loc[imp['std'] != 0, 'std']
    
    # Change the index name
    imp.index = [f'C_{i}' for i in imp.index]
        
    return imp


#------------------------------------------------------------------------------
def plot_logistic_pvals(X, y, feat_names, alpha = 0.05, title = None, 
                        filename = None, **kwargs):
    
    # Import logistic regression
    from statsmodels.discrete.discrete_model import Logit
    
    # Fit the regression
    logit = Logit(y.values, X.values).fit()
    
    # Record the p-values
    p_vals = pd.Series(logit.pvalues, index = feat_names)
    
    # Sort the p-values
    p_vals = p_vals.sort_values(ascending = False)
    
    # Initialize figure size of plot
    plt.figure(figsize = (10, int(np.max([10, p_vals.shape[0]/4]))))
    
    # Plot histogram
    ax = p_vals.plot(kind = 'barh', color = 'b', alpha = 0.25, 
                     error_kw = {'ecolor':'r'})
        
    # Set x-range
    plt.xlim([-0.01 * np.max([p_vals.max(), alpha]), 
              1.01 * np.max([p_vals.max(), alpha])])
        
    # draw verticle line at significance level
    plt.axvline(alpha, linewidth = 1, color = 'r', linestyle = 'dotted')

    # Make the y-axis invisible
    ax.get_yaxis().set_visible(False)

    # Place feature name as center of each bar
    for bar, feature_name in zip(ax.patches, p_vals.index): 
        
        ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                feature_name, ha = 'center', va = 'center', color = 'black')
    
    if title is not None:
        
        # Give plot title
        plt.title(title)
    
    # If filename is defined...
    if filename is not None:
        
        # ... save the plot
        plt.savefig(filename, dpi = 100)
    
    # Show the plot
    plt.show()
    
    # Close figure
    plt.close()
  
#------------------------------------------------------------------------------
def plot_feat_importance(imp, method, title = None, filename = None, **kwargs):
    
    # Plot mean imp bars with std
    plt.figure(figsize = (10, int(np.max([10, imp.shape[0]/4]))))
    
    # Sort features by mean
    imp = imp.sort_values('mean', ascending = True)
    
    # Plot histogram
    ax = imp['mean'].plot(kind = 'barh', color = 'b', alpha = 0.25,
                          xerr = imp['std'], error_kw = {'ecolor':'r'})
    
    # If MDI...
    if method == 'MDI':
        
        # ... set x-range
        plt.xlim([-0.01 * imp[['mean', 'std']].sum(axis = 1).max(), 
                  1.01 * imp[['mean', 'std']].sum(axis = 1).max()])
        
        # ... draw verticle line to see average feature importance
        plt.axvline(1.0/imp.shape[0], linewidth = 1, color = 'r', 
                    linestyle = 'dotted')

    # Make the y-axis invisible
    ax.get_yaxis().set_visible(False)

    # Place feature name as center of each bar
    for bar, feature_name in zip(ax.patches, imp.index): 
        
        ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                feature_name, ha = 'center', va = 'center', color = 'black')
    
    if title is not None:
        
        # Give plot title
        plt.title(title)
    
    # If filename is defined...
    if filename is not None:
        
        # ... save the plot
        plt.savefig(filename, dpi = 100)
    
    # Show the plot
    plt.show()
    
    # Close figure
    plt.close()


if __name__ == '__main__':
    
    # Start the clock!
    start_time = time.perf_counter()
    
    # Set random seet
    np.random.seed(0)
    
    # Generate the data; in this case we're doing a classification problem
    X, y = gen_test_data(n_features = 40, n_informative = 5, n_redundant = 30, 
                         n_samples = 10000, random_state = 0, scale = 0.5)
    
    # Plot the p-values
    plot_logistic_pvals(X, y, feat_names = X.columns, 
                        title = '$p$-values\n $\\alpha = 5\%$',
                        filename = path + 'fig6.1.png')
    
    # Initialize an ML model
    clf = DecisionTreeClassifier(criterion = 'entropy', max_features = 1, 
                                 class_weight = 'balanced')
    
    # Using bagged decision trees to be consistant with book, but can use anything
    clf = BaggingClassifier(estimator = clf, n_estimators = 1000,
                            max_features = 1.0, max_samples = 1.0, 
                            oob_score = False)
    
    # Fit ML model
    clf.fit(X, y)
    
    # Calculate MDI feature importances
    mdi = feat_imp_MDI(clf, feat_names = X.columns)
    
    # Generate plot of feature importances
    plot_feat_importance(mdi, method = 'MDI',
                         title = 'Feature Importance MDI',
                         filename = path + 'fig6.2.png')
    
    # Calculate MDA feature importances
    mda = feat_imp_MDA(clf, X, y, n_splits = 5)  

    # Generate plot of feature importances
    plot_feat_importance(mda, method = 'MDA',
                         title = 'Feature Importance MDA',
                         filename = path + 'fig6.3.png')
    
    # Calculate correlations between explanatory variables
    corr0 = X.corr()
    
    # I'm skipping figure 6.4, because I don't want to use Seaborn
    
    # Cluster results
    corr1, clusters, silh = four.cluster_kmeans_top(corr0)

    # Calculate clustered MDI feature importances
    mdi_clustered = feat_imp_MDI_clustered(clf, feat_names = X.columns, 
                                           clusters = clusters)

    # Generate plot of feature importances    
    plot_feat_importance(mdi_clustered, method = 'MDI',
                         title = 'Feature Importance MDI Clustered',
                         filename = path + 'fig6.5.png')
    
    # Calculate clustered MDA feature importances
    mda_clustered = feat_imp_MDA_clustered(clf, X, y, clusters, n_splits = 5)
    
    # Generate plot of feature importances
    plot_feat_importance(mda_clustered, method = 'MDA',
                         title = 'Feature Importance MDA Clusted',
                         filename = path + 'fig6.6.png')
    
    # Orthoginalize clusters
    X_prime = orthogonalize_clusters(X, clusters)
    
    # Fit classifier on orthoginalized clusters
    clf.fit(X_prime, y)
    
    # Calculate clustered MDI feature importances
    mdi_prime_clustered = feat_imp_MDI_clustered(clf, feat_names = X_prime.columns, 
                                       clusters = clusters)
    
    # Generate plot of feature importances
    plot_feat_importance(mdi_prime_clustered, method = 'MDI',
                         title = 'Feature Importance MDI Clustered with Modified X')
    
    # Calculate clustered MDA feature importances
    mda_prime_clustered = feat_imp_MDA_clustered(clf, X_prime, y, clusters, n_splits = 5)  
    
    # Generate plot of feature importances
    plot_feat_importance(mda_prime_clustered, method = 'MDA',
                         title = 'Feature Importance MDA Clustered with Modified X')
    
    #--------------------------------------------------------------------------
    
    # Generate the data; in this case we're doing a regression problem
    X, y = gen_test_data(n_features = 40, n_informative = 5, n_redundant = 30, 
                         n_samples = 10000, random_state = 0, scale = 0.5, 
                         is_clf = False)
    
    # Initialize regressor
    reg = DecisionTreeRegressor(criterion = 'squared_error', max_features = 1)
    
    # We're doing bagged regression trees to be consistant with classification problem above 
    reg = BaggingRegressor(estimator = reg, n_estimators = 1000,
                            max_features = 1.0, max_samples = 1.0, 
                            oob_score = False)

    # Fit the ML model
    reg.fit(X, y)

    # Calculate MDI feature importances    
    mdi_reg = feat_imp_MDI(reg, feat_names = X.columns)

    # Generate plot of feature importances    
    plot_feat_importance(mdi_reg, method = 'MDI', 
                         title = 'Regression Feature Importance MDI')
    
    # Calculate MDA feature importances
    mda_reg = reg_feat_imp_MDA(reg, X, y, n_splits = 5)  

    # Generate plot of feature importances   
    plot_feat_importance(mda_reg, method = 'MDA',
                         title = 'Regression Feature Importance MDA')
    
    # Calculate correlations between explanatory variables
    corr0 = X.corr()
    
    # Cluster results
    corr1, clusters, silh = four.cluster_kmeans_top(corr0)
    
    # Calculate clustered MDI feature importances
    mdi_reg_clustered = feat_imp_MDI_clustered(reg, feat_names = X.columns, 
                                               clusters = clusters)
    
    # Generate plot of feature importances  
    plot_feat_importance(mdi_reg_clustered, method = 'MDI',
                         title = 'Regression Feature Importance MDI Clustered')
    
    # Calculate clustered MDA feature importances
    mda_reg_clustered = reg_feat_imp_MDA_clustered(reg, X, y, clusters, n_splits = 5)
    
    # Generate plot of feature importances  
    plot_feat_importance(mda_reg_clustered, method = 'MDA',
                         title = 'Regression Feature Importance MDA Clusted')
     
    # Orthoginalize clusters
    X_prime = orthogonalize_clusters(X, clusters)
    
    # Fit regression on orthoginalized clusters
    reg.fit(X_prime, y)
    
    # Calculate clustered MDI feature importances
    mdi_prime_clustered = feat_imp_MDI_clustered(reg, feat_names = X_prime.columns, 
                                       clusters = clusters)

    # Generate plot of feature importances      
    plot_feat_importance(mdi_prime_clustered, method = 'MDI',
                         title = 'Regression Feature Importance MDI Clustered with Modified X')
    
    # Calculate clustered MDA feature importances
    mda_prime_clustered = reg_feat_imp_MDA_clustered(reg, X_prime, y, clusters = clusters, 
                                           n_splits = 5)  
    
    # Generate plot of feature importances  
    plot_feat_importance(mda_prime_clustered, method = 'MDA',
                         title = 'Regression Feature Importance MDA Clustered  with Modified X')
    
    print(f'This program took {(time.perf_counter() - start_time)/60:.2f} minutes.') 