# %%
import numpy as np
import random
n_h_emc = 5
n_cte_emc = 5
n_h_sub = 5
n_h_adhesive = 5
n_h_die = 5
n_variables = 5
with open('./full_scan_5d.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',')
full_scan[:, 0] = (full_scan[:, 0]-200)/30
full_scan[:, 1] = (full_scan[:, 1]-20)/5
full_scan[:, 2] = (full_scan[:, 2]-200)/25
full_scan[:, 3] = (full_scan[:, 3]-550)/100
full_scan[:, 4] = (full_scan[:, 4]-8)/1
result = full_scan[:, -1]
n_train_init = 2 # Number of initial data
# Ramdon selection
train_data = full_scan[random.sample(range(0, len(result)), n_train_init)]

# %%
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

from numpy.linalg import inv

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s
# %%
train_x = train_data[:, 0:-1]
train_y = train_data[:, -1]

x1, x2, x3, x4, x5 = np.mgrid[0:5, 0:5, 0:5, 0:5, 0:5]
x_t = np.concatenate((x1[:, :, :, :, :, None], x2[:, :, :, :, :, None], x3[:, :, :, :, :, None], 
x4[:, :, :, :, :, None], x5[:, :, :, :, :, None]), -1).reshape(-1, 5)
mu_, cov_ = posterior_predictive(x_t, train_x, train_y)


# %%
