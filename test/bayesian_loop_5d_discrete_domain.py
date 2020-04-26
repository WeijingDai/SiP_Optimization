# %%
# Import Botorch library
import torch
import os
import numpy as np
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models import FixedNoiseGP
from botorch.models import HeteroskedasticSingleTaskGP
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
print('Libraries imported')

# %%
# Load data
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
fig, ax = plt.subplots(ncols=2)
ax[0].hist(result, bins=25)
ax[1].hist(train_data[:,-1], bins=n_train_init)

# Specific selection
# x1, x2, x3, x4, x5 = np.mgrid[0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j]
# points = np.hstack((x1.reshape(32, 1), x2.reshape(32, 1), x3.reshape(32, 1), x4.reshape(32, 1), x5.reshape(32, 1))).transpose()
# train_data_32 = full_scan[points.astype(int)[0], points.astype(int)[1], points.astype(int)[2], points.astype(int)[3], points.astype(int)[4]]
# train_data_4 = train_data_32[np.random.randint(32, size=4)]

print('Initial data loaded')

# %%
# Implement PyTorch and Bayesian loop
acqf_para_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.double
design_domain = torch.as_tensor(full_scan[:, 0:n_variables], device=device, dtype=dtype)

for exp_run in range(1):
#for j in range(len(acqf_para_list)):

    acqf_para = 1
    print('Acquisition parameter = ', acqf_para)
    train_x = torch.as_tensor(train_data[:, 0:-1], dtype=dtype, device=device)
    train_y_origin = torch.as_tensor(train_data[:, -1], dtype=dtype, device=device).unsqueeze(1)*1000
    # train_y = train_y_origin**2
    train_y = torch.log(train_y_origin**2 + 1e-16) # To find y closet to zero 
    best_observed_value = train_y.min().item()

    verbose = False

    # Bayesian loop
    Trials = 50
    for trial in range(1, Trials+1):

        #print(f"\nTrial {trial:>2} of {Trials} ", end="\n")  
        #print(f"Current best: {best_observed_value} ", end="\n")

        # fit the model
        model = FixedNoiseGP(train_x, -train_y, train_Yvar=torch.full_like(train_y, 0.1)).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # for best_f, we use the best observed values as an approximation
        # EI = ExpectedImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        UCB = UpperConfidenceBound(model = model, beta = acqf_para)
        # PI = ProbabilityOfImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        
        # Evaluate acquisition function over the discrete domain of parameters
        acqf = UCB(design_domain.unsqueeze(-2))
        acqf_sorted = torch.argsort(acqf, descending=True)
        acqf_max = acqf_sorted[0].unsqueeze(0)
        for j in range(1, 10):
            if acqf[acqf_max[0]] > acqf[acqf_sorted[j]]:
                break
            else:
                acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        print(acqf_max.numpy())
        candidate_id = acqf_max[torch.randint(len(acqf_max), size=(1, ))]
        candidate = design_domain[candidate_id]
        new_y = result[candidate_id]
        train_new_y_origin = torch.as_tensor([[new_y]], dtype=dtype, device=device)*1000
        # train_new_y = train_new_y_origin**2
        train_new_y = torch.log(train_new_y_origin**2 + 1e-16)
        # update training points
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, train_new_y])
        train_y_origin = torch.cat([train_y_origin, train_new_y_origin])

        current_value = train_new_y.item()
        best_observed_value = train_y.min().item()

        if False:
            print(
                f"\nTrial {trial:>2}: current_value = "
                f"{current_value}, "
                f"best_value = "
                f"{best_observed_value} ", end=".\n"
                )
        else:
            print(".", end="")
    optim_result = torch.cat([train_x, train_y_origin, train_y], 1)
    print('Bayesian loop completed')
    fig1, ax1 = plt.subplots()
    y_plot = abs(train_y_origin.numpy())
    ax1.plot(y_plot, marker='.')
    ax1.set_yscale('log')
    ax1.set_ylim(auto=True)
    ax1.set_xlabel('Loop iteration', fontsize='large')
    ax1.set_ylabel('Absolute warpage (um)', fontsize='large')
# %%


