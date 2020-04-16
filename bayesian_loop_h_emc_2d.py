#%%
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
import matplotlib.pyplot as plt
from matplotlib import cm

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.double

def generate_initial_data():
    # generate training data
    # call abaqus to run initial scripts
    # command = "abaqus cae noGUI=C:/SUSTech-Postdoc/SDIM-Projects/Warpage/bayesian_loop/bayesian_initial.py && exit"
    # os.system(command)

    # read simulated result from file
    with open('D:/ABAQUS/Warpage/ke_ch2/2_variables/EMC_h_cte_2d/init_data_cte_h_cte_subwarp.csv') as f:
        train_data = np.loadtxt(f, delimiter=",")
    train_x = torch.as_tensor(train_data[:, 0:-1], dtype=dtype, device=device)
    train_x = train_x
    train_y_origin = torch.as_tensor(train_data[:, -1], dtype=dtype, device=device).unsqueeze(1)
    train_y = train_y_origin**2
    best_observed_value = train_y.max().item()
    return train_x, train_y_origin, train_y, best_observed_value


# def generate_new_data(new_x):
#     # generate training data
#     temp = new_x
#     np.savetxt('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/bayesian_loop/EMC_h_cte_2d/pi/temp.csv', temp, delimiter=',')
#     # call abaqus to run iteration scripts
#     command = "abaqus cae noGUI=C:/SUSTech-Postdoc/SDIM-Projects/Warpage/bayesian_loop/EMC_h_cte_2d/iteration.py && exit"
#     os.system(command)

#     # read new simulated result from file
#     with open('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/bayesian_loop/EMC_h_cte_2d/pi/temp.csv') as f:
#         train_new = np.loadtxt(f, delimiter=",")
#     new_x = torch.as_tensor(train_new[1:-1], dtype=dtype, device=device)
#     new_x = new_x.unsqueeze(0)
#     new_y = torch.as_tensor(train_new[-1], dtype=dtype, device=device)**2
#     new_y = new_y.unsqueeze(-1).unsqueeze(-1)
#     return new_x, new_y

# Parameter domain
h_emc = torch.linspace(550., 950., 21, device=device)
cte_emc = torch.linspace(80., 120., 21, device=device)
h_grid, cte_grid = torch.meshgrid(h_emc, cte_emc)
h_cte_emc = torch.cat((h_grid.unsqueeze(-1), cte_grid.unsqueeze(-1)), -1).double()
m = h_cte_emc.shape[0]
n = h_cte_emc.shape[1]

verbose = True
train_x, train_y_origin, train_y, best_observed_value = generate_initial_data()
best_observed = []

with open('D:/ABAQUS/Warpage/ke_ch2/2_variables/EMC_h_cte_2d/full_scan/full_scan.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',').reshape(21, 21, 3)

# Bayesianc loop
Trials = 100
for trial in range(1, Trials + 1):

    print(f"\nTrial {trial:>2} of {Trials} ", end="\n")

    # call helper functions to generate initial training data and initialize model
    
    print(f"Current best: {best_observed_value} ", end="\n")

    # run N_BATCH rounds of BayesOpt after the initial random batch
    # fit the model
    model = FixedNoiseGP(train_x, -train_y, train_Yvar=torch.full_like(train_y, 0.000001)).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # for best_f, we use the best observed values as an approximation
    # EI = ExpectedImprovement(model = model, best_f = -train_y.min() * 1.,)
    UCB = UpperConfidenceBound(model = model, beta = 1)
    # PI = ProbabilityOfImprovement(model = model, best_f = -train_y.min() * 1.,)

    # Evaluate acquisition function over the discrete domain of parameters
    acqf = UCB(h_cte_emc.unsqueeze(2))
    np.savetxt('D:/ABAQUS/Warpage/ke_ch2/2_variables/EMC_h_cte_2d/ucb_10/Acquisition_' + str(trial) + '.csv', acqf.detach().cpu().numpy(), delimiter=',')
    candidate = h_cte_emc[acqf.argmax()//m, acqf.argmax()%n]
    print(acqf.argmax().item()//m, acqf.argmax().item()%n)
    # norm = cm.colors.Normalize(vmax=acqf.detach().numpy().max(), vmin=acqf.detach().numpy().min())
    # fig, ax = plt.subplots()
    # ax.set_aspect('auto')
    # cset1 = ax.contourf(h_emc.numpy(), cte_emc.numpy(), acqf.detach().squeeze(-1).numpy(), len(h_emc), norm=norm)
    # ax.plot(candidate.numpy()[0], candidate.numpy()[1], "*", c='red')
    # fig.colorbar(cset1)
    # ax.set_xlim(h_emc.min().item(), h_emc.max().item())
    # ax.set_ylim(cte_emc.min().item(), cte_emc.max().item())
    # plt.savefig('D:/ABAQUS/Warpage/ke_ch2/2_variables/EMC_h_cte_2d/ucb_01/' + str(trial) + '.jpg')
    # get new observation
    new_x = candidate.detach().cpu().numpy()
    new_y = full_scan[acqf.argmax()//m, acqf.argmax()%n, 2]
    print(new_x, new_y)
    train_new_x =  candidate.unsqueeze(0)
    train_new_y_origin = torch.as_tensor(new_y, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
    train_new_y = train_new_y_origin**2
    # update training points
    train_x = torch.cat([train_x, train_new_x])
    train_y = torch.cat([train_y, train_new_y])
    train_y_origin = torch.cat([train_y_origin, train_new_y_origin])

    current_value = train_new_y.item()
    best_observed_value = train_y.min().item()
    best_observed.append(best_observed_value)

    if verbose:
        print(
            f"\nTrial {trial:>2}: current_value = "
            f"{current_value}, "
            f"best_value = "
            f"{best_observed_value} ", end=".\n"
            )
    else:
        print(".", end="")
optim_result = torch.cat([train_x, train_y_origin, train_y], 1)
np.savetxt('D:/ABAQUS/Warpage/ke_ch2/2_variables/EMC_h_cte_2d/ucb_10/Optimization_loop.csv', optim_result.cpu().numpy(), delimiter=',')

# %%
