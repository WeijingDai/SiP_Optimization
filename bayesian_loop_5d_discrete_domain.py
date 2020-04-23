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
f_path = 'C:/Users/weijing/Documents/Nutstore/Abaqus_warpage_result/5_variables_result/'
f_path_c = 'C:\\Users\\weijing\\Documents\\Nutstore\Abaqus_warpage_result\\5_variables_result\\'
with open(f_path + 'full_scan.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',').reshape(n_h_die, n_h_adhesive, n_h_sub, n_h_emc, n_cte_emc, n_variables+1)

# %%
result = full_scan[:, :, :, :, :, -1].flatten()*1000
fig, ax = plt.subplots()
ax.hist(result, bins=25)
# %%
n_train_init = 32
# Ramdon selection
# train_data = full_scan[(random.randint(0, n_h_emc-1), random.randint(0, n_cte_emc-1))][None]
# while len(train_data)<n_train_init:
#     temp = full_scan[(random.randint(0, n_h_emc-1), random.randint(0, n_cte_emc-1))]
#     if temp not in train_data:
#         train_data = np.insert(train_data, -1, temp, axis=0)

# Specific selection
x1, x2, x3, x4, x5 = np.mgrid[0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j]
points = np.hstack((x1.reshape(32, 1), x2.reshape(32, 1), x3.reshape(32, 1), x4.reshape(32, 1), x5.reshape(32, 1))).transpose()
train_data = full_scan[points.astype(int)[0], points.astype(int)[1], points.astype(int)[2], points.astype(int)[3], points.astype(int)[4]]
print('Initial data loaded')

# %%
# Implement PyTorch and Bayesian loop
acqf_para_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
directory_list = ['001', '01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.double
design_domain = torch.as_tensor(full_scan[:, :, :, :, :, 0:n_variables], device=device, dtype=dtype)
design_domain_flattened = design_domain.flatten(0, -2)

for j in range(10, 11):

    # Create and change directory accordingly
    fr_path_c = 'Specific_selection\\min_max_32\\ucb_'+directory_list[j]+'\\'
    os.makedirs(f_path_c+fr_path_c)
    fr_path = 'Specific_selection/min_max_32/ucb_'+directory_list[j]+'/'
    print(f_path+fr_path)

    acqf_para = acqf_para_list[j]
    print('Acquisition parameter = ', acqf_para)
    train_x = torch.as_tensor(train_data[:, 0:-1], dtype=dtype, device=device)
    train_y_origin = torch.as_tensor(train_data[:, -1], dtype=dtype, device=device).unsqueeze(1)*1000
    train_y = train_y_origin**2
    best_observed_value = train_y.min().item()

    verbose = False

    # Bayesian loop
    Trials = 100
    for trial in range(1, Trials+1):

        #print(f"\nTrial {trial:>2} of {Trials} ", end="\n")  
        #print(f"Current best: {best_observed_value} ", end="\n")

        # fit the model
        model = FixedNoiseGP(train_x, -train_y, train_Yvar=torch.full_like(train_y, 0.000001)).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # for best_f, we use the best observed values as an approximation
        # EI = ExpectedImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        UCB = UpperConfidenceBound(model = model, beta = acqf_para)
        # PI = ProbabilityOfImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        
        # Evaluate acquisition function over the discrete domain of parameters
        acqf = UCB(design_domain.unsqueeze(n_variables))
        np.savetxt(f_path+fr_path+'Acqf_matrix_' + str(trial) + '.csv', acqf.detach().cpu().numpy().flatten(), delimiter=',')
        acqf_flattened = acqf.flatten()
        acqf_sorted = torch.argsort(acqf.flatten(), descending=True)
        acqf_max = acqf_sorted[0].unsqueeze(0)
        for j in range(1, 10):
            if acqf_flattened[acqf_max[0]] > acqf_flattened[acqf_sorted[j]]:
                break
            else:
                acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        # for j in range(1, 10):
        #     if acqf[acqf_max[0]//(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc), 
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)//(n_h_sub*n_h_emc*n_cte_emc), 
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)//(n_h_emc*n_cte_emc), 
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)//n_cte_emc, 
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)%n_cte_emc] > \
        #         acqf[acqf_sorted[j]//(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc),
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)//(n_h_sub*n_h_emc*n_cte_emc),
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)//(n_h_emc*n_cte_emc), 
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)//n_cte_emc,
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)%n_cte_emc]:
        #         break
        #     else:
        #         acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        candidate_id = acqf_max[torch.randint(len(acqf_max), size=(1, ))]
        candidate = design_domain_flattened[candidate_id]
        #print(candidate.tolist())
        new_y = result[candidate_id]
        #print(new_y)
        train_new_y_origin = torch.as_tensor([[new_y]], dtype=dtype, device=device)
        train_new_y = train_new_y_origin**2
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
    np.savetxt(f_path+fr_path+'Optimization_loop.csv', optim_result.cpu().numpy(), delimiter=',')
    print('Bayesian loop completed')
    fig1, ax1 = plt.subplots()
    y_plot = abs(train_y_origin.numpy())*1000
    l1 = ax1.plot(y_plot, marker='.')
    ax1.set_yscale('log')
    #ax1.set_ylim(10**-(3.7), 10**-(1.6))
    ax1.set_xlabel('Loop iteration', fontsize='large')
    ax1.set_ylabel('Absolute warpage (um)', fontsize='large')
    fig1.savefig(f_path+fr_path+'Warpage_reduction.jpg', dpi=600)


# %%
x = torch.tensor([0, 4])
x1, x2, x3, x4, x5 = torch.meshgrid(x, x, x, x, x)

# %%
