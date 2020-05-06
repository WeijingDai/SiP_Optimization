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
n_h_emc = 21
n_cte_emc = 21
n_variables = 2
#f_path = 'C:/Users/weijing/Documents/Nutstore/Abaqus_warpage_result/2_variables_result/'
with open('./full_scan_2d.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',').reshape(n_h_emc, n_cte_emc, n_variables+1)
n_train_init = 3

# Ramdon selection
train_data = full_scan[(random.randint(0, n_h_emc-1), random.randint(0, n_cte_emc-1))][None]
while len(train_data)<n_train_init:
    temp = full_scan[(random.randint(0, n_h_emc-1), random.randint(0, n_cte_emc-1))]
    if temp not in train_data:
        train_data = np.insert(train_data, -1, temp, axis=0)

# Specific selection
# train_data = full_scan[(0, 20), (20, 0)]

cte_emc, h_emc = np.meshgrid(full_scan[0, :, 1]/10, full_scan[:, 0, 0])
fig, ax = plt.subplots()
cs = ax.contourf(cte_emc, h_emc, abs(full_scan[:, :, -1])*1000, levels = np.power(10, np.linspace(-1.5, 1.5, 31)), 
                locator=ticker.LogLocator(), cmap=cm.PuBu)
cbar1 = fig.colorbar(cs)
cbar1.set_ticks([0.01, 0.1, 1., 10., 100])
cbar1.set_label('Absolute warpage (um)')
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC CTE (ppm)')
ax.set_ylabel('EMC thickness (um)')
ax.scatter(train_data[:, 1]/10, train_data[:, 0], marker='.', c='green', s=100)
ax.set_xlim(7.95, 12.05)
ax.set_ylim(545, 955)
print('Initial data loaded')

# %%
# Implement PyTorch and Bayesian loop
acqf_para_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
directory_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
for j in range(len(acqf_para_list)):

    # Create and change directory accordingly
    fr_path = 'Specific_selection/Right_bottom_two_diagonal/ucb_'+directory_list[j]+'/'
    print(f_path+fr_path)

    acqf_para = acqf_para_list[j]
    print('Acquisition parameter = ', acqf_para)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    dtype = torch.double
    design_domain = torch.as_tensor(full_scan[:, :, 0:n_variables], device=device, dtype=dtype)
    train_x = torch.as_tensor(train_data[:, 0:-1], dtype=dtype, device=device)
    train_y_origin = torch.as_tensor(train_data[:, -1], dtype=dtype, device=device).unsqueeze(1)
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
        acqf = UCB(design_domain.unsqueeze(2))
        np.savetxt(f_path+fr_path+'Acqf_matrix_' + str(trial) + '.csv', acqf.detach().cpu().numpy(), delimiter=',')
        acqf_sorted = torch.argsort(acqf.flatten(), descending=True)
        acqf_max = acqf_sorted[0].unsqueeze(0)
        for j in range(1, 10):
            if acqf[acqf_max[0]//n_h_emc, acqf_max[0]%n_cte_emc] > acqf[acqf_sorted[j]//n_h_emc, acqf_sorted[j]%n_cte_emc]:
                break
            else:
                acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        candidate_id = acqf_max[torch.randint(len(acqf_max), size=(1, ))]
        candidate = design_domain[candidate_id//n_h_emc, candidate_id%n_cte_emc]
        #print(candidate.tolist())
        new_y = full_scan[candidate_id//n_h_emc, candidate_id%n_cte_emc, -1]
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

    fig, ax = plt.subplots()
    cs = ax.contourf(cte_emc, h_emc, abs(full_scan[:, :, -1])*1000, levels = np.power(10, np.linspace(-1.5, 1.5, 31)), 
                    locator=ticker.LogLocator(), cmap=cm.PuBu)
    cbar1 = fig.colorbar(cs)
    cbar1.set_ticks([0.01, 0.1, 1., 10., 100])
    cbar1.set_label('Absolute warpage (um)')
    ax.set_title('Absolute warpage mapping (um)')
    ax.set_xlabel('EMC CTE (ppm)')
    ax.set_ylabel('EMC thickness (um)')
    ax.scatter(train_data[:, 1]/10, train_data[:, 0], marker='.', c='green')
    x1_plot = train_x.numpy()[10:, 1]/10
    x2_plot = train_x.numpy()[10:, 0]
    c_plot = np.arange(1, len(x1_plot)+1)
    s1 = ax.scatter(x1_plot, x2_plot, marker ='*', c=c_plot, cmap=cm.plasma, alpha=0.5, s=c_plot/2)
    cbar2 = fig.colorbar(s1, orientation="horizontal")
    cbar2.set_ticks([1, 20, 40, 60, 80, 100])
    cbar2.set_label('Iteration')
    ax.set_xlim(7.95, 12.05)
    ax.set_ylim(545, 955)
    plt.show()
    fig.savefig(f_path+fr_path+'Optimization_loop.jpg', dpi=600)

    fig1, ax1 = plt.subplots()
    y_plot = abs(train_y_origin.numpy())*1000
    l1 = ax1.plot(y_plot, marker='.')
    ax1.set_yscale('log')
    #ax1.set_ylim(10**-(3.7), 10**-(1.6))
    ax1.set_xlabel('Loop iteration', fontsize='large')
    ax1.set_ylabel('Absolute warpage (um)', fontsize='large')
    fig1.savefig(f_path+fr_path+'Warpage_reduction.jpg', dpi=600)

## %%
# def heatmap(data, row_labels, col_labels, row_name, col_name, fig_name, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.

#     Parameters
#     ----------
#     data
#         A 2D numpy array of shape (N, M).
#     row_labels
#         A list or array of length N with the labels for the rows.
#     col_labels
#         A list or array of length M with the labels for the columns.
#     ax
#         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
#         not provided, use current axes or create a new one.  Optional.
#     cbar_kw
#         A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
#     cbarlabel
#         The label for the colorbar.  Optional.
#     **kwargs
#         All other arguments are forwarded to `imshow`.
#     """

#     if not ax:
#         fig, ax = plt.subplots()

#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

#     # We want to show all ticks...
#     ax.set_xticks(np.arange(data.shape[1]))
#     ax.set_yticks(np.arange(data.shape[0]))
#     # ... and label them with the respective list entries.
#     ax.set_xticklabels(col_labels)
#     ax.set_yticklabels(row_labels)

#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=False, bottom=True,
#                    labeltop=False, labelbottom=True)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
#              rotation_mode="anchor")

#     # Turn spines off and create white grid.
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)

#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", top=False, left=False, bottom=False)
#     ax.set_title(fig_name)
#     ax.set_xlabel(row_name)
#     ax.set_ylabel(col_name)
#     return fig, im, cbar

## %%
# fig, im, cbar = heatmap(abs(full_scan[:, :, -1]), x2_ticks, x1_ticks, 'EMC CTE (ppm)', 'EMC thickness (um)', 'Absolute warpage mapping (um)',
#                    cmap="PuBu_r", cbarlabel="Abasolute warpage (um)")

# # Create and change directory accordingly
# fig.savefig(f_path+'/ucb_1/1/h_emc_vs_cte_emc_mapping_pixel.jpg', dpi=500, bbox_inches = 'tight')
## %%


# %%
