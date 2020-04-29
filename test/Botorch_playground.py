# %%
# Import Botorch library
import torch
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
def forrester(x):
    return (6 * x - 2)**2 * torch.sin(12 * x - 4)

x = torch.linspace(0, 1, dtype=torch.double, device=torch.device('cpu'))
y = forrester(x)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, linewidth=3)
ax.set_title('Forrester function', fontsize=15)
ax.set_xlabel('x', fontsize = 15)
ax.set_ylabel('y', fontsize = 15)
ax.set_xlim(0,1)
plt.show()

# %%
def gen_init_data(data_num=2, noise_sd=0.1):
    # generate training data
    x_init = torch.rand(data_num)
    y_init_exact = -forrester(x_init)
    y_init = y_init_exact + noise_sd * torch.randn(data_num)
    return x_init, y_init, y_init_exact

# %%
x_init, y_init, y_init_exact = gen_init_data(2, 0.1)


# %%
gp_model = SingleTaskGP(train_x, train_y, covar_module=RBFKernel(ard_num_dims=train_x.shape[-1]), )