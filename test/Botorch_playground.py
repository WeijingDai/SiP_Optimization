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
    x_init = torch.rand(data_num, dtype=torch.double, device=torch.device('cpu')).unsqueeze(-1)
    y_init_exact = -forrester(x_init)
    y_init = y_init_exact + noise_sd * torch.randn(data_num,dtype=torch.double, device=torch.device('cpu')).unsqueeze(-1)
    return x_init, y_init, y_init_exact

# %%
x_init, y_init, y_init_exact = gen_init_data(10, 1)


# %%
gp_model_1 = SingleTaskGP(x_init, y_init)
gp_model_2 = SingleTaskGP(x_init, y_init, )
gp_model_3 = FixedNoiseGP(x_init, y_init, train_Yvar=torch.full_like(y_init, 1))

mll_1 = ExactMarginalLogLikelihood(gp_model_1.likelihood, gp_model_1)
mll_2 = ExactMarginalLogLikelihood(gp_model_2.likelihood, gp_model_2)
mll_3 = ExactMarginalLogLikelihood(gp_model_3.likelihood, gp_model_3)

fit_gpytorch_model(mll_1)
fit_gpytorch_model(mll_2)
fit_gpytorch_model(mll_3)
# %%
p = gp_model_3.posterior(x)
lower, upper = p.mvn.confidence_region()
fig, ax = plt.subplots()
ax.plot(x, p.mean.detach(), 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(x.flatten(), lower.detach(), upper.detach(), alpha=0.5)
ax.plot(x, -y)
ax.plot(x_init, y_init, 'o')


# %%
