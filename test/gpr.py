
# %% 

import numpy as np 
import matplotlib.pyplot as plt 

#%% input parameters 

sig1 = 1.0
sig2 = 0.03
len_ = 0.1

def kernal_fun(x, y, sig1=1.0, sig2=0.03, len_=0.1): 
    return sig1 * np.exp(-np.square((x-np.transpose(y))/2/len_)) + sig2 * (x == np.transpose(y))


# %% 

x = np.mat([-1.5, -1, -0.75, -0.4, -0.25, 0])
y = np.mat([-1.6, -1.2, -0.4, 0, 0.5, 0.9])

x = np.transpose(x)
y = np.transpose(y)

K = kernal_fun(x, x)

print(K)

# %% 

M = 10
N = 100

x_range = [-1.7, 0.3]

xs = np.mat(np.linspace(x_range[0], x_range[1], N))

invK = np.linalg.inv(K)

ratio_ = np.linspace(-1, 1, 10)
ys = np.mat(np.zeros(shape=(N, 11)))

for i in range(N): 
    K_ = kernal_fun(xs[0, i], x)
    K__ = kernal_fun(xs[0, i], xs[0, i])
    y_ = K_*invK*y
    var_ys_ = K__ - K_*invK*np.transpose(K_)
    ys[i, 0] = y_[0]
    for j in range(10): 
        ys[i, j+1] = y_[0] + 1.96*np.sqrt(var_ys_[0,0]) * ratio_[j] 

# %% 

xs_ = np.array(xs)
ys_ = np.array(ys)

plt.figure()
plt.plot(x[:, 0], y[:, 0], 'ro')
plt.plot(xs_[0, :], ys_[:, 0], 'b')
for j in range(10): 
    plt.plot(xs_[0, :], ys_[:, j+1], 'c') 
plt.show()





# %%
