# Particle Swarm Optimization
#%%
#%% 
import os
import numpy as np
import random

# %%
# Load data
variable_num = 5
n_h_sub = 5
n_h_adhesive = 5
n_h_die = 5
n_cte_emc = 5
n_h_emc = 5
with open('./full_scan_5d.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',')
full_scan[:, 0] = (full_scan[:, 0]-200)/30  # h_die
full_scan[:, 1] = (full_scan[:, 1]-20)/5    # h_adhesive
full_scan[:, 2] = (full_scan[:, 2]-200)/25  # h_sub
full_scan[:, 3] = (full_scan[:, 3]-550)/100 # h_emc
full_scan[:, 4] = (full_scan[:, 4]-8)/1     # cte_emc
print('Initial data loaded')

pop_size = 10
rand_pick = full_scan[random.sample(range(0, len(result)), pop_size)]
init_pop = rand_pick[:, 0:-1]
init_res = abs(rand_pick[:, -1])

# %%
## Discrete PSO
global_best = init_res.min()
global_best_variable = init_pop[init_res.argmin()]
local_best = init_res.copy()
local_best_variables = init_pop.copy()
w = 0.5 # Momentum proportion
c1 = 1  # Local proportion
c2 = 1  # Global proportion
v = np.zeros((pop_size, variable_num)).astype(int)
current_variable = np.array(init_pop)
current_result = np.array(init_res)
iteration_num = 10
for j in range(iteration_num):
    result_list = []
    current_best = max(current_result)    
    if current_best > global_best:
        global_best = current_best
        global_best_variable = current_variable[current_result.argmin()]
    for i in range(pop_size):
        if current_result[i] > local_best[i]:
            local_best[i] = current_result[i]
            local_best_variables[i] =  current_variable[i]
        local_direction = local_best_variables[i] - current_variable[i] # Move towards local_best
        global_direction = global_best_variable - current_variable[i]   # Move towards global_best
        local_direction[local_direction > 1] = 1      # Move no more than one unit
        local_direction[local_direction < -1] = -1
        global_direction[global_direction > 1] = 1
        global_direction[global_direction < -1] = -1
        # Combine momentum term, local term and global term
        v[i] = v[i] * w + local_direction * c1 * int(random.uniform(0.99, 1.5)) + global_direction * c2 * int(random.uniform(0.99, 1.5))
        current_variable[i] += v[i]
        current_variable[i][current_variable[i] > 4] = 4 
        current_variable[i][current_variable[i] < 0] = 0
        temp_result = abs(full_scan[int(v[i, 2] + v[i, 1]*n_h_sub + v[i, 0]*n_h_sub*n_h_adhesive + \
            v[i, 4]*n_h_sub*n_h_adhesive*n_h_die + v[i, 3]*n_h_sub*n_h_adhesive*n_h_die*n_cte_emc)][-1])
        current_result[i] = np.array(temp_result)
        result_list.append(temp_result)

# %%
## Continuous PSO
# import copy
# global_best = max(i1)
# global_best_variable = init_pop[i1.index(global_best)]
# local_best = i1
# local_best_variables = copy.deepcopy(init_pop)
## PSO parameters
# w_ini = 0.9 
# w_end = 0.4
# c1 = 1 
# c2 = 1
# v = [0. for i in range(10)]
# current_variable = copy.deepcopy(init_pop)
# current_result = i1
# iteration_num = 10
# for j in range(iteration_num):
#     w = (w_ini - w_end) * (iteration_num - j)/iteration_num + w_end
#     current_best = max(current_result)    
#     if current_best > global_best:
#         global_best = current_best
#         global_best_variable = current_variable[current_result.index(current_best)]
#     for i in range(pop_size):
#         if current_result[i] > local_best[i]:
#             local_best[i] = current_result[i]
#             local_best_variables[i] =  current_variable[i]
#         v[i] = v[i] * w \
#                 + (local_best_variables[i] - current_variable[i]) * c1 * random.random() \
#                 + (global_best_variable - current_variable[i]) * c2 * random.random()
#         current_variable[i] += v[i] 
#     current_result = [random.random() for i in range(10)]
# %%

# %%
