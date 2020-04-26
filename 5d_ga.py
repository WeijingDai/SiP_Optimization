#%% 
import os
import numpy as np
import random

# Genetic algorithm
def ga_selection(pop_size = 10, population = [], variable_num = 5, fitness = [], mutation_ratio = 0.1):
    # Parents selection (max k)
    parent_size = pop_size//2 # Keep half of the initial population
    temp_1 = population.copy()
    temp_2 = fitness.copy()
    parent = []
    for seq in range(parent_size):
        min_id = temp_2.index(min(temp_2))
        parent.append(temp_1[min_id])
        del(temp_2[min_id])
        del(temp_1[min_id])

    # Crossover
    offspring = parent.copy()
    mating_seq = random.sample(range(0, pop_size), parent_size) # randomly select half of pop_size to crossover
    for seq in range(parent_size):
        mating_rule = np.random.randint(2, size=variable_num) # 1 -> keep, 0 -> swap
        child = parent[seq] * mating_rule + population[mating_seq[seq]] * abs(mating_rule - 1)
        offspring.append(child.tolist())

    # Mutation
    mutation_num = int(len(offspring) * len(offspring[0]) * mutation_ratio) # Randomly select num of position to mutate
    mutation_pos = []
    while len(mutation_pos) < mutation_num:
        temp = [random.randint(0, pop_size-1), random.randint(0, variable_num-1)]
        if temp not in mutation_pos:
            mutation_pos.append(temp)
    for pos in mutation_pos:
        mute_x = random.randint(0, variable_num-1)
        while mute_x == offspring[pos[0]][pos[1]]:
            mute_x = random.randint(0, variable_num-1)
        else:
            offspring[pos[0]][pos[1]] = mute_x
    return offspring

print('Library loaded')

# %%
# Load data
n_variables = 5
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
init_pop = rand_pick[:, 0:-1].tolist()
init_fit = abs(rand_pick[:, -1]).tolist()
# %%
generation_num = 10
for generation in range(generation_num):
    offspring = ga_selection(pop_size = pop_size, population = init_pop, variable_num = n_variables, fitness = init_fit, mutation_ratio = 0.1)
    offspring_fit = []
    for child in offspring:
        offspring_fit.append(abs(full_scan[int(child[2] + child[1]*n_h_sub + child[0]*n_h_sub*n_h_adhesive + \
            child[4]*n_h_sub*n_h_adhesive*n_h_die + child[3]*n_h_sub*n_h_adhesive*n_h_die*n_cte_emc)][-1]))
    init_pop = offspring
    init_fit = offspring_fit
    #np.savetxt('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/Generation_' + str(generation + 1) + '.csv', \
     #   np.hstack((np.array(init_pop), np.array(init_fit)[:, None])), delimiter=',')


# %%
parent_size = pop_size//2
temp = init_pop.copy()
parent = []
for seq in range(parent_size):
    min_id = init_fit.index(min(init_fit))
    parent.append(temp[min_id])
    del(init_fit[min_id])
    del(temp[min_id])

# %%
offspring = parent.copy()
mating_seq = random.sample(range(0, pop_size), parent_size)
for seq in range(parent_size):
    mating_rule = np.random.randint(2, size=5)
    child = parent[seq] * mating_rule + init_pop[mating_seq[seq]] * abs(mating_rule - 1)
    offspring.append(child.tolist())

# %%
