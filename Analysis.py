# %%
# Load library
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
import numpy as np

#%%
# h_emc = np.linspace(550, 950, 5, endpoint=True)
# h_adhesive = np.linspace(20, 40, 5, endpoint=True)
# h_die = np.linspace(200, 320, 5, endpoint=True)
# h_sub = np.linspace(200, 300, 5, endpoint=True)
# emc_cte = np.linspace(8, 12, 5, endpoint=True)

# full_scan = []

# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             for m in range(5):
#                 for n in range(5):
#                     with open(r'G:\Projects\Warpage_Botorch\result\ke_ch2_3d-mm-merge-h_die_' 
#                     + str(int(h_die[k])) +'-h_adhesive_'+ str(int(h_adhesive[m])) +'-h_sub_' + str(int(h_sub[n])) + '-h_emc_'+ str(int(h_emc[i])) +'-emc_cte_'+ str(int(emc_cte[j])) +'.csv') as f:
#                         temp = np.loadtxt(f, delimiter=',')[0]
#                         full_scan.append([h_die[k], h_adhesive[m], h_sub[n], h_emc[i], emc_cte[j], temp])
# np.savetxt('G:/Projects/Warpage_Botorch/5_variables/full_scan.csv', np.array(full_scan), delimiter=',')

# %%

# %%
# Load data
n_h_emc = 21
n_emc_cte = 21
n_h_sub = 5
n_h_adhesive = 5
n_h_die = 5
n_variables = 2
with open('C:/Users/weijing/Documents/Nutstore/Abaqus_warpage_result/2_variables_result/full_scan.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',').reshape(n_h_emc, n_emc_cte, n_variables+1)
obj_1 = abs(full_scan[:, :, -1])
h_emc, emc_cte = np.meshgrid(full_scan[:, 0, 0], full_scan[0, :, 1]/10)

# %%
fig, ax = plt.subplots()
cs = ax.contourf(h_emc, emc_cte, obj_1, levels = np.power(10, np.linspace(-5, -1, 21)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
cbar.set_ticks([0.00001, 0.0001, 0.001,0.01, 0.1])
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC thickness (um)')
ax.set_ylabel('EMC CTE (ppm)')
fig.savefig('C:/Users/weijing/Documents/Nutstore/Abaqus_warpage_result/2_variables_result/h_emc_vs_emc_cte.jpg', dpi=600)

# %%
with open(r'D:\ABAQUS\Warpage\ke_ch2\2_variables\EMC_h_cte_2d\init_data_cte_h_cte_subwarp.csv') as f:
    init_data = np.loadtxt(f, delimiter=',')[:,0:2]
init_data[:, 1] = init_data[:, 1]/10
# %%
with open(r'D:\ABAQUS\Warpage\ke_ch2\2_variables\EMC_h_cte_2d\ucb_10\Optimization_loop.csv') as f:
    ucb_data = np.loadtxt(f, delimiter=',')[10:,0:2]
ucb_data[:, 1] = ucb_data[:, 1]/10

with open(r'D:\ABAQUS\Warpage\ke_ch2\2_variables\EMC_h_cte_2d\ei\Optimization_loop.csv') as f:
    ei_data = np.loadtxt(f, delimiter=',')[10:,0:2]
ei_data[:, 1] = ei_data[:, 1]/10

with open(r'D:\ABAQUS\Warpage\ke_ch2\2_variables\EMC_h_cte_2d\pi\Optimization_loop.csv') as f:
    pi_data = np.loadtxt(f, delimiter=',')[10:,0:2]
pi_data[:, 1] = pi_data[:, 1]/10

# %%
ucb_x = ucb_data[:, 0]
ucb_y = ucb_data[:, 1]
fig, ax = plt.subplots()
cs = ax.contourf(x, y, z,  levels = np.power(10, np.linspace(-5, -1, 21)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
cbar.set_ticks([0.00001, 0.0001, 0.001,0.01, 0.1])
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC thickness (um)')
ax.set_ylabel('EMC CTE (ppm)')
ax.scatter(init_data[:, 0], init_data[:, 1], marker='*', c='orange')
ax.scatter(ucb_x, ucb_y, marker ='*', c='green')
ax.set_xlim(550, 950)
ax.set_ylim(8, 12)
plt.show()

# %%
ei_x = ei_data[:, 0]
ei_y = ei_data[:, 1]
fig, ax = plt.subplots()
cs = ax.contourf(x, y, z,  levels = np.power(10, np.linspace(-5, -1, 21)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
cbar.set_ticks([0.00001, 0.0001, 0.001,0.01, 0.1])
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC thickness (um)')
ax.set_ylabel('EMC CTE (ppm)')
ax.scatter(init_data[:, 0], init_data[:, 1], marker='*', c='orange')
ax.scatter(ei_x, ei_y, marker ='*', c='green')
ax.set_xlim(550, 950)
ax.set_ylim(8, 12)
plt.show()

# %%
pi_x = pi_data[:, 0]
pi_y = pi_data[:, 1]
fig, ax = plt.subplots()
cs = ax.contourf(x, y, z,  levels = np.power(10, np.linspace(-5, -1, 21)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
cbar.set_ticks([0.00001, 0.0001, 0.001,0.01, 0.1])
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC thickness (um)')
ax.set_ylabel('EMC CTE (ppm)')
ax.scatter(init_data[:, 0], init_data[:, 1], marker='*', c='orange')
ax.scatter(pi_x, pi_y, marker ='*', c='green')
ax.set_xlim(550, 950)
ax.set_ylim(8, 12)
plt.show()

# %%
