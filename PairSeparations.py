# Script to compute evolution of mean squared particle separations for different initial separations
# For connecting classical analyses (Richardson 1926, Batchelor 1952) to sensory-based frequency transformations by the flow field
# Elle Stark, Ecological Fluid Dynamics Lab, CU Boulder, September 2024

import h5py
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging
import cmasher as cmr

# Set up logging for convenient messages
logger = logging.getLogger('PairSepsPy')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warn
DEBUG = logger.debug


# Load particle tracking data; original dimensions (time, features, particles) = (9001, 3, 180000); data type = float32
particle_matrix = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_nanUpstream_0to180s_normal.npy')

########## COMPUTE PAIRED PARTICLE SEPARATIONS OVER TIME ##########

# Convert x and y to cm if needed
# particle_matrix[:, 1:3, :] = particle_matrix[:, 1:3, :] * 100
mean_u = 10  # cm/s
dt = 0.02
p_num = 20  # particles released per timestep

# Select times for plotting - for current log-log plotting setup, ALWAYS HAVE 0 AND FIRST TIMESTEP
t_list = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5])
t_list_idx = (t_list / dt).astype(int) 
n_tsteps = particle_matrix.shape[0] - t_list_idx[-1]

# Select initial separation times - each will be plotted on a separate line
d_release_list = [0, 1, 2, 5, 10, 20, 30, 40, 50]

r2_array = np.zeros((n_tsteps, p_num))
# D_LL_array = np.zeros((n_tsteps, 20))
r2_0 = np.zeros((n_tsteps, 20))
r2_list = np.zeros(len(t_list))
all_r2_list = np.zeros((len(d_release_list), len(r2_list)))
all_r0_list = np.zeros((len(d_release_list)))
n_particles_list = np.zeros((len(d_release_list)))
# all_r_list = np.zeros((len(d_release_list)))
# all_D_LL_list = np.zeros((len(d_release_list)))
all_idx = 0

for delta_release in d_release_list:
    idx = 0
    batchelor_t = ((delta_release*dt * mean_u)**2)
    n_particles = n_tsteps * 20
    for compute_t in t_list_idx:
        t=0
        for t in range(n_tsteps-1):
            t += 1
            t0 = int(round(particle_matrix[0, 0, t*20 + 20*delta_release] / dt, 0))

            # compute squared distance as delta x squared + delta y squared
            separations = np.sqrt((particle_matrix[t0 + compute_t, 1, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 1, (t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)])**2 + (particle_matrix[t0 + compute_t, 2, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 2, (t*20 + 20*delta_release):(t*20 + 20*delta_release +20)])**2)
            r2_array[t, :] = separations

            # if compute_t == t_list_idx[0]:
            #     ########## D_LL(t=0) COMPUTATION ##########
            #     # extract velocity field at each particle position
            #     u_vals = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_0to180s_udata.npy')[t0, :]
            #     v_vals = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_0to180s_vdata.npy')[t0, :]

            #     # find difference in velocities for each particle pair
            #     u_diffs = u_vals[(t*20-20):(t*20)] - u_vals[(t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]  
            #     v_diffs = v_vals[(t*20-20):(t*20)] - v_vals[(t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]

            #     # find and square the longitudinal component
            #     # store D_LL for all particle offsets
            #     for particle_idx in range(D_LL_array.shape[1]):
            #         D_LL_array[t, particle_idx] = (np.dot(np.array([x_diffs[particle_idx], y_diffs[particle_idx]]), np.array([u_diffs[particle_idx], v_diffs[particle_idx]])) / separations[particle_idx])**2
        
        if compute_t == t_list_idx[0]:
            r2_0[:] = r2_array[:]
        else:
            r2_0 = r2_0

        n_particles = np.min([n_particles, np.count_nonzero(~np.isnan(r2_array))])
        r2_list[idx] = np.nanmean((r2_array - r2_0)**2)
        # r2_list[idx] = np.nanmean(r2_array)**2
        # r_list[idx] = np.nanmean(r2_array)
        idx += 1
    all_r2_list[all_idx, :] = r2_list
    all_r0_list[all_idx] = np.nanmean(r2_0**2)
    n_particles_list[all_idx] = n_particles
    # all_r_list[all_idx] = np.nanmean(r_list)
    # all_D_LL_list[all_idx] = np.nanmean(D_LL_array)
    all_idx += 1


print(n_particles_list)

C2 = 2.13
# epsilon = np.load('ignore/inputs/viscous_dissipation_extendedSim.npy')  # 1201 x 1501 matrix of vals
# epsilon = 0.001015  # m^2/s^3, for simplicity start with domain average
# epsilon = 0.0025
epsilon = 0.006728  # m^2/s^3, first 0.05 m
epsilon_end = 0.000165  # m^2/s^3, last 0.05 m
nu = 1.5 * 10**(-5)  # kinematic viscosity, m/s
t_eta = 0.1216  # s, kolmogorov microscale in time, entire domain
g = 0.5  # Richardson constant
# t_eta = 0.0472  # first 0.05 m


# t2_array = t_list_10 ** 2
# t3_array =t_list_10 ** 3

fig, ax = plt.subplots(figsize=(8, 6))

# for i in range(len(d_release_list)-1):
#     i+=1
    # Normalize r squared according to Ouellette et al., 2006
    # normalized_r2 = all_r2_list[i, :]/ ((11/3*C2*(epsilon*all_r0_list[i])**(2/3))*t_eta**2)
    # Normalize r according to Tan & Ni 2022
#     t_zero = epsilon**(-1/3)*all_r0_list[i]**(2/3)
#     D_LL = all_D_LL_list[i]
#     normalized_r2 = all_r2_list[i, :]/ (t_zero**2 * D_LL)
#     plt.loglog(t_list[1:]/t_eta, normalized_r2[1:], label=f'p_sep: {round(d_release_list[i]*dt, 2)}')
# plt.loglog(t_list[1:]/t_zero, (t_list[1:]/t_zero)**2, color='black')

# plt.xlabel('t/t_microscale')
# plt.ylabel('<r^2> compensated per Ouellette et al.')
# plt.title('log-log scaling: initial separations 0 to 1 s')

c_list = cmr.take_cmap_colors('cmr.sapphire', int(len(d_release_list)+6))

for i in range(len(d_release_list)):
    # i+=1
    plt.loglog(t_list[1:], all_r2_list[i, 1:], label=f'{round(d_release_list[i]*dt, 2)}', color = c_list[i+6])
    # plt.loglog(t_list[:], g*epsilon*t_list[:]**(3) + 2*all_r0_list[i]**2 - 2*all_r_list[i]*all_r0_list[i], color='blue')
    # plt.loglog(t_list[:], g*epsilon*t_list[:]**(3) + 2*all_r0_list[i]**2, color='red')

# Exponential fit at early times for particle separations for pairs with 0 initial separation 
plt.loglog(t_list[1:8], 0.4*(all_r0_list[0]*np.exp(t_list[1:8]/0.07) - all_r0_list[0]), 'r--', linewidth=2)

# t^3 fit for later times for particle separations for pairs with 0 initial separation, compensated by t^3
# plt.loglog(t_list[1:], all_r2_list[0, 1:])
# plt.scatter(t_list[1], g*epsilon* t_list[1]**3)
# plt.scatter(t_list[-1], g*0.0001*t_list[-1]**3)

# t^3 curve for later times
# plt.loglog(t_list[4:], g*epsilon*t_list[4:]**(3/2), color='black')
epsilon_list = 0.007*t_list[1:]**(-3/2)
plt.loglog(t_list[4:], g*epsilon_list[3:]*t_list[4:]**3, '--', color='#80328c', linewidth=2)

# # Expected fit for early times but large separations:
plt.loglog(t_list[1:8], 0.007*t_list[1:8]**(2), 'k--', linewidth=2)
# Better fit for our data:
# plt.loglog(t_list[1:8], 0.8*epsilon*t_list[1:8]**(29/16), color='black')

# Expected linear regime at end???
# plt.loglog(t_list[-9:], 5*epsilon_list[-9:]*t_list[-9:]**(3/2), color='green')


plt.xlabel('t (s)')
plt.ylabel('<|r-r0|>^2')
plt.title('log-log scaling: initial separations 0 to 1 s')

plt.legend()
plt.savefig('ignore/plots/pairseps_woverlay_0to1_loglog.png', dpi=300)
plt.show()

