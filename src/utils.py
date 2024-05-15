# Utility functions for two-particle simulations for temporal reformatting study
# Elle Stark May 2024

import h5py
import matplotlib.colors 
import matplotlib.pyplot as plt
import numpy as np


# def plot_traj_snap(): 

# Read in H5 Concentration data for background plume
f_name = 'D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
x_lims = slice(None, None)
y_lims = slice(None, None)
time_lims = slice(1158, 1159)

with h5py.File(f_name, 'r') as f:
        # Numeric grids
        xmesh_uv = f.get('Model Metadata/xGrid')[x_lims, y_lims].T
        ymesh_uv = f.get('Model Metadata/yGrid')[x_lims, y_lims].T  

        # Odor concentration field data
        odor_c = f.get('Odor Data/c1a')[time_lims, x_lims, y_lims].T

# Load full trajectory dataset - format is [frame, release_frame, x_pos, y_pos]
traj_data = np.load('ignore/tests/particleTracking_n20_fullsim.npy')

# Subset to a few particles released at 2 different times
# Recall integral time scale ~0.15s (~8 frames) at release point
dt_frames = 5
dt = 0.02
# try different start times and trajectory durations for nice results
first_frame = 1023  
duration = int(2.7 * 50)  # ~5 seconds to cross domain, times 50 Hz resolution
n_particles = 20  # probably don't need all 20 particles released in each frame
start_particle1 = 0
start_particle = 0

select_trajs1 = traj_data[first_frame:(first_frame+duration), :, traj_data[0, 0, :]==(first_frame*dt)]
traj1_x = select_trajs1[:, 1, start_particle1:(start_particle1+n_particles)]  # find x vals for trajectories
traj1_y = select_trajs1[:, 2, start_particle1:(start_particle1+n_particles)]  # find y vals for trajectories

select_trajs2 = traj_data[(first_frame + dt_frames):(first_frame+duration), :, traj_data[0, 0, :]==((first_frame+dt_frames)*dt)]
traj2_x = select_trajs2[:, 1, start_particle:(start_particle+n_particles)]
traj2_y = select_trajs2[:, 2, start_particle:(start_particle+n_particles)]
alpha = 0.6

# Plot trajectories as lines with point at start and end of each trajectory
# plt.pcolormesh(xmesh_uv, ymesh_uv, odor_c[:, :, 0], cmap='Greys', vmin=0, vmax=0.5)

# Create greyscale color map for background plume
c_list = ["#f9f9f9", "#a4a4a4"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", c_list)

plt.pcolormesh(xmesh_uv, ymesh_uv, odor_c[:, :, 0], cmap=cmap, vmin=0, vmax=0.2)
# plt.colorbar()
# for i in range(len(traj2_x[0, :])):
    # plt.plot(traj1_x[:, i], traj1_y[:, i], color='#B85B51', alpha=alpha, linestyle='dashed')
    # plt.plot(traj2_x[:, i], traj2_y[:, i], label=f'p_{i}')
    # plt.plot(traj2_x[:, i], traj2_y[:, i], color='#588D9D', alpha=alpha, linestyle='dashed')

traj1_list = [4, 16, 7]
traj2_list = [15, 4, 10]

for i in range(3):
    plt.plot(traj1_x[:, traj1_list[i]], traj1_y[:, traj1_list[i]], color='#B85B51', alpha=alpha, linestyle='dashed')
    plt.scatter(select_trajs1[duration-1, 1, traj1_list[i]], select_trajs1[duration-1, 2, traj1_list[i]], color='#B85B51', alpha=alpha)
    plt.plot(traj2_x[:, traj2_list[i]], traj2_y[:, traj2_list[i]], color='#588D9D', alpha=alpha, linestyle='dashed')
    plt.scatter(select_trajs2[duration-dt_frames-1, 1, traj2_list[i]], select_trajs2[duration-dt_frames-1, 2, traj2_list[i]], color='#588D9D', alpha=alpha)

# plt.scatter(select_trajs1[duration-1, 1, start_particle1:(start_particle1+n_particles)], select_trajs1[duration-1, 2, start_particle1:(start_particle1+n_particles)], color='#B85B51', alpha=alpha)
# plt.scatter(select_trajs2[duration-dt_frames-1, 1, start_particle:(start_particle+n_particles)], select_trajs2[duration-dt_frames-1, 2, start_particle:(start_particle+n_particles)], color='#588D9D', alpha=alpha)

plt.ylim(-0.211, 0.211)
plt.xlim(0, 0.50)
plt.axis("equal")
# plt.legend()

plt.savefig('ignore/plots/traj_snap_JVposter_simuldet.png', dpi=600)
plt.show()
