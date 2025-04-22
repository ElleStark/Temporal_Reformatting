# Utility functions for two-particle simulations for temporal reformatting study
# Elle Stark May 2024

import h5py
import matplotlib.colors 
import matplotlib.pyplot as plt
import numpy as np


# def plot_traj_snap(): 

# Read in H5 Concentration data for background plume
f_name = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
x_lims = slice(None, None)
y_lims = slice(None, None)
time_lims = slice(1153, 1154)

with h5py.File(f_name, 'r') as f:
        # Numeric grids
        xmesh_uv = f.get('Model Metadata/xGrid')[x_lims, y_lims].T
        ymesh_uv = f.get('Model Metadata/yGrid')[x_lims, y_lims].T  

        # Odor concentration field data
        odor_c = f.get('Odor Data/c')[time_lims, x_lims, y_lims].T

# Load full trajectory dataset - format is [frame, release_frame, x_pos, y_pos]
traj_data = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D5em5_RK4method_180to360s.npy')

# Subset to a few particles released at 2 different times
# Recall integral time scale ~0.15s (~8 frames) at release point
dt_frames = 5
dt = 0.02
# try different start times and trajectory durations for nice results
# option 1: 3 trajectories all simultaneously detected
# first_frame = 1023  
# duration = int(2.7 * 50)  # ~5 seconds to cross domain, times 50 Hz resolution
# option 2: 1 simultaneous detection & 2 far-flung trajectories
first_frame = 1030
duration = 123

n_particles = 5  # probably don't need all 20 particles released in each frame
start_particle1 = 2
start_particle = 15

select_trajs1 = traj_data[first_frame:(first_frame+duration), :, traj_data[0, 0, :]==(first_frame*dt)]
traj1_x = select_trajs1[:, 1, start_particle1:(start_particle1+n_particles)]  # find x vals for trajectories
traj1_y = select_trajs1[:, 2, start_particle1:(start_particle1+n_particles)]  # find y vals for trajectories
select_trajs1 = select_trajs1[:, :, start_particle1:(start_particle1+n_particles)]

select_trajs2 = traj_data[(first_frame + dt_frames):(first_frame+duration), :, traj_data[0, 0, :]==((first_frame+dt_frames)*dt)]
traj2_x = select_trajs2[:, 1, start_particle:(start_particle+n_particles)]
traj2_y = select_trajs2[:, 2, start_particle:(start_particle+n_particles)]
select_trajs2 = select_trajs2[:, :, start_particle:(start_particle+n_particles)]
alpha = 0.5

# release_times = np.unique(traj_data[0, 0, :])
# releases = {}

# for release in release_times:
#       end_frames = []
#       traj_data_r = traj_data[:, :, traj_data[0, 0, :]==release]
#       traj_data_offset = traj_data[:, :, traj_data[0, 0, :]==(release+dt_frames*dt)]
#       frames = range(np.shape(traj_data)[0])
#       for frame in frames:
#         conditions = (np.all(traj_data_r[frame, 1, :] > 0.25) & np.any(np.abs(traj_data_r[frame, 2, :]) < 0.005) & np.any(np.abs(traj_data_r[frame, 2, :]) > 0.05) & 
#                         np.any(np.abs(traj_data_offset[frame, 2, :]) < 0.005) & np.any(np.abs(traj_data_offset[frame, 2, :])))
#         if conditions:
#               end_frames.append(frame)

#       if len(end_frames)>0:
#             releases[str(release)] = end_frames

# print(releases)
# Plot trajectories as lines with point at start and end of each trajectory
# plt.pcolormesh(xmesh_uv, ymesh_uv, odor_c[:, :, 0], cmap='Greys', vmin=0, vmax=0.5)

# Create greyscale color map for background plume
c_list = ["#f9f9f9", "#a4a4a4"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", c_list)

plt.pcolormesh(xmesh_uv, ymesh_uv, odor_c[:, :, 0], cmap=cmap, vmin=0, vmax=0.2)
# plt.colorbar()
# for i in range(len(traj2_x[0, :])):
#     # plt.plot(traj1_x[:, i], traj1_y[:, i], color='#B85B51', alpha=alpha, linestyle='dashed')
#     plt.plot(traj2_x[:, i], traj2_y[:, i], label=f'traj2_{i}')
#     plt.plot(traj1_x[:, i], traj1_y[:, i], label=f'traj1_{i}')
#     # plt.plot(traj2_x[:, i], traj2_y[:, i], color='#588D9D', alpha=alpha, linestyle='dashed')

# # Option 1: 3 trajectories all simultaneously detected
# traj1_list = [4, 16, 7]
# traj2_list = [15, 4, 10]

# Option 2: 1 simultaneous detection & 2 far-flung trajectories
traj1_list = [0, 1, 3]
traj2_list = [0, 4, 3]

for i in range(3):
    plt.plot(traj1_x[:, traj1_list[i]], traj1_y[:, traj1_list[i]], color='#B85B51', alpha=alpha, linestyle='solid')
    plt.scatter(select_trajs1[duration-1, 1, traj1_list[i]], select_trajs1[duration-1, 2, traj1_list[i]], color='#B85B51', alpha=alpha)
    plt.plot(traj2_x[:, traj2_list[i]], traj2_y[:, traj2_list[i]], color='#588D9D', alpha=alpha, linestyle='solid')
    plt.scatter(select_trajs2[duration-dt_frames-1, 1, traj2_list[i]], select_trajs2[duration-dt_frames-1, 2, traj2_list[i]], color='#588D9D', alpha=alpha)

# different formatting for simultaneously detected particles
alpha = 0.7
idx1 = 2
idx2 = 2
plt.plot(traj1_x[:, idx1], traj1_y[:, idx1], color='#B85B51', alpha=alpha, linestyle='solid', linewidth=2.5)
plt.scatter(select_trajs1[duration-1, 1, idx1], select_trajs1[duration-1, 2, idx1], color='#B85B51', alpha=alpha)
plt.plot(traj2_x[:, idx2], traj2_y[:, idx2], color='#588D9D', alpha=alpha, linestyle='solid', linewidth=2.5)
plt.scatter(select_trajs2[duration-dt_frames-1, 1, idx2], select_trajs2[duration-dt_frames-1, 2, idx2], color='#588D9D', alpha=alpha)

# plt.scatter(select_trajs1[duration-1, 1, start_particle1:(start_particle1+n_particles)], select_trajs1[duration-1, 2, start_particle1:(start_particle1+n_particles)], color='#B85B51', alpha=alpha)
# plt.scatter(select_trajs2[duration-dt_frames-1, 1, start_particle:(start_particle+n_particles)], select_trajs2[duration-dt_frames-1, 2, start_particle:(start_particle+n_particles)], color='#588D9D', alpha=alpha)

plt.ylim(-0.3, 0.3)
plt.xlim(0, 0.750)
plt.axis("equal")
# plt.legend()

plt.savefig('ignore/plots/traj_snap_comps_simuldet_opt2.png', dpi=600)
plt.show()
