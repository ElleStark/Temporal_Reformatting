# Utility functions for two-particle simulations for temporal reformatting study
# Elle Stark May 2024

import matplotlib.pyplot as plt
import numpy as np


# def plot_traj_snap():



#     

# Load full trajectory dataset - format is [frame, release_frame, x_pos, y_pos]
traj_data = np.load('ignore/tests/particleTracking_n20_fullsim.npy')

# Subset to a few particles released at 2 different times
# Recall integral time scale ~0.15s (~8 frames) at release point
dt_frames = 40
dt = 0.02
# try different start times and trajectory durations for nice results
first_frame = 102  
duration = int(2.5 * 50)  # ~5 seconds to cross domain, times 50 Hz resolution
n_particles = 5  # probably don't need all 20 particles released in each frame

select_trajs1 = traj_data[first_frame:(first_frame+duration), :, traj_data[0, 0, :]==(first_frame*dt)]
traj1_x = select_trajs1[:, 1, :n_particles]  # find x vals for trajectories
traj1_y = select_trajs1[:, 2, :n_particles]  # find y vals for trajectories

select_trajs2 = traj_data[(first_frame + dt_frames):(first_frame+duration+dt_frames), :, traj_data[0, 0, :]==((first_frame+dt_frames)*dt)]
traj2_x = select_trajs2[:, 1, :n_particles]
traj2_y = select_trajs2[:, 2, :n_particles]

# Plot trajectories as lines with point at start and end of each trajectory
for i in range(len(traj1_x[0, :])):
    plt.plot(traj1_x[:, i], traj1_y[:, i], color='#B85B51')
    plt.plot(traj2_x[:, i], traj2_y[:, i], color='#588D9D')
plt.scatter(select_trajs1[0, 1, :n_particles], select_trajs1[0, 2, :n_particles], color='#B85B51')
plt.scatter(select_trajs1[duration-1, 1, :n_particles], select_trajs1[duration-1, 2, :n_particles], color='#B85B51')
plt.scatter(select_trajs2[0, 1, :n_particles], select_trajs2[0, 2, :n_particles], color='#588D9D')
plt.scatter(select_trajs2[duration-1, 1, :n_particles], select_trajs2[duration-1, 2, :n_particles], color='#588D9D')

plt.ylim(-0.211, 0.211)
plt.xlim(0, 0.50)
plt.show()

plt.savefig('ignore/plots/traj_snap_JVposter.png', dpi=600)
