# Script to compute instantaneous strains along particle trajectories (generated in main_ParticleTrack.py)
# Elle Stark, June 2024
import h5py
import numpy as np
import matplotlib.pyplot as plt

def add_newfield_to_particles(particle_data, new_data, field2=None):
    n_particles, n_features, n_tsteps = particle_data.shape

    # Empty array to hold new values
    # new_vals = np.full((n_particles, n_tsteps), np.nan)
    
    # Flatten and mask particle data
    particles_flat = particle_data.reshape(-1, n_features)
    valid_mask = ~np.isnan(particles_flat[:, 1]) & ~np.isnan(particles_flat[:, 2])
    
    # Flattened array for timesteps
    timesteps = np.repeat(np.arange(n_tsteps), n_particles)

    # Filter to valid particles & timesteps
    valid_particles = particles_flat[valid_mask]
    valid_timesteps = timesteps[valid_mask]

    # Obtain x and y coordinates and extract new field values
    x_coords = valid_particles[:, 1].astype(int)
    y_coords = valid_particles[:, 2].astype(int)

    new_vals = np.full(particles_flat.shape[0], np.nan)
    new_vals[valid_mask] = new_data[valid_timesteps, x_coords, y_coords]
    new_vals = new_vals.reshape(n_tsteps, n_particles)
    # Concatenate to original data
    particle_data_expanded = np.concatenate((particle_data, new_vals[:, np.newaxis, :]), axis=1)
    
    if field2 != None:
        new_vals2 = np.full(particles_flat.shape[0], np.nan)
        new_vals2[valid_mask] = field2[valid_timesteps, x_coords, y_coords]
        new_vals2 = new_vals2.reshape(n_tsteps, n_particles)
        particle_data_expanded = np.concatenate((particle_data_expanded, new_vals2[:, np.newaxis, :]), axis=1)

    return particle_data_expanded


def main():
    # Load particle tracking data; original dimensions (time, features, particles) = (3000, 3, 60000)
    particle_matrix = np.load('ignore/ParticleTrackingData/particleTracking_n20_fullsim_D1.5000000000000002e-05_nanUpstream.npy')

    # Obtain the instantaneous max principal strain at all time steps (HDF5 file)
    f_name = 'E:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
    with h5py.File(f_name, 'r') as f:
        # Numeric grids; original dimensions (x, y) = (1001, 846)
        x_grid = f.get('Model Metadata/xGrid')[:]
        y_grid = f.get('Model Metadata/yGrid')[:]

        # Max Principal Strain & U velocity; original dimensions (time, columns, rows) = (3000, 1001, 846)
        strain_data = f.get('Flow Data/Strains/max_p_strains')[:]
        u_data = f.get('Flow Data/u')[:-1, :, :]  # Remove final(?) timestep to match particle tracking & strain data

    # Compute streamwise acceleration at all timesteps
    accel_x = np.gradient(u_data, axis=0)

    # For each x, y, t location listed in particle tracking data matrices, retrieve the associated strain and acceleration at that time step
    particles_w_strain_acc = add_newfield_to_particles(particle_matrix, strain_data, field2=accel_x)
    # Data matrix is now: release time, x, y, strain, & acceleration at each time step
    np.save('ignore/ParticleTrackingData/ParticleStrainsAccel_sim1_n20_t60_D1.5.npy')

    # PLOT: many-line plot of strain as f(t) with 0 as release time
    # QC: spatial plot of strain vals at a few times
    plot_times = [100, 500, 1000, 3000]
    for t in plot_times:
        plot_data = particles_w_strain_acc[t, :, :]
        plt.scatter(plot_data[0, :], plot_data[1, :], c=plot_data[2, :], s=500)
        plt.show()

    # PLOT: many-line plot of acceleration as f(t) with 0 as release time

    # Think about particle PAIRS: xxxx as f(diff in release times)

if __name__=='__main__':
    main()

