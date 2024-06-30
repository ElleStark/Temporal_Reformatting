# Script to compute instantaneous strains along particle trajectories (generated in main_ParticleTrack.py)
# Elle Stark, June 2024
import h5py
import numpy as np
import matplotlib.pyplot as plt

# def add_newfield_to_particles(particle_data, new_data):
#     n_particles, n_features, n_tsteps = particle_data.shape




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


    # Data matrix is now: release time, x, y, strain, & acceleration at each time step


    # PLOT: many-line plot of strain as f(t) with 0 as release time

    # PLOT: many-line plot of acceleration as f(t) with 0 as release time

    # Think about particle PAIRS: xxxx as f(diff in release times)

if __name__=='__main__':
    main()

