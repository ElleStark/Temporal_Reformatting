# Script to compute instantaneous strains along particle trajectories (generated in main_ParticleTrack.py)
# Elle Stark, June 2024
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging
import time
import cmasher as cmr

# Set up logging for convenient messages
logger = logging.getLogger('LagrangeStrainsPy')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warn
DEBUG = logger.debug

def add_newfield_to_particles(particle_data, new_data, field2=None):
    INFO("Begin function to add new fields")
    n_tsteps, n_features, n_particles = particle_data.shape
    DEBUG(f"number of particles={n_particles}, number of features={n_features}, number of timesteps={n_tsteps}")
    
    # Flatten and mask particle data
    particles_reorder = particle_data.transpose(1, 0, 2)
    particles_flat = particles_reorder.reshape(n_features, n_tsteps * n_particles)
    valid_mask = ~np.isnan(particles_flat[1, :]) & ~np.isnan(particles_flat[2, :])
    
    # Flattened array for timesteps
    timesteps = np.repeat(np.arange(n_tsteps), n_particles)

    # Filter to valid particles & timesteps
    valid_particles = particles_flat[:, valid_mask]
    DEBUG(f"Shape of valid particles array: {valid_particles.shape}")
    valid_timesteps = timesteps[valid_mask]
    DEBUG(f"Shape of valid timesteps array: {valid_timesteps.shape}")

    # Obtain x and y coordinates and extract new field values
    x_coords = valid_particles[1, :]
    y_coords = valid_particles[2, :]
    x_idx = (x_coords / 0.5 * 1000).astype(int)
    y_idx = ((y_coords + 0.211) / 0.422 * 845).astype(int)

    new_vals = np.full(particles_flat.shape[1], np.nan)
    new_vals[valid_mask] = new_data[valid_timesteps, x_idx, y_idx]
    new_vals = new_vals.reshape(n_tsteps, n_particles)
    # Concatenate to original data
    DEBUG("Concatenating new data to existing matrix")
    start = time.time()
    particle_data_expanded = np.concatenate((particle_data, new_vals[:, np.newaxis, :]), axis=1)
    DEBUG(f"Concatenation completed in {round(time.time()-start, 2)} sec.")
    
    if field2 is None:
        INFO("1 New field added to particle data.")
    
    else:
        new_vals2 = np.full(particles_flat.shape[0], np.nan)
        new_vals2[valid_mask] = field2[valid_timesteps, x_coords, y_coords]
        new_vals2 = new_vals2.reshape(n_tsteps, n_particles)
        DEBUG("Concatenating new data field 2 to existing matrix")
        start = time.time()
        particle_data_expanded = np.concatenate((particle_data_expanded, new_vals2[:, np.newaxis, :]), axis=1)
        DEBUG(f"Concatenation 2 completed in {round(time.time()-start, 2)} sec.")
        INFO("2 new fields added to particle data.")

    return particle_data_expanded


def main():
    
    ########## PART 1: Extract particle tracking data and add strain and acceleration data, if needed

    # # Load particle tracking data; original dimensions (time, features, particles) = (3000, 3, 60000)
    # particle_matrix = np.load('ignore/ParticleTrackingData/particleTracking_n20_fullsim_D1.5000000000000002e-05_nanUpstream.npy')

    # # Obtain the instantaneous max principal strain at all time steps (HDF5 file)
    # f_name = 'E:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
    # with h5py.File(f_name, 'r') as f:
    #     # Numeric grids; original dimensions (x, y) = (1001, 846)
    #     x_grid = f.get('Model Metadata/xGrid')[:]
    #     y_grid = f.get('Model Metadata/yGrid')[:]

    #     # Max Principal Strain & U velocity; original dimensions (time, columns, rows) = (3000, 1001, 846)
    #     strain_data = f.get('Flow Data/Strains/max_p_strains')[:]
    #     u_data = f.get('Flow Data/u')[:-1, :, :]  # Remove final(?) timestep to match particle tracking & strain data

    # # Compute streamwise acceleration at all timesteps
    # accel_x = np.gradient(u_data, axis=0)

    # # For each x, y, t location listed in particle tracking data matrices, retrieve the associated strain and acceleration at that time step
    # particles_w_strain = add_newfield_to_particles(particle_matrix, strain_data)
    # # Data matrix is now: release time, x, y, strain, & acceleration at each time step
    # file_name = 'ignore/ParticleTrackingData/ParticleStrains_sim1_n20_t60_D1.5v5.npy'
    # INFO(f"Saving expanded particle data to {file_name}.")
    # np.save(file_name, particles_w_strain)
    # INFO("Save complete.")

    ################ END PART 1 ###################

    # Load expanded particle tracking data if already computed
    file_name = 'ignore/ParticleTrackingData/ParticleStrains_sim1_n20_t60_D1.5v5.npy'
    particles_w_strain = np.load(file_name)

    # PART 2: plotting and analyzing strain & acceleration along Lagrangian trajectories

    # QC: spatial plot of strain vals at a few times
    # plot_times = [100, 500, 1000, 2999]
    # for t in plot_times:
    #     plot_data = particles_w_strain[t, :, :]
    #     plt.scatter(plot_data[1, :], plot_data[2, :], c=plot_data[3, :], s=100)
    #     plt.colorbar()
    #     plt.xlim(0, 0.5)
    #     plt.ylim(-0.211, 0.211)
    #     plt.show()

    # PLOT: spatial plot of strain along trajectories for all time for particles 1-20
    fig, ax = plt.subplots()
    startidx = 2100
    endidx = 2120
    for p in range(startidx, endidx):
        plt.scatter(particles_w_strain[:, 1, p], particles_w_strain[:, 2, p], c=particles_w_strain[:, 3, p], cmap=cmr.ember, 
                    norm=colors.LogNorm(), s=25, alpha=0.5)
    plt.colorbar()
    plt.xlim(0, 0.5)
    plt.ylim(-0.211, 0.211)
    plt.title(f'Strain along trajectories, release time {round(particles_w_strain[0, 0, startidx], 2)} s')
    plt.show()

    # PLOT: many-line plot of strain as f(t) with 0 as release time
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    for p in range(startidx, endidx):
        plot_data = particles_w_strain[:, 3, p]
        plot_data = plot_data[~np.isnan(plot_data)]
        plt.plot(plot_data)
    plt.ylim(0, 15)
    plt.xlim(0, 250)
    plt.ylabel('max principal strain')
    plt.xlabel('timesteps from release')
    plt.show()

    # PLOT: ILS line plots of slice at x=0, x=0.025, x=0.05 m
    # file_path = 'C:/Users/elles/Documents/CU_Boulder/Fluids_Research/FisherPlume_plots/flow_stats_plots/ignore/ILS_ustreamwise.npy'
    # ILS_data = np.load(file_path)  # original dimensions: (y, x) = (846, 1001)
    # ILS_subset = ILS_data[123:724, 0:101]

    # # spatial average: time-averaged ILS for x=0 to 0.05 m, y=-0.15 to y=0.15 m
    # INFO(f'ILS spatial avg, x range [0, 0.05] m, y range [-0.15, 0.15] m: {round(np.mean(ILS_subset), 4)} m.')

    # xidxs = [0, 50, 100]
    # plt.close()
    # fig, ax = plt.subplots()
    # for xidx in xidxs:
    #     plt.plot(ILS_subset[:, xidx], label=(f'x={xidx*0.0005} m'))
    # plt.ylabel('ILS (m)')
    # plt.xlabel('y location (idx)')
    # plt.legend()
    # plt.show()        

    # PLOT: many-line plot of acceleration as f(t) with 0 as release time

    # Think about particle PAIRS: xxxx as f(diff in release times)

if __name__=='__main__':
    main()

