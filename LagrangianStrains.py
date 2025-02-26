# Script to compute instantaneous strains along particle trajectories (generated in main_ParticleTrack.py)
# Elle Stark, June 2024
import h5py
from itertools import combinations
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging
import pandas as pd
import pickle
from scipy.optimize import curve_fit
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

# def add_newfield_to_particles(particle_data, new_data, field2=None):
def add_newfield_to_particles(particle_data, field2=None):
    INFO("Begin function to add new fields")
    n_tsteps, n_features, n_particles = particle_data.shape
    DEBUG(f"number of particles={n_particles}, number of features={n_features}, number of timesteps={n_tsteps}")
    
    # Flatten and mask particle data
    valid_particles = particle_data.transpose(1, 0, 2)
    valid_particles = valid_particles.reshape(n_features, n_tsteps * n_particles)
    valid_mask = ~np.isnan(valid_particles[1, :]) & ~np.isnan(valid_particles[2, :])
    
    # Flattened array for timesteps
    valid_timesteps = np.repeat(np.arange(n_tsteps), n_particles)

    # Filter to valid particles & timesteps
    valid_particles = valid_particles[:, valid_mask]
    DEBUG(f"Shape of valid particles array: {valid_particles.shape}")
    valid_timesteps = valid_timesteps[valid_mask]
    DEBUG(f"Shape of valid timesteps array: {valid_timesteps.shape}")

    # Obtain x and y coordinates and extract new field values
    x_idx = valid_particles[1, :]
    y_idx = valid_particles[2, :]
    x_idx = (x_idx / 0.75 * 1500).astype(int)
    y_idx = ((y_idx + 0.3) / 0.6 * 1200).astype(int)

    new_vals = np.full(n_tsteps * n_particles, np.nan)
    # new_vals[valid_mask] = new_data[valid_timesteps, x_idx, y_idx]
    # f_name = 'D:/singlesource_2d_extended/FTLE_extendedsim_180s.h5'
    # with h5py.File(f_name, 'r') as f:
    # #     # Max Principal Strain & U velocity; original dimensions (time, columns, rows) = (9001, 1201, 1501)
    #     new_data = f.get('maxPstrain')[:, :, :]
    
    f_name = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(f_name, 'r') as f:
        v_data = f.get('Flow Data/v')[:-1, :, :].transpose(0, 2, 1)  # Remove final(?) timestep to match particle tracking & strain data
        # v_data = f.get('Flow Data/v')[:-1, :, :].transpose(0, 2, 1)

    # DEBUG(f"strain data dimensions: {new_data.shape}")
    new_vals[valid_mask] = v_data[valid_timesteps, y_idx, x_idx].astype(np.float32)
    # DEBUG(f"Filtered strain data dimensions: {new_data.shape}")
    new_vals = new_vals.reshape(n_tsteps, n_particles)
    np.save('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_0to180s_vdata.npy', new_vals)

    # Concatenate to original data
    # DEBUG("Concatenating u data to existing matrix")
    # start = time.time()
    # particle_data_expanded = np.concatenate((particle_data, new_vals[:, np.newaxis, :]), axis=1)
    # DEBUG(f"u Concatenation completed in {round(time.time()-start, 2)} sec.")
    
    # new_vals[valid_mask] = v_data[valid_timesteps, y_idx, x_idx]
    # new_vals = new_vals.reshape(n_tsteps, n_particles)
    # DEBUG("Concatenating v data to existing matrix")
    # start = time.time()
    # particle_data_expanded = np.concatenate((particle_data_expanded, new_vals[:, np.newaxis, :]), axis=1)
    # DEBUG(f"v Concatenation completed in {round(time.time()-start, 2)} sec.")

    # if field2 is None:
    #     INFO("1 New field added to particle data.")
    
    # else:
    #     new_vals2 = np.full(particles_flat.shape[0], np.nan)
    #     new_vals2[valid_mask] = field2[valid_timesteps, x_coords, y_coords]
    #     new_vals2 = new_vals2.reshape(n_tsteps, n_particles)
    #     DEBUG("Concatenating new data field 2 to existing matrix")
    #     start = time.time()
    #     particle_data_expanded = np.concatenate((particle_data_expanded, new_vals2[:, np.newaxis, :]), axis=1)
    #     DEBUG(f"Concatenation 2 completed in {round(time.time()-start, 2)} sec.")
    #     INFO("2 new fields added to particle data.")

    # return particle_data_expanded
    return


def main():
    
    ########## PART 1: Extract particle tracking data and add strain and acceleration data, if needed

    # Load particle tracking data; original dimensions (time, features, particles) = (9001, 3, 180000)
    particle_matrix = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_nanUpstream_0to180s_normal.npy')

    # particle_matrix = particle_matrix[:9001, :, :180000]

    # # Obtain the instantaneous max principal strain at all time steps (HDF5 file)
    # # f_name = 'D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
    # f_name = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    # with h5py.File(f_name, 'r') as f:
    #     # Numeric grids; original dimensions (x, y) = (1501, 1201)
    #     # x_grid = f.get('Model Metadata/xGrid')[:]
    #     # y_grid = f.get('Model Metadata/yGrid')[:]

    #     # dx = f.get('Model Metadata/spatialResolution')[0].item()
    #     freq = f.get('Model Metadata/timeResolution')[0].item()
    #     dt = 1 / freq  # convert from Hz to seconds
    
    # # f_name = 'D:/singlesource_2d_extended/FTLE_extendedsim_180s.h5'
    # # with h5py.File(f_name, 'r') as f:
    # # #     # Max Principal Strain & U velocity; original dimensions (time, columns, rows) = (9001, 1201, 1501)
    # #     strain_data = f.get('maxPstrain')[:]
    # # #     u_data = f.get('Flow Data/u')[:-1, :, :]  # Remove final(?) timestep to match particle tracking & strain data

    # # # Compute streamwise acceleration at all timesteps
    # # accel_x = np.gradient(u_data, axis=0)

    # # # For each x, y, t location listed in particle tracking data matrices, retrieve the associated velocity at that time step
    # add_newfield_to_particles(particle_matrix)
    # particles_w_velocities = add_newfield_to_particles(particle_matrix)
    # # Data matrix is now: release time, x, y, strain, at each time step
    # file_name = 'ignore/ParticleTrackingData/ParticleVelocities_Extendedsim_n20_t0to180_D1.5.npy'
    # INFO(f"Saving expanded particle data to {file_name}.")
    # np.save(file_name, particles_w_velocities)
    # INFO("Save complete.")

    ################ END PART 1 ###################

    #### STRAINS VS TRAVEL TIME CALCS ##########

    # Load expanded particle tracking data if already computed
    # Numpy file columns (in this order): release time, x, y, strain (at each time step) = (9001, 4, 180000)




    # file_name = 'ignore/ParticleTrackingData/ParticleStrains_Extendedsim_n20_t0to180_D1.5.npy'
    # particles_w_strain = np.load(file_name)
    # n_tsteps, n_features, n_particles = particles_w_strain.shape
    # dt = 0.02

    # # Flatten and mask particle data
    # particles_flat = particles_w_strain.transpose(1, 0, 2)
    # particles_flat = particles_flat.reshape(n_features, n_tsteps * n_particles)
    # valid_mask = ~np.isnan(particles_flat[1, :]) & ~np.isnan(particles_flat[2, :])
    
    # # Flattened array for timesteps
    # valid_timesteps = np.repeat(np.arange(n_tsteps), n_particles)

    # # Filter to valid particles & timesteps
    # valid_particles = particles_flat[:, valid_mask]
    # # DELETE particles_flat 
    # del particles_flat
    # DEBUG(f"Shape of valid particles array: {valid_particles.shape}")
    # valid_timesteps = valid_timesteps[valid_mask]
    # DEBUG(f"Shape of valid timesteps array: {valid_timesteps.shape}")

    # # Obtain x and y coordinates and extract new field values
    # x_coords = valid_particles[1, :]
    # y_coords = valid_particles[2, :]


    # # List of detector properties for looped version
    # det_x_vals = [0.05, 0.0707, 0.1, 0.1414, 0.2, 0.2828, 0.4]
    # det_height_vals = [0.015, 0.01842, 0.02191, 0.02605, 0.0310, 0.0368, 0.04382]  # Computed as 4*sqrt(2*1.5*ts)*0.01, where ts is avg time to sensor, to match JDV heights

    # # Dictionary to store avg speed values at each location
    # avg_u_vals = {'det 1': None, 'det 2': None, 'det 3': None, 'det 4': None, 'det 5': None, 'det 6': None, 'det 7': None}
    # avg_totvel_vals = {'det 1': None, 'det 2': None, 'det 3': None, 'det 4': None, 'det 5': None, 'det 6': None, 'det 7': None}

    # for i in range(len(det_x_vals)):
    #     # Compute travel times of each particle to detector (sensor); keep only particles that reach sensor
    #     det_x = det_x_vals[i]  # downstream detector distance (m)
    #     det_y = 0  # cross-stream detector position (m, center of detector)
    #     det_width = 0.01  # detector width, streamwise (m)
    #     det_height = det_height_vals[i]  # detector height, cross-stream (m)


    #     # Define conditions for sensor detection
    #     det_condition = ((det_x <= x_coords) & (x_coords <= (det_x + det_width))) & (((det_y - det_height / 2) <= y_coords) & (y_coords <= (det_y + det_height / 2)))

    #     cond_array = np.full(n_tsteps * n_particles, False)
    #     cond_array[valid_mask] = np.where(det_condition, det_condition, False)
    #     cond_array = cond_array.reshape(n_tsteps, n_particles).T  # Now 60000 x 3000 array of T/F, or NAN
    #     # first_detect = np.zeros_like(cond_array, dtype=bool)
    #     # first_detect_time = np.arange(len(cond_array)), cond_array.argmax(axis=1)
    #     first_detect_idxs = cond_array.argmax(axis=1)

    #     # Vector of particle id numbers
    #     particle_ids = np.linspace(0, n_particles-1, n_particles, dtype=int)
    #     # extract release times and cumulative strain from particle matrix
    #     release_times = particles_w_strain[0, 0, :].round(2)
    #     # release_idxs = ((release_times / dt).astype(int))
    #     travel_times = (first_detect_idxs - ((release_times / dt).astype(int))) * dt

    #     # average strain from release time to first detection for each detected particle
    #     # First, create a mask to select the appropriate slices for each particle
    #     mask = np.arange(particles_w_strain.shape[0])[:, np.newaxis]  # shape (timesteps, 1)
    #     # Create a boolean mask for each particle's range
    #     bool_mask = (mask >= ((release_times / dt).astype(int))) & (mask < first_detect_idxs)
    #     # # Calculate the average strain across the valid range for each particle
    #     # strains = np.where(bool_mask[:, :], particles_w_strain[:, 3, :], np.nan)
    #     # # Now compute the mean along the time dimension, ignoring the NaNs
    #     # avg_strains = np.nanmean(strains, axis=0)
    #     # cum_strains = np.nansum(strains, axis=0)
    #     # max_strains = np.nanmax(strains, axis=0)

    #     # Compute length of trajectories
    #     traj_coords_x = np.where(bool_mask[:, :], particles_w_strain[:, 1, :], np.nan)
    #     traj_x_prev = np.roll(traj_coords_x, 1, axis=0)
    #     traj_coords_y = np.where(bool_mask[:, :], particles_w_strain[:, 2, :], np.nan)
    #     traj_y_prev = np.roll(traj_coords_y, 1, axis=0)
    #     traj_length = np.nansum(1000*(np.sqrt((traj_coords_x-traj_x_prev)**2 + (traj_coords_y-traj_y_prev)**2)), axis=0)
    #     # strains_distavg = cum_strains / traj_length
    #     # t2_avg_strain = cum_strains / (travel_times)**2
    #     # # t5_avg_strain = cum_strains / (travel_times)**5

    #     # Compute average speed of particles
    #     speed = traj_length/ 1000 / travel_times  # m/sec
    #     horiz_speed = det_x_vals[i] / travel_times

    #     # Create mask for detected particles based on detect time not equal to zero
    #     # detected_idxs = np.where((first_detect_idxs != 0) & (traj_length >=525) & (traj_length <=575), True, False)
    #     detected_idxs = np.where((first_detect_idxs != 0), True, False)

    #     avg_totvel_vals[f'det {i+1}'] = speed[detected_idxs]
    #     avg_u_vals[f'det {i+1}'] = horiz_speed[detected_idxs]
  
    # # Save dictionaries of velocities
    # with open('ignore/tests/avg_total_vel.pkl', 'wb') as fp:
    #     pickle.dump(avg_totvel_vals, fp)
    # with open('ignore/tests/avg_u_vel.pkl', 'wb') as fp:
    #     pickle.dump(avg_u_vals, fp)





    # Load dictionaries of average velocities if already saved
    # with open('ignore/tests/avg_total_vel.pkl', 'rb') as fp:
    #     avg_totvel_vals = pickle.load(fp)
    with open('ignore/tests/avg_u_vel.pkl', 'rb') as fp:
        avg_u_vals = pickle.load(fp)
    
    # Dataframe of detected particles
    # particles_df = pd.DataFrame({'Particle_ID': particle_ids[detected_idxs], 'Release_t': release_times[detected_idxs], 'Detect_t': first_detect_idxs[detected_idxs], 
    #                             'Travel_t': travel_times[detected_idxs]**2, 'travel_dist': traj_length[detected_idxs], 'avg_horiz_speed': horiz_speed[detected_idxs], 't2_avg_strain': t2_avg_strain[detected_idxs], 
    #                             'dist_avg_strain': strains_distavg[detected_idxs], 'cumulative_strain': cum_strains[detected_idxs], 'max_strain': max_strains[detected_idxs]})

    # # particles_df = pd.DataFrame({'Particle_ID': particle_ids[detected_idxs], 'Release_t': release_times[detected_idxs], 'Detect_t': first_detect_idxs[detected_idxs], 
    # #                             'Travel_t': travel_times[detected_idxs], 'cumulative_strain': cum_strains[detected_idxs]})
    
    # particles_df.plot(x='Travel_t', y='travel_dist', style='o')
    # plt.ylabel('travel distance (mm)')
    # plt.xlabel('travel time^2 (s)')
    # plt.title('travel length vs travel time, traj length = 525 to 575')
    # plt.show()

    # 1D PLOTS OF AVERAGE VELOCITY: HORIZONTAL AND TOTAL
    # for i in range(len(det_x_vals)):
    #     plt.hist(avg_totvel_vals[f'det {i+1}'])
    #     plt.show()

    mean_ic = [0.5, 0.707, 1, 1.414, 2, 2.828, 4]

    for i in range(7):
        # data = (avg_u_vals[f'det {i+1}'])
        data = np.log(avg_u_vals[f'det {i+1}'])

        # Set bin width using Freedman-Diaconis rule (https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule)
        # q1 = np.percentile(data, 25)
        # q3 = np.percentile(data, 75)
        # iqr = q3 - q1
        # bin_width = 2 * iqr / len(data)**(1/3)
        # nbins = (max(data) - min(data)) / bin_width

        # Fit Gaussian function
        def gaussian(x, amplitude, mean, std):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        nbins = 20
        hist, bin_edges = np.histogram(data, bins=nbins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.hist(data, bins=nbins, density=True, alpha=0.6)

        # Fit the Gaussian to the histogram
        p0 = [0.05, mean_ic[i], 1]  # Initial parameter guess: [amplitude, mean, stddev]
        params, covariance = curve_fit(gaussian, bin_centers, hist, p0=p0)
        A_fit, mu_fit, sigma_fit = params

        # Plot the fitted Gaussian
        # x = np.linspace(min(data), max(data), 100)
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
        y_fit = gaussian(x_fit, A_fit, mu_fit, sigma_fit)
        plt.hist(data, bins=nbins, density=True, alpha=0.6, color='gray', label='Data histogram')
        plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Gaussian fit\n$A={A_fit:.2f}$, $\mu={mu_fit:.2f}$, $\sigma={sigma_fit:.2f}$')
        plt.xlabel('mean horizontal velocity, m/s')
        plt.ylabel('Density')
        plt.title(f'Histogram and Gaussian Fit, Sensor {i+1}')
        plt.legend()
        plt.show()

        # Diagonal elements of the covariance matrix are the variances
        # param_errors = np.sqrt(np.diag(covariance))
        # print("Fitted parameters:")
        # print(f"A = {A_fit:.2f} ± {param_errors[0]:.2f}")
        # print(f"mu = {mu_fit:.2f} ± {param_errors[1]:.2f}")
        # print(f"sigma = {sigma_fit:.2f} ± {param_errors[2]:.2f}")


    ### JOINTLY DETECTED PARTICLES ANALYSIS #########

    # Find pairs of jointly detected particles (same Detect_t; at least 2)
    # joint_detect_particles = particles_df[particles_df.duplicated('Detect_t', keep=False) == True]

    # # Self-merge dataframe with itself then use filtering to create dataframe with all possible combinations of particle pairs
    # merged_df = joint_detect_particles.merge(joint_detect_particles, on='Detect_t', suffixes=('_1', '_2'))
    # merged_df = merged_df[merged_df['Particle_ID_1'] != merged_df['Particle_ID_2']]
    # merged_df = merged_df[merged_df['Particle_ID_1'] < merged_df['Particle_ID_2']]

    # # Create unique pair_ID
    # merged_df['Pair_ID'] = merged_df['Particle_ID_1'].astype(str) + '_' + merged_df['Particle_ID_2'].astype(str)

    # # Compute differences in travel times, release times, strains
    # merged_df['delta_travel'] = abs(merged_df['Travel_t_1'] - merged_df['Travel_t_2'])
    # merged_df['delta_release'] = abs(merged_df['Release_t_1'] - merged_df['Release_t_2'])
    # merged_df['delta_t_avg_strain'] = abs(merged_df['t_avg_strain_1'] - merged_df['t_avg_strain_2'])
    # merged_df['delta_d_avg_strain'] = abs(merged_df['dist_avg_strain_1'] - merged_df['dist_avg_strain_2'])
    # merged_df['delta_sum_strain'] = abs(merged_df['cumulative_strain_1'] - merged_df['cumulative_strain_2'])
    # # merged_df['delta_detect_t'] = abs(merged_df['Detect_t_1'] - merged_df['Detect_t_2'])

    # particle_pair_df = merged_df[['Pair_ID', 'Detect_t', 'delta_travel', 'delta_release', 'delta_t_avg_strain', 'delta_d_avg_strain', 'delta_sum_strain']]
    # particle_pair_samefreq = particle_pair_df.loc[particle_pair_df['delta_release']==0]
    # particle_pair_difffreq = particle_pair_df.loc[particle_pair_df['delta_release'] > 0]

    # plt.close()
    # particle_pair_difffreq.plot.scatter(x='delta_travel', y='delta_sum_strain', style='o')
    # plt.xlabel('difference in travel time')
    # plt.ylabel('difference in cumulative strain')
    # plt.title('Jointly detected particles: difference cumulative strain vs difference in travel time')
    # plt.show()


    # ####### plotting and analyzing strain & acceleration along Lagrangian trajectories ########

    # # xlim = [0, 0.5]
    # # ylim = [-0.211, 0.211]

    # # # QC: spatial plot of strain vals at a few times
    # # plot_times = [100, 500, 1000, 2999]
    # # for t in plot_times:
    # #     plot_data = particles_w_strain[t, :, :]
    # #     plt.scatter(plot_data[1, :], plot_data[2, :], c=plot_data[3, :], s=100)
    # #     plt.colorbar()
    # #     plt.xlim(xlim[0], xlim[1])
    # #     plt.ylim(ylim[0], ylim[1])
    # #     plt.show()

    # # # PLOT: spatial plot of strain along trajectories for all time for particles 1-20
    # # fig, ax = plt.subplots()
    # # startidx = 2100
    # # endidx = 2120
    # # for p in range(startidx, endidx):
    # #     plt.scatter(particles_w_strain[:, 1, p], particles_w_strain[:, 2, p], c=particles_w_strain[:, 3, p], cmap=cmr.ember, 
    # #                 norm=colors.LogNorm(), s=25, alpha=0.5)
    # # plt.colorbar()
    # # plt.xlim(xlim[0], xlim[1])
    # # plt.ylim(ylim[0], ylim[1])
    # # plt.title(f'Strain along trajectories, release time {round(particles_w_strain[0, 0, startidx], 2)} s')
    # # plt.show()

    # # # PLOT: many-line plot of strain as f(t) with 0 as release time
    # # plt.close()
    # # fig, ax = plt.subplots(figsize=(8, 4))
    # # for p in range(startidx, endidx):
    # #     plot_data = particles_w_strain[:, 3, p]
    # #     plot_data = plot_data[~np.isnan(plot_data)]
    # #     plt.plot(plot_data)
    # # plt.ylim(0, 15)
    # # plt.xlim(0, 250)
    # # plt.ylabel('max principal strain')
    # # plt.xlabel('timesteps from release')
    # # plt.show()

    # # PLOT: ILS line plots of slice at x=0, x=0.025, x=0.05 m
    # # file_path = 'C:/Users/elles/Documents/CU_Boulder/Fluids_Research/FisherPlume_plots/flow_stats_plots/ignore/ILS_ustreamwise.npy'
    # # ILS_data = np.load(file_path)  # original dimensions: (y, x) = (846, 1001)
    # # ILS_subset = ILS_data[123:724, 0:101]

    # # # spatial average: time-averaged ILS for x=0 to 0.05 m, y=-0.15 to y=0.15 m
    # # INFO(f'ILS spatial avg, x range [0, 0.05] m, y range [-0.15, 0.15] m: {round(np.mean(ILS_subset), 4)} m.')

    # # xidxs = [0, 50, 100]
    # # plt.close()
    # # fig, ax = plt.subplots()
    # # for xidx in xidxs:
    # #     plt.plot(ILS_subset[:, xidx], label=(f'x={xidx*0.0005} m'))
    # # plt.ylabel('ILS (m)')
    # # plt.xlabel('y location (idx)')
    # # plt.legend()
    # # plt.show()        

    # # PLOT: many-line plot of acceleration as f(t) with 0 as release time

    # # Think about particle PAIRS: xxxx as f(diff in release times)

if __name__=='__main__':
    main()

