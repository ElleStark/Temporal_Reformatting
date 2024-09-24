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
    # particle_matrix = particle_matrix.astype(np.float32)
    # np.save('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_nanUpstream_0to180s_f32.npy', particle_matrix)

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

    # # Compute travel times of each particle to detector (sensor); keep only particles that reach sensor
    # det_x = 0.45  # downstream detector distance (m)
    # det_y = 0  # cross-stream detector position (m, center of detector)
    # det_width = 0.01  # detector width, streamwise (m)
    # det_height = 0.0465  # detector height, cross-stream (m)

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

    # # Define conditions for sensor detection
    # det_condition = ((det_x <= x_coords) & (x_coords <= (det_x + det_width))) & (((det_y - det_height / 2) <= y_coords) & (y_coords <= (det_y + det_height / 2)))

    # cond_array = np.full(n_tsteps * n_particles, False)
    # cond_array[valid_mask] = np.where(det_condition, det_condition, False)
    # cond_array = cond_array.reshape(n_tsteps, n_particles).T  # Now 60000 x 3000 array of T/F, or NAN
    # # first_detect = np.zeros_like(cond_array, dtype=bool)
    # # first_detect_time = np.arange(len(cond_array)), cond_array.argmax(axis=1)
    # first_detect_idxs = cond_array.argmax(axis=1)

    # # Vector of particle id numbers
    # particle_ids = np.linspace(0, n_particles-1, n_particles, dtype=int)
    # # extract release times and cumulative strain from particle matrix
    # release_times = particles_w_strain[0, 0, :].round(2)
    # # release_idxs = ((release_times / dt).astype(int))
    # travel_times = (first_detect_idxs - ((release_times / dt).astype(int))) * dt

    # # average strain from release time to first detection for each detected particle
    # # First, create a mask to select the appropriate slices for each particle
    # mask = np.arange(particles_w_strain.shape[0])[:, np.newaxis]  # shape (timesteps, 1)
    # # Create a boolean mask for each particle's range
    # bool_mask = (mask >= ((release_times / dt).astype(int))) & (mask < first_detect_idxs)
    # # Calculate the average strain across the valid range for each particle
    # strains = np.where(bool_mask[:, :], particles_w_strain[:, 3, :], np.nan)
    # # Now compute the mean along the time dimension, ignoring the NaNs
    # avg_strains = np.nanmean(strains, axis=0)
    # cum_strains = np.nansum(strains, axis=0)
    # max_strains = np.nanmax(strains, axis=0)

    # # Compute length of trajectories
    # traj_coords_x = np.where(bool_mask[:, :], particles_w_strain[:, 1, :], np.nan)
    # traj_x_prev = np.roll(traj_coords_x, 1, axis=0)
    # traj_coords_y = np.where(bool_mask[:, :], particles_w_strain[:, 2, :], np.nan)
    # traj_y_prev = np.roll(traj_coords_y, 1, axis=0)
    # traj_length = np.nansum(1000*(np.sqrt((traj_coords_x-traj_x_prev)**2 + (traj_coords_y-traj_y_prev)**2)), axis=0)
    # strains_distavg = cum_strains / traj_length
    # t2_avg_strain = cum_strains / (travel_times)**2
    # # t5_avg_strain = cum_strains / (travel_times)**5

    # # Compute average speed of particles
    # speed = traj_length / travel_times  # mm/sec
    # horiz_speed = 450 / travel_times

    # # Create mask for detected particles based on detect time not equal to zero
    # detected_idxs = np.where((first_detect_idxs != 0) & (traj_length >=525) & (traj_length <=575), True, False)
  
    # # Dataframe of detected particles
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

    ########## COMPUTE PAIRED PARTICLE SEPARATIONS OVER TIME ##########

    # Convert x and y to cm
    # particle_matrix[:, 1:3, :] = particle_matrix[:, 1:3, :] * 100
    mean_u = 10 # cm/s
    # epsilon =  # energy dissipation rate per unit mass

    max_t_steps = 300
    n_tsteps = particle_matrix.shape[0] - max_t_steps - 100
    r2_array = np.zeros((n_tsteps, 20))
    # D_LL_array = np.zeros((n_tsteps, 20))
    r2_0 = np.zeros((n_tsteps, 20))
    # Select times for plotting - for log-log, always want to have 0 as well as first possible time
    t_list = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6])
    t_list_10 = t_list*10
    # t_list = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    dt = 0.02
    t_list_idx = (t_list / dt).astype(int) 
    r2_list = np.zeros(len(t_list))
    # r_list = np.zeros(len(t_list))

    # d_release_list = [0, 10, 25, 50]
    d_release_list = [0, 1, 2, 5, 10, 20, 30, 40, 50]
    # d_release_list = [0, 1, 2, 3, 4, 5]
    all_r2_list = np.zeros((len(d_release_list), len(r2_list)))
    all_r0_list = np.zeros((len(d_release_list)))
    # all_r_list = np.zeros((len(d_release_list)))
    # all_D_LL_list = np.zeros((len(d_release_list)))
    all_idx = 0

    for delta_release in d_release_list:
        idx = 0
        batchelor_t = ((delta_release*dt * mean_u)**2)
        for compute_t in t_list_idx:
            t=0
            for t in range(n_tsteps-1):
                t += 1
                t0 = int(round(particle_matrix[0, 0, t*20 + 20*delta_release] / dt, 0))
                # find separation vectors for particle pairs
                # x_diffs = particle_matrix[t0 + compute_t, 1, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 1, (t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]
                # y_diffs = particle_matrix[t0 + compute_t, 2, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 2, (t*20 + 20*delta_release):(t*20 + 20*delta_release +20)]

                # compute squared distance as delta x squared + delta y squared
                # t0 = int(round(particle_matrix[0, 0, t*20 + 20*delta_release] / dt, 0))
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

            r2_list[idx] = np.nanmean((r2_array - r2_0)**2)
            # r2_list[idx] = np.nanmean(r2_array)**2
            # r_list[idx] = np.nanmean(r2_array)
            idx += 1
        all_r2_list[all_idx, :] = r2_list
        all_r0_list[all_idx] = np.nanmean(r2_0**2)
        # all_r_list[all_idx] = np.nanmean(r_list)
        # all_D_LL_list[all_idx] = np.nanmean(D_LL_array)
        all_idx += 1

    C2 = 2.13
    # epsilon = np.load('ignore/inputs/viscous_dissipation_extendedSim.npy')  # 1201 x 1501 matrix of vals
    # epsilon = 0.001015  # m^2/s^3, for simplicity start with domain average
    # epsilon = 0.0025
    epsilon = 0.006728  # m^2/s^3, first 0.05 m
    nu = 1.5 * 10**(-5)  # kinematic viscosity, m/s
    t_eta = 0.1216  # s, kolmogorov microscale in time, entire domain
    g = 0.5  # Richardson constant
    # t_eta = 0.0472  # first 0.05 m


    # t2_array = t_list_10 ** 2
    # t3_array =t_list_10 ** 3

    fig, ax = plt.subplots()

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

    for i in range(len(d_release_list)-1):
        # i+=1
        plt.loglog(t_list[1:], all_r2_list[i, 1:], label=f'p_sep: {round(d_release_list[i]*dt, 2)}')
        # plt.loglog(t_list[:], g*epsilon*t_list[:]**(3) + 2*all_r0_list[i]**2 - 2*all_r_list[i]*all_r0_list[i], color='blue')
        # plt.loglog(t_list[:], g*epsilon*t_list[:]**(3) + 2*all_r0_list[i]**2, color='red')
    
    # Exponential fit at early times for particle separations for pairs with 0 initial separation 
    plt.loglog(t_list[1:8], 0.4*(all_r0_list[0]*np.exp(t_list[1:8]/0.07) - all_r0_list[0]), color='red')
    
    # t^3 fit for later times for particle separations for pairs with 0 initial separation, compensated by t^3
    # plt.loglog(t_list[1:], all_r2_list[0, 1:])
    # plt.scatter(t_list[1], g*epsilon* t_list[1]**3)
    # plt.scatter(t_list[-1], g*0.0001*t_list[-1]**3)

    # t^(3/2) curve for later times
    plt.loglog(t_list[4:], g*epsilon*t_list[4:]**(3/2), color='black')
    
    # Expected fit:
    plt.loglog(t_list[1:8], epsilon*t_list[1:8]**(2), color='black')
    # Better fit for our data:
    # plt.loglog(t_list[1:8], 0.8*epsilon*t_list[1:8]**(29/16), color='black')
    plt.xlabel('t (s)')
    plt.ylabel('<|r-r0|>^2')
    plt.title('log-log scaling: initial separations 0 to 1 s')

    plt.legend()
    plt.show()

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

