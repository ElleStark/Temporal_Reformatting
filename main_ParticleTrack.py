# Main function for generating two-particle simulations for analyzing temporal reformatting of odor sources by flow fields
# Elle Stark, Ecological Fluid Dynamics Lab - CU Boulder, with A True & J Crimaldi
# In collaboration with J Victor, Weill Cornell Medical College
# May 2024

import h5py
import hdf5storage
import numpy as np
from src import flowfield, odor, simulation
import scipy.io

def main():
    # # Define data subset
    # x_lims = slice(None, None)
    # y_lims = slice(None, None)
    # time_lims = slice(0, 9001)

    # # Import required data from H5 file
    # f_name = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    # with h5py.File(f_name, 'r') as f:
    #     # Metadata: spatiotemporal resolution and domain size
    #     freq = f.get('Model Metadata/timeResolution')[0].item()
    #     dt = 1 / freq  # convert from Hz to seconds
    #     time_array_data = f.get('Model Metadata/timeArray')[:]
    #     dx = f.get('Model Metadata/spatialResolution')[0].item()
    #     domain_size = f.get('Model Metadata/domainSize')
    #     domain_width = domain_size[0].item()  # [m] cross-stream distance
    #     domain_length = domain_size[1].item()  # [m] stream-wise distance

    #     # Numeric grids
    #     xmesh_uv = f.get('Model Metadata/xGrid')[x_lims, y_lims].T
    #     ymesh_uv = f.get('Model Metadata/yGrid')[x_lims, y_lims].T

    #     # Velocities: for faster reading, can read in subset of u and v data here
    #     # dimensions of multisource plume data (time, columns, rows) = (3001, 1001, 846); extended is (9001, 1501, 1201)
    #     # u_data = f.get('Flow Data/u')[time_lims, x_lims, y_lims].T
    #     # v_data = f.get('Flow Data/v')[time_lims, x_lims, y_lims].T

    # # desired flow field resolution
    # dx_sim = dx
    # dt_sim = dt
    # # construct simulation mesh
    # xvec_sim = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(np.shape(xmesh_uv)[1] * dx/dx_sim))
    # yvec_sim = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(np.shape(ymesh_uv)[0] * dx/dx_sim))
    # xmesh_sim, ymesh_sim = np.meshgrid(xvec_sim, yvec_sim, indexing='xy')
    # ymesh_sim = np.flipud(ymesh_sim)

    # # Create flowfield object
    # flow = flowfield.FlowField(xmesh_sim, ymesh_sim, xmesh_uv, ymesh_uv, dt_sim, x_lims, y_lims)

    # # Approximate and plot vortex shedding frequency at input if desired using fft
    # # flow.find_plot_psd([900, 901], [483, 484], plot=True)

    # # Odor source properties
    # osrc_loc = [0, 0]  # location (m) relative to x_lims and y_lims subset of domain, source location at which to release particles
    # tau = dt  # seconds, time between particle releases
    # D_osrc = 1.5*10**(-5)  # meters squared per second; particle diffusivity
    # # D_osrc = 0 

    # # Create odor object
    # odor_src = odor.OdorSource(tau, osrc_loc, D_osrc)

    # # Use flowfield, odor, and simulation parameters to generate particle simulation object
    # duration = time_array_data[-2]
    # t0 = 0
    # test_sim = simulation.Simulation(flow, odor_src, duration, t0, dt_sim)

    # # Compute simulation trajectories: array with time each particle is released & trajectory at each timestep (x, y position at each dt)
    # n_particles = 20  # particles to be released AT EACH TIMESTEP
    # test_sim.track_particles_rw(n_particles, method='IE')

    # # Save raw trajectory data
    # note = 'nanUpstream'
    # # save to Numpy array:
    # sim = '_extended'
    # f_name = f'ignore/ParticleTrackingData/particleTracking_sim{sim}_n{n_particles}_fullsim_D1.5_{note}_0to180s_normal.npy'
    # np.save(f_name, test_sim.trajectories)

    # # Plot results
    # f_path = f'ignore/ParticleTrackingData/traj_plot_sim{sim}_n{n_particles}_d{round(odor_src.D_osrc, 1)}_{note}_0to180s_normal'
    # test_sim.plot_trajectories(f_path, frames=list(range(test_sim.n_frames)), domain_width=domain_width, domain_length=domain_length, movie=True)

    # save to .mat file:
    dataset = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_nanUpstream_0to180s_normal.npy')
    data_dict = {'trajs1': dataset}
    # dataset = test_sim.trajectories
    f_path = f'ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_0to180s_D1.5.mat'
    hdf5storage.savemat(f_path, data_dict, format='7.3', matlab_compatible=True, compress=True)
    # scipy.io.savemat(f_path, {'data': dataset, 'meta':{'ParticleTrackingParams':{'num_particles': f'{n_particles} seeded each frame', 'num_frames': '3000', 'dt': '0.02 sec', 'duration': '60 sec', 'diffusionCoefficient': f'{D_osrc} m^2/s', 'gridResolution': '0.0005 meter', 'ParticleReleasePoint': '(0, 0)', 'NumericalAdvectionMethod': 'Improved Euler'}, 
    #                                             'FlowfieldSimulationInfo':{'description':'2D grid turbulence Comsol model', 'source': 'Fisher Plume manuscript Tootoonian et al., 2024', 'meanVelocity': '10 cm/s', 'xDomain': '[0, 0.75] meters', 'yDomain': '[-0.3, 0.3] meters'}, 
    #                                             'FileCreationInfo': {'creationDate': 'Aug 2024', 'createdBy': 'Elle Stark, EFD Lab, CU Boulder CEAE Dept', 'contact': 'elle.stark@colorado.edu or aaron.true@colorado.edu'}}})

if __name__=='__main__':
    main()

