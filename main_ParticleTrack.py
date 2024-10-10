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
    # Define data subset
    x_lims = slice(1400, 1420)
    y_lims = slice(100, 1100)
    time_lims = slice(0, 9001)

    # Import required data from H5 file
    f_name = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(f_name, 'r') as f:
        # Metadata: spatiotemporal resolution and domain size
        freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / freq  # convert from Hz to seconds
        time_array_data = f.get('Model Metadata/timeArray')[:]
        dx = f.get('Model Metadata/spatialResolution')[0].item()
        domain_size = f.get('Model Metadata/domainSize')
        domain_width = domain_size[0].item()  # [m] cross-stream distance
        domain_length = domain_size[1].item()  # [m] stream-wise distance

        # Numeric grids
        xmesh_uv = f.get('Model Metadata/xGrid')[x_lims, y_lims].T
        ymesh_uv = f.get('Model Metadata/yGrid')[x_lims, y_lims].T

        # Velocities: for faster reading, can read in subset of u and v data here
        # dimensions of multisource plume data (time, columns, rows) = (3001, 1001, 846); extended is (9001, 1501, 1201)
        u_data = f.get('Flow Data/u')[time_lims, x_lims, y_lims]
        v_data = f.get('Flow Data/v')[time_lims, x_lims, y_lims]

    # desired flow field resolution
    dx_sim = dx
    dt_sim = dt
    # construct simulation mesh
    xvec_sim = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(np.shape(xmesh_uv)[1] * dx/dx_sim))
    yvec_sim = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(np.shape(ymesh_uv)[0] * dx/dx_sim))
    xmesh_sim, ymesh_sim = np.meshgrid(xvec_sim, yvec_sim, indexing='xy')
    ymesh_sim = np.flipud(ymesh_sim)

    # Create flowfield object
    flow = flowfield.FlowField(xmesh_sim, ymesh_sim, xmesh_uv, ymesh_uv, dt_sim, x_lims, y_lims)

    # Approximate and plot vortex shedding frequency at input if desired using fft
    # flow.find_plot_psd([900, 901], [483, 484], plot=True)

    # ## TEST WITH SYNTHETIC DATA
    # nx, ny = 128, 128
    # n_timesteps = 20
    # kx = np.fft.fftfreq(nx).reshape(-1, 1)
    # ky = np.fft.fftfreq(ny).reshape(1, -1)
    # k = np.sqrt(kx**2 + ky**2)

    # # Generate synthetic power-law spectrum
    # power_law_spectrum = k**(-5/3)
    # power_law_spectrum[k == 0] = 0  # Avoid division by zero at k=0

    # u_data = np.zeros((n_timesteps, nx, ny))
    # v_data = np.zeros((n_timesteps, nx, ny))

    # for t in range(n_timesteps):
    #     # Generate random phase
    #     # random_phase = np.exp(2j * np.pi * np.random.rand(nx, ny))
    #     # random_phase_v = np.exp(2j * np.pi * np.random.rand(nx, ny))
    #     random_phase = 1
    #     random_phase_v = 1

    #     # Inverse FFT to create synthetic velocity field
    #     u_data[t] = np.fft.ifftn(np.sqrt(power_law_spectrum) * random_phase).real
    #     v_data[t] = np.fft.ifftn(np.sqrt(power_law_spectrum) * random_phase_v).real

    ## END SYNTHETIC DATA CREATION


    # Compute and plot energy spectrum
    u_flx = u_data - np.mean(u_data, axis=0)
    v_flx = v_data - np.mean(v_data, axis=0)
    flow.find_plot_esd(u_flx, v_flx)

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
    # f_name = f'ignore/ParticleTrackingData/particleTracking_sim{sim}_n{n_particles}_fullsim_D1.5_{note}_180to360s.npy'
    # np.save(f_name, test_sim.trajectories)

    # # Plot results
    # f_path = f'ignore/ParticleTrackingData/traj_plot_sim{sim}_n{n_particles}_d{round(odor_src.D_osrc, 1)}_{note}_180to360s'
    # test_sim.plot_trajectories(f_path, frames=list(range(test_sim.n_frames)), domain_width=domain_width, domain_length=domain_length, movie=True)

    # # save to .mat file:
    # # dataset = np.load('ignore/ParticleTrackingData/particleTracking_sim_extended_n20_fullsim_D1.5_nanUpstream_180to360s.npy')
    # data_dict = {'data': test_sim.trajectories}
    # # dataset = test_sim.trajectories
    # f_path = f'ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_180to360s_D1.5.mat'
    # hdf5storage.savemat(f_path, data_dict, format='7.3', matlab_compatible=True, compress=True)
    # # scipy.io.savemat(f_path, {'data': test_sim.trajectories, 'meta':{'ParticleTrackingParams':{'num_particles': f'{n_particles} seeded each frame', 'num_frames': '9000', 'dt': '0.02 sec', 'duration': '180 sec (start:180s end:360s)', 'diffusionCoefficient': f'{D_osrc} m^2/s', 'gridResolution': '0.0005 meter', 'ParticleReleasePoint': '(0, 0)', 'NumericalAdvectionMethod': 'Improved Euler'}, 
    # #                                             'FlowfieldSimulationInfo':{'description':'2D grid turbulence Comsol model', 'source': 'modified from Fisher Plume manuscript Tootoonian et al., 2024', 'meanVelocity': '10 cm/s', 'xDomain': '[0, 0.75] meters', 'yDomain': '[-0.3, 0.3] meters'}, 
    # #                                             'FileCreationInfo': {'creationDate': 'Aug 2024', 'createdBy': 'Elle Stark, EFD Lab, CU Boulder CEAE Dept', 'contact': 'elle.stark@colorado.edu or aaron.true@colorado.edu'}}})

if __name__=='__main__':
    main()

