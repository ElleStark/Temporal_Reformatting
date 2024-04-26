# Main function for generating two-particle simulations for analyzing temporal reformatting of odor sources by flow fields
# Elle Stark, Ecological Fluid Dynamics Lab - CU Boulder, with A True & J Crimaldi
# In collaboration with J Victor, Weill Cornell Medical College
# May 2024

import h5py
import numpy as np
from src import flowfield, odor, simulation
import scipy.io

def main():
    # Define data subset
    x_lims = slice(None, None)
    y_lims = slice(None, None)
    time_lims = slice(None, None)
    odor_name = 'c1a'

    # Import required data from H5 file
    f_name = 'D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
    with h5py.File(f_name, 'r') as f:
        # Metadata: spatiotemporal resolution and domain size
        freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / freq  # convert from Hz to seconds
        time_array_data = f.get('Model Metadata/timeArray')[time_lims]
        dx = f.get('Model Metadata/spatialResolution')[0].item()
        domain_size = f.get('Model Metadata/domainSize')
        domain_width = domain_size[0].item()  # [m] cross-stream distance
        domain_length = domain_size[1].item()  # [m] stream-wise distance

        # Numeric grids
        xmesh_uv = f.get('Model Metadata/xGrid')[x_lims, y_lims].T
        ymesh_uv = f.get('Model Metadata/yGrid')[x_lims, y_lims].T

        # Velocities: for faster reading, can read in subset of u and v data here
        # dimensions of multisource plume data (time, columns, rows) = (3001, 1001, 846)
        u_data = f.get('Flow Data/u')[time_lims, x_lims, y_lims].T
        v_data = f.get('Flow Data/v')[time_lims, x_lims, y_lims].T

    # desired flow field resolution
    dx_sim = dx
    dt_sim = dt
    # construct simulation mesh
    xvec_sim = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(np.shape(u_data)[1] * dx/dx_sim))
    yvec_sim = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(np.shape(u_data)[0] * dx/dx_sim))
    xmesh_sim, ymesh_sim = np.meshgrid(xvec_sim, yvec_sim, indexing='xy')
    ymesh_sim = np.flipud(ymesh_sim)

    # Create flowfield object
    flow = flowfield.FlowField(xmesh_sim, ymesh_sim, u_data, v_data, xmesh_uv, ymesh_uv, dt_sim)

    # Odor source properties
    osrc_loc = [0, 0]  # location (m) relative to x_lims and y_lims subset of domain, source location at which to release particles
    tau = dt  # seconds, time between particle releases
    # D_osrc = 1.5*10**(-5)  # meters squared per second; particle diffusivity
    D_osrc = 0 

    # Create odor object
    odor_src = odor.OdorSource(tau, osrc_loc, D_osrc)

    # Use flowfield, odor, and simulation parameters to generate particle simulation object
    duration = time_array_data[-2]
    t0 = 0
    test_sim = simulation.Simulation(flow, odor_src, duration, t0, dt_sim)

    # Compute simulation trajectories: array with time each particle is released & trajectory at each timestep (x, y position at each dt)
    n_particles = 1  # particles to be released AT EACH TIMESTEP
    test_sim.track_particles_rw(n_particles, method='IE')

    # Save raw trajectory data
    # save to Numpy array:
    f_name = f'ignore/tests/particleTracking_n{n_particles}_fullsim_D{D_osrc}.npy'
    np.save(f_name, test_sim.trajectories)

    # Plot results
    f_path = f'ignore/tests/traj_plot_n{n_particles}_d{round(odor_src.D_osrc, 1)}'
    test_sim.plot_trajectories(f_path, frames=list(range(test_sim.n_frames)), domain_width=domain_width, domain_length=domain_length, movie=True)

    # save to .mat file:
    f_path = f'ignore/tests/ParticleTracking_MSPlumeSim_n{n_particles}_t60s_D{D_osrc}.mat'
    scipy.io.savemat(f_path, {'data': test_sim.trajectories, 'meta':{'ParticleTrackingParams':{'num_particles': f'{n_particles} seeded each frame', 'num_frames': '2999', 'dt': '0.02 sec', 'duration': '60 sec', 'diffusionCoefficient': f'{D_osrc} m^2/s', 'gridResolution': '0.0005 meter', 'ParticleReleasePoint': '(0, 0)', 'NumericalAdvectionMethod': 'Improved Euler'}, 
                                                'FlowfieldSimulationInfo':{'description':'2D grid turbulence Comsol model', 'source': 'Fisher Plume manuscript Tootoonian et al., 2024', 'meanVelocity': '10 cm/s', 'xDomain': '[0, 0.5] meters', 'yDomain': '[-0.211, 0.211] meters'}, 
                                                'FileCreationInfo': {'creationDate': 'April 2024', 'createdBy': 'Elle Stark, EFD Lab, CU Boulder CEAE Dept', 'contact': 'elle.stark@colorado.edu or aaron.true@colorado.edu'}}})

if __name__=='__main__':
    main()

