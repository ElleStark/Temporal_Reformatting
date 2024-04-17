# Main function for generating two-particle simulations for analyzing temporal reformatting of odor sources by flow fields
# Elle Stark, Ecological Fluid Dynamics Lab - CU Boulder, with A True & J Crimaldi
# In collaboration with J Victor, Weill Cornell Medical College
# May 2024

from flowfield import FlowField
import h5py
import numpy as np
from odor import OdorSource

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
    flow = FlowField(xmesh_sim, ymesh_sim, u_data, v_data, xmesh_uv, ymesh_uv, dt)

    # Odor source properties
    osrc_loc = [423, 0]  # indexes relative to x_lims and y_lims subset of domain, source location at which to release particles
    tau = 0.1  # seconds, time between particle releases
    D_osrc = 1.5*10**(-5)  # meters squared per second; particle diffusivity

    # Create odor object
    odor = OdorSource(tau, osrc_loc, D_osrc)

    # Use flowfield, odor, and simulation parameters to generate particle simulation object
    test_sim = 

    # output: array with tagged particle #, time released, trajectory (x, y position at each dt)


    # Save raw trajectory data

    # Compute desired information from simulation

    # Output processed data

    # Plot results (if desired)

if __name__=='__main__':
    main()

