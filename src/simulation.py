# Class to create and run simulation
# Elle Stark, May 2024

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import utils


class Simulation:
    def __init__(self, flowfield, odorsource, duration, t0, dt) -> None:
        self.flowfield = flowfield
        self.odorsource = odorsource
        self.duration = duration
        self.t0 = t0
        self.dt = dt
        self.trajectories = None

    # track LOCAL STRAIN RATE - transfer function in terms of Peclet # (see Villermo paper)
    def track_particles_rw(self, n_particles, D, method = 'IE'):
        """
        Uses Lagrangian particle tracking model with random walk diffusion to calculate particle positions over time
        for sets of particles initialized at the same location at different times.
        :param n_particles: float, number of particles to release and track AT EACH FRAME
        :param dt: float, length of timestep
        :param duration: float, total time to transport particles in seconds
        :param D: float, diffusion coefficient
        :return: nd array representing the positions over time for sets of particles released at a single location dt apart
        """

        # Define number of frames and list of times
        dt = self.dt
        duration = self.duration
        n_frames = abs(int(duration / dt))  
        t_list = np.linspace(self.t0, self.t0 + duration, n_frames)

        # Extract relevant variables
        src_loc = self.odorsource.osrc_loc

        # initialize array of particle trajectory data
        # stack of matrices with dimensions: frame, particle release time, x index, y index
        trajectories = np.empty((n_frames, n_particles*n_frames, 3))
        trajectories[:] = np.nan
        trajectories[:, :, 0] = np.repeat(t_list, n_particles)

        start_idx = 0
        end_idx = n_particles

        # at each timestep, release particles and transport all particles in domain using advection and random walk diffusion 
        for step in range(n_frames):
            tstep = t_list[step]

            # seed new particles at source location
            trajectories[step, start_idx:end_idx, 1] = src_loc[0]
            trajectories[step, start_idx:end_idx, 2] = src_loc[1]
            loc_in = trajectories[step, 0:end_idx, 1:2]

            # numerical advection & diffusion of all particles
            if method=='IE':
                loc_out = self.flowfield.ImprovedEuler_singlestep(dt, tstep, loc_in) + sqrt(2 * D * dt) * np.random.randn(*trajectories.shape)
            elif method=='RK4':
                loc_out = self.flowfield.rk4singlestep(dt, tstep, loc_in)

            # save position of each particle at this step
            trajectories[step + 1, 0:end_idx, 1:2] = loc_out
            
            # shift indices to include next batch of particles starting at (0, 0)
            start_idx = end_idx
            end_idx += n_particles

        self.trajectories = trajectories

    def plot_trajectories(self, frames, domain_width, domain_length):
        """
        
        """
    
    