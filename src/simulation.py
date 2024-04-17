# Class to create and run simulation
# Elle Stark, May 2024

import numpy as np


class Simulation:
    def __init__(self, flowfield, odorsource, duration, t0, dt) -> None:
        self.flowfield = flowfield
        self.odorsource = odorsource
        self.duration = duration
        self.t0 = t0
        self.dt = dt

    def track_particles_rw(self, n_particles, dt, duration, D, Lb, method = 'IE'):
        """
        Uses Lagrangian particle tracking model with random walk diffusion to calculate particle positions over time
        for sets of particles initialized at the same location at different times.
        :param n_particles: float, number of particles to release and track AT EACH FRAME
        :param dt: float, length of timestep
        :param duration: float, total time to transport particles
        :param D: float, diffusion coefficient
        :return: nd array representing the positions over time for sets of particles released at a single location dt apart
        """

        # Define starting matrix for particles (all released at same location)
        src_loc = self.odorsource.osrc_loc

        L = abs(int(duration / dt))  # number of frames
        #nx = len(self.x[0, :])
        #ny = len(self.y[:, 0])

        # at each timestep, advect particles and add diffusion with random walk
        trajectories = np.zeros((2, L, n_particles))

        for step in range(L):
            tstep = step * dt

            # numerical advection & diffusion of particles
            if method=='IE':
                loc_out = utils.ImprovedEuler_singlestep(dt, tstep, )
            blob1_out = self.improvedEuler_singlestep(dt, tstep, blob1) + sqrt(2 * D * dt) * np.random.randn(*blob1.shape)
            #blob1_out = blob1 + self.vfield(tstep, blob1) * dt + sqrt(2 * D * dt) * np.random.randn(blob1.shape[0], blob1.shape[1])
            #blob1_out = blob1 + self.vfield(tstep, blob1) * dt
            blob1 = blob1_out
            blob1_single_steps[:, step, :] = blob1_out
            # Blob 1 concentrations - use numpy's built-in histogram function
            # conc1, xbins1, ybins1 = np.histogram2d(blob1_out[1, :], blob1_out[0, :], bins=(50, 100))
                                                   #bins=(np.linspace(0, 1, ny+1), np.linspace(0, 2, nx+1)))
            # blob1_conc_steps[step, :, :] = conc1

            # Blob 2 (blue blob)
            blob2_out = self.improvedEuler_singlestep(dt, tstep, blob2) + sqrt(2 * D * dt) * np.random.randn(*blob2.shape)
            #blob2_out = self.improvedEuler_singlestep(dt, tstep, blob2)
            blob2 = blob2_out
            blob2_single_steps[:, step, :] = blob2_out

        self.trajs_w_diff = [blob1_single_steps, blob2_single_steps]

        return blob1_single_steps, blob2_single_steps
    
    