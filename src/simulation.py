# Class to create and run simulation
# Elle Stark, May 2024

from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Simulation:
    def __init__(self, flowfield, odorsource, duration, t0, dt) -> None:
        self.flowfield = flowfield
        self.odorsource = odorsource
        self.duration = duration
        self.t0 = t0
        self.dt = dt
        self.n_frames = abs(int(duration / dt)) 
        self.trajectories = None

    # track LOCAL STRAIN RATE - transfer function in terms of Peclet # (see Villermo paper)
    def track_particles_rw(self, n_particles, seed = 42, method = 'IE'):
        """
        Uses Lagrangian particle tracking model with random walk diffusion to calculate particle positions over time
        for sets of particles initialized at the same location at different times.
        :param n_particles: float, number of particles to release and track AT EACH FRAME
        :param dt: float, length of timestep
        :param duration: float, total time to transport particles in seconds
        :param D: float, diffusion coefficient
        saves nd array representing the positions over time for sets of particles released at a single location dt apart
        """
        # Construct random number generator
        rng = np.random.default_rng(seed=seed)

        # Define number of frames and list of times
        dt = self.dt
        duration = self.duration
        n_frames = abs(int(duration / dt))  
        t_list = np.linspace(self.t0, self.t0 + duration, n_frames+1)

        # Extract relevant variables
        src_loc = self.odorsource.osrc_loc
        D = self.odorsource.D_osrc

        # initialize array of particle trajectory data
        # stack of matrices with dimensions: frame, particle release time, x index, y index
        trajectories = np.empty((n_frames+1, 3, n_particles*(n_frames+1)), dtype=np.float32)
        trajectories[:] = np.nan
        # trajectories[:, :, 0] = np.repeat(t_list, n_particles)

        start_idx = 0
        end_idx = n_particles

        trajectories[:, 0, :] = np.repeat(t_list[:], n_particles)

        # at each timestep, release particles and transport all particles in domain using advection and random walk diffusion 
        for step in range(n_frames):
            tstep = t_list[step][0]

            # seed new particles at source location
            # trajectories[step, 0, start_idx:end_idx] = tstep
            trajectories[step, 1, start_idx:end_idx] = src_loc[1]
            trajectories[step, 2, start_idx:end_idx] = src_loc[0]
            loc_in = trajectories[step, 1:3, 0:end_idx]

            # numerical advection & diffusion of all particles
            if method=='IE':
                loc_out = self.flowfield.improvedEuler_singlestep(dt, tstep, loc_in) + np.sqrt(2 * D * dt) * rng.random(loc_in.shape)
            elif method=='RK4':
                loc_out = self.flowfield.rk4singlestep(dt, tstep, loc_in) + np.sqrt(2 * D * dt) * rng.random(loc_in.shape)

            # save position of each particle after this step
            trajectories[step+1, 1:3, 0:end_idx] = loc_out

            # If particle has left the domain, set x and y position to Nan
            mask = ((trajectories[step+1, 1, :] > 0.5) | (trajectories[step+1, 2, :] > 0.211) | (trajectories[step+1, 2, :] < -0.211))
            trajectories[step+1, 1:3, mask] = np.nan
            
            # shift indices to include next batch of particles starting at (0, 0)
            start_idx = end_idx
            end_idx += n_particles

        self.trajectories = trajectories

    def plot_trajectories(self, filepath, frames, domain_width, domain_length, dpi=300, movie=False):
        """

        if movie=False, saves individual plots of particle locations for each frame 
        if movie=True, all frames are saved together as an .mp4 file
        """
        if movie:
            fig, ax = plt.subplots()

            positions = plt.scatter(self.trajectories[frames[0], 1, :], self.trajectories[frames[0], 2, :], s=20, alpha=0.6)

            # Plotting configuration
            ax.set(xlim=[0, domain_length], ylim=[0-domain_width/2, domain_width/2], xlabel='x', ylabel='y')
            ax.set_aspect('equal', adjustable='box')

            def animate(frame):
                positions.set_offsets(self.trajectories[frame, 1:3, :].T)
                return positions,
    
            # use FuncAnimation from matplotlib
            ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(frames), blit=False)

            # save and show video
            f = filepath + '_movie.mp4'
            writervideo = animation.FFMpegWriter(fps=50)
            ani.save(f, writer=writervideo)

        else:
            for frame in frames:
                plt.close()
                fig, ax = plt.subplots()
                plt.scatter(self.trajectories[frame, 1, :], self.trajectories[frame, 2, :])
                plt.xlim(0, 0.5)
                plt.ylim(-0.211, 0.211)
                f_name = filepath + f'_frame{frame}.png'
                plt.savefig(f_name, dpi=dpi)


    
    