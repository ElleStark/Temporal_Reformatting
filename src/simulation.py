



class Simulation:
    def __init__(self, flowfield, odorsource, duration, t0, dt) -> None:
        pass

    def track_particles_rw(self, n_particles, ic_idx_1, ic_idx_2, dt, duration, D, Lb, scheme = 'IE'):
        """
        Uses Lagrangian particle tracking model with random walk diffusion to calculate particle positions over time
        for two 'blobs' of particles initialized at two different locations.
        :param n_particles: float, number of particles to track
        :param ic_idx_1: list [x, y] of center of particle group 1 (will be colored red)
        :param ic_idx_2: list [x, y] of center of particle group 1 (will be colored blue)
        :param dt: float, length of timestep
        :param duration: float, total time to transport particles
        :param D: float, diffusion coefficient
        :return: two nd arrays, each representing the positions over time for one 'blob' (set of particles)
        """

        L = abs(int(duration / dt))  # need to calculate if dt definition is not based on T
        #nx = len(self.x[0, :])
        #ny = len(self.y[:, 0])
        nx = 100
        ny = 50

        # Se up initial conditions for particles in both 'blobs'
        # Even concentration of particles in square of size (batchelor scale x batchelor scale)
        # Calculations result in a 'rounding' of the number of particles to make the square
        square_length = ceil(sqrt(n_particles))
        n_particles = square_length**2

        # Blob 1
        blob1 = np.zeros((2, n_particles))
        x_idxs1 = np.linspace(ic_idx_1[0] - Lb/2, ic_idx_1[0] + Lb/2, square_length)
        y_idxs1 = np.linspace(ic_idx_1[1] - Lb/2, ic_idx_1[1] + Lb/2, square_length)
        x_ic1, y_ic1 = np.meshgrid(x_idxs1, y_idxs1)
        blob1[0, :] = x_ic1.reshape(n_particles)
        blob1[1, :] = y_ic1.reshape(n_particles)

        # Blob 2
        blob2 = np.zeros((2, n_particles))
        x_idxs2 = np.linspace(ic_idx_2[0] - Lb/2, ic_idx_2[0] + Lb/2, square_length)
        y_idxs2 = np.linspace(ic_idx_2[1] - Lb/2, ic_idx_2[1] + Lb/2, square_length)
        x_ic2, y_ic2 = np.meshgrid(x_idxs2, y_idxs2)
        blob2[0, :] = x_ic2.reshape(n_particles)
        blob2[1, :] = y_ic2.reshape(n_particles)

        # at each timestep, advect particles and add diffusion with random walk
        blob1_single_steps = np.zeros((2, L, n_particles))
        blob2_single_steps = np.zeros((2, L, n_particles))

        for step in range(L):
            tstep = step * dt

            # Blob 1 (red blob) - particle positions
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
    
    