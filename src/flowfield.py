# FlowField class
# Elle Stark May 2024

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq
from scipy.signal import stft


class FlowField:

    def __init__(self, xmesh, ymesh, u_data, v_data, xmesh_uv, ymesh_uv, dt_uv):
        super().__init__()

        self.x = xmesh
        self.y = ymesh
        self.u_data = u_data
        self.v_data = v_data
        self.xmesh_uv = xmesh_uv
        self.ymesh_uv = ymesh_uv
        self.dt_uv = dt_uv

    def improvedEuler_singlestep(self, dt, t0, y0):
        """
        Single step of 2nd-order improved Euler integration. vfield must be a function that returns an array of [u, v] values
        :param u_field: nd_array of velocity data (time, x position, y position)
        :param dt: scalar value of desired time step
        :param t0: scalar start time for integration
        :param y0: nd_array starting position of particles, can be matrix of x, y positions
        :return: nd_array of final position of particles
        """
        # get the slopes at the initial and end points
        f1 = self.u_field(t0, y0)
        f2 = self.u_field(t0 + dt, y0 + dt * f1)
        y_out = y0 + dt / 2 * (f1 + f2)

        return y_out

    def rk4singlestep(self, dt, t0, y0):
        """
        Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
        vectorized computation of bundle of initial conditions. Reference: https://www.youtube.com/watch?v=LRF4dGP4xeo
        Note that u_field must be a function that returns an array of [u, v] values
        :param u_field: nd_array of velocity data (time, x position, y position)
        :param dt: scalar value of desired time step
        :param t0: start time for integration
        :param y0: starting position of particles
        :return: final position of particles
        """
        # RK4 first computes velocity at full steps and partial steps
        f1 = self.u_field(t0, y0)
        f2 = self.u_field(t0 + dt / 2, y0 + (dt / 2) * f1)
        f3 = self.u_field(t0 + dt / 2, y0 + (dt / 2) * f2)
        f4 = self.u_field(t0 + dt, y0 + dt * f3)
        # RK4 then takes a weighted average to move the particle
        y_out = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return y_out
    
    def u_field(self, time, y, flipuv=True):
        """
        Calculates velocity field based on interpolation from existing data.
        :param y: array of particle locations where y[0] is array of x locations and y[1] is array of y locations
        :param time: scalar value for time
        :return: array of u and v, where u is size x by y ndarray of horizontal velocity magnitudes,
        and v is size x by y ndarray of vertical velocity magnitudes.
        """
        # Convert from time to frame
        frame = int(time / self.dt_uv)

        # vector of x values
        xmesh_vec = self.xmesh_uv[0, :]

        # Set up interpolation functions
        # can use cubic interpolation for continuity of the between the segments (improve smoothness)
        # set bounds_error=False to allow particles to go outside the domain by extrapolation
        if flipuv: 
            # axes must be in ascending order, so need to flip y-axis, which also means flipping u and v upside-down
            ymesh_vec = np.flipud(self.ymesh_uv)[:, 0]
            u_matrix = np.squeeze(np.flipud(self.u_data[:, :, frame]))
            v_matrix = np.squeeze(np.flipud(self.v_data[:, :, frame]))
        else: 
            ymesh_vec = self.ymesh_uv[0, :]
            u_matrix = np.squeeze(self.u_data[:, :, frame])
            v_matrix = np.squeeze(self.v_data[:, :, frame])

        u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), u_matrix,
                                           method='linear', bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), v_matrix,
                                           method='linear', bounds_error=False, fill_value=None)

        # Interpolate u and v values at desired x (y[0]) and y (y[1]) points
        u = u_interp((y[1], y[0]))
        v = v_interp((y[1], y[0]))

        vfield = np.array([u, v])

        return vfield
    
    def compute_ufields(self, t):
        """
        Computes spatial velocity field for list of desired times
        :param t: ndarray of time values at which velocity field will be calculated
        :return: dictionary of velocity fields, one for each time value.
                 Each velocity field is a list of 4 ndarrays: [x, y, u, v].
        """
        vfields = []

        # Loop through time, assigning velocity field [x, y, u, v] for each t
        for time in t:
            vfield = self.u_field(time, [self.x, self.y])
            # need to extract u and v from vfield array
            u = vfield[0]
            v = vfield[1]
            vfield = [self.x, self.y, u, v]
            vfields.append(vfield)
        vfield_dict = dict(zip(t, vfields))

        self.velocity_fields = vfield_dict

    def find_plot_psd(self, xlim, ylim, plot=True):
        """
        Computes vortex shedding frequency of cylinder array by computing the fundamental frequency at each location across the input, 
        based on the Direct Fourier Transform of the time series of u velocity. 
        :param xlim: array [min, max] of index for x limits
        :param ylim: array [min, max] of index for y limits
        :param plot: boolean, if True, plots fft for random sample of 50 locations
        """
        # Define number of sample points N and sample spacing T
        N = len(self.u_data[0, 0, :])
        T = self.dt_uv
        nx = xlim[1] - xlim[0]
        ny = ylim[1] - ylim[0]

        xidxs = list(range(xlim[0], xlim[1]))
        yidxs = list(range(ylim[0], ylim[1]))

        # fft_list = np.empty((nx*ny, 1))
        fft_list = []
        # at each point, compute Direct Fourier Transform using numpy's fft() function
        for y in yidxs:
            for x in xidxs:
                u_timeseries = self.u_data[y, x, :]
                u_fluct = u_timeseries - np.mean(u_timeseries)
                # Compute autocorrelation coefficients for u velocity
                # r_u = np.correlate(u_fluct, u_fluct, mode='full')
                # r_u = r_u[r_u.size//2:]        
                
                # u_freq = fft(u_fluct)
                f, t, Zxx = stft(u_fluct, fs=0.02, nperseg=50, scaling='psd')
                df, dt = f[1] - f[0], t[1] - t[0]
                psd = np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df
                # psd = np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df * dt
                # u_freq_power = np.square(fft(u_fluct))
                # fft_list.append(u_freq)
                
                # QC: time series of u data
                plt.close()
                fig, ax = plt.subplots(figsize=(12, 5))
                plt.plot(u_fluct)
                plt.savefig('ignore/tests/stft_u_timeseries473.png', dpi=300)

        # x_vals = np.linspace(0.0, N*T, N, endpoint=False)
        # x_freq = fftfreq(N, T)[:N//2]

        if plot:
            plt.close()
            plt.plot(psd)
            plt.savefig('ignore/tests/psd_test.png', dpi=300)

            # plt.plot(x_freq, 2.0/N * np.abs(u_freq[:N//2]))
            # # plt.plot(x_freq, 2.0/N * np.abs(u_freq_power[:N//2]), alpha=0.5)
            # plt.xlim(0, 5)
            # plt.savefig('ignore/tests/fft_test473.png', dpi=300)
                

                


