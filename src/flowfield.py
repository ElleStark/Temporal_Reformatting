# FlowField class
# Elle Stark May 2024

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq
import scipy.signal as sp

class FlowField:

    def __init__(self, xmesh, ymesh, xmesh_uv, ymesh_uv, dt_uv, xlim, ylim):
        super().__init__()

        self.x = xmesh
        self.y = ymesh
        # self.u_data = u_data
        # self.v_data = v_data
        self.xlims = xlim
        self.ylims = ylim
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

        # read in u and v data from h5
        f_name = 'D:/singlesource_2d_extended/180to360/Re100_0_5mm_50Hz_singlesource_2d_180to360s.h5'
        with h5py.File(f_name, 'r') as f:
            u_data = f.get('Flow Data/u')[frame, self.xlims, self.ylims].T
            v_data = f.get('Flow Data/v')[frame, self.xlims, self.ylims].T

        # Set up interpolation functions
        # can use cubic interpolation for continuity of the between the segments (improve smoothness)
        # set bounds_error=False to allow particles to go outside the domain by extrapolation
        if flipuv: 
            # axes must be in ascending order, so need to flip y-axis, which also means flipping u and v upside-down
            ymesh_vec = np.flipud(self.ymesh_uv)[:, 0]
            u_matrix = np.squeeze(np.flipud(u_data))
            v_matrix = np.squeeze(np.flipud(v_data))
        else: 
            ymesh_vec = self.ymesh_uv[0, :]
            u_matrix = np.squeeze(u_data)
            v_matrix = np.squeeze(v_data)

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

    def find_plot_esd(self, u_data, v_data):

        # data = np.loadtxt('C:/Users/elles/Downloads/velocityfld_ascii.dat', skiprows=2)
        # with open('C:/Users/elles/Downloads/velocityfld_ascii.dat', 'r') as f:
        #     first_line = f.readline()
        #     second_line = f.readline()
        # print("shape of data = ",data.shape)

        eps = 1e-50 # to void log(0)
        N = int(u_data.shape[1] * u_data.shape[2])
        amplsU = np.fft.fftn(u_data, axes=(1, 2))
        amplsV = np.fft.fftn(v_data, axes=(1, 2))
        # amplsU = np.mean(amplsU, axis=0)
        # amplsV = np.mean(amplsV, axis=0)
        # Lx = 0.01  # not sure if domain distance length scale
        Lx = 1

        EK_U  = np.abs(amplsU)**2*2/(N)
        EK_V  = np.abs(amplsV)**2*2/(N)  
        EK_U = np.mean(EK_U, axis=0)
        EK_V = np.mean(EK_V, axis=0)
        
        EK_U = np.fft.fftshift(EK_U)
        EK_V = np.fft.fftshift(EK_V)

        # nx, ny = u_data.shape[1], u_data.shape[2]
        # kx = np.fft.fftfreq(nx, d=dx).reshape(-1, 1)  # Wavenumbers in x
        # ky = np.fft.fftfreq(ny, d=dx).reshape(1, -1)  # Wavenumbers in y
        # k = np.sqrt(kx**2 + ky**2)  # Radial wavenumber

        # # Flatten the arrays
        # k_flat = k.flatten()
        # EK_U_flat = EK_U.flatten()
        # EK_V_flat = EK_V.flatten()

        # # Define bins for wavenumbers
        # num_bins = 100
        # k_bins = np.linspace(0, np.max(k), num_bins)
        # k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

        # # Compute radial average
        # power_spectrum_u = np.zeros(num_bins - 1)
        # power_spectrum_v = np.zeros(num_bins - 1)
        # for i in range(num_bins -1):
        #     mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
        #     if np.sum(mask) > 0:
        #         power_spectrum_u[i] = np.sum(EK_U_flat[mask]) / np.sum(mask)
        #         power_spectrum_v[i] = np.sum(EK_V_flat[mask]) / np.sum(mask)

        # dk = np.diff(k_bins)  # Bin widths in k-space
        # total_energy_u = np.sum(power_spectrum_u * dk)
        # total_energy_v = np.sum(power_spectrum_v * dk)

        # EK_avsphr = 0.5*(power_spectrum_u + power_spectrum_v)

        # # Plot results
        # plt.loglog(k_bin_centers, EK_avsphr, label='Power Spectrum')
        # plt.loglog(k_bin_centers, power_spectrum_u, label='u component')
        # plt.loglog(k_bin_centers, power_spectrum_v, label='v component')
        # plt.loglog(k_bin_centers, k_bin_centers**(-5/3), '--', label='$k^{-5/3}$')
        # plt.legend()
        # plt.show()



        ## Time avg of energy spectrum
        ## t_avg_EK_U = np.mean(EK_U, axis=0)
        ## t_avg_EK_V = np.mean(EK_V, axis=0)
        ## x_avg_EK_U = np.mean(EK_U)


        sign_sizex = np.shape(EK_U)[0]
        sign_sizey = np.shape(EK_U)[1]

        box_sidex = sign_sizex
        box_sidey = sign_sizey

        # box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)
        box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

        centerx = int(box_sidex/2)
        centery = int(box_sidey/2)
                        
        EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
        EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius

        for i in range(box_sidex):
            for j in range(box_sidey):            
                wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
                EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]
                EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j]
            print(f'row{i} of {box_sidex} complete.')            

        EK_avsphr = 1.5*(EK_U_avsphr + EK_V_avsphr)

        plt.close()                        
        fig = plt.figure()
        plt.title("Kinetic Energy Spectrum")
        plt.xlabel(r"k (wavenumber)")
        plt.ylabel(r"TKE of the k$^{th}$ wavenumber")

        realsize = len(np.fft.rfft(u_data[0,0,:]))
        # plt.loglog(np.arange(0,realsize),((EK_avsphr[0:realsize] )),'k')
        plt.loglog(np.arange(0,realsize),((EK_U_avsphr[0:realsize] )),'g')
        plt.loglog(np.arange(0,realsize),((EK_V_avsphr[0:realsize] )),'b')
        # plt.loglog(np.arange(realsize,len(EK_avsphr),1),((EK_avsphr[realsize:] )),'k--')
        plt.loglog(np.arange(0,realsize),np.arange(0,realsize)**(-5/3),'r')
        plt.show()

        # realsize = len(np.fft.rfft(u_data[0,:,0]))
        # # realsize=502
        # k = np.arange(0,realsize)
        # kx = np.round((2*np.pi/np.arange(0, realsize)))
        # # plt.loglog(np.arange(0,realsize),((EK_avsphr[0:realsize] )),'k', label='combined')
        # plt.loglog(np.arange(0,realsize),(np.abs(EK_U_avsphr[0:realsize])),'r', label='u component')
        # # plt.loglog(np.asarray(kx[0:realsize]),(EK_U_avsphr[0:realsize])*2*np.pi*np.asarray(kx[0:realsize])**2,'r', label='u component')
        # # plt.loglog(np.arange(0,realsize),((EK_V_avsphr[0:realsize] )),'b', label='v component')
        # # plt.loglog(np.arange(realsize,len(EK_avsphr),1),((EK_avsphr[realsize:] )),'k--')
        # # plt.loglog(np.arange(realsize,len(EK_U_avsphr),1),((EK_U_avsphr[realsize:] )),'r--')
        # # # plt.loglog(np.arange(realsize,len(EK_V_avsphr),1),((EK_V_avsphr[realsize:] )),'b--')
        # plt.loglog(np.arange(0, realsize), (np.arange(0, realsize)+eps)**(-5/3)*0.001, 'g')
        # # plt.loglog(np.arange(0, realsize), (np.arange(0, realsize)+eps)**(-1)*0.001, 'orange')
        # plt.legend()
        # axes = plt.gca()
        # axes.set_ylim([10**-14,10**-3])

        # plt.show()


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
                # f, t, Zxx = stft(u_fluct, fs=0.02, nperseg=50, scaling='psd')
                # window = sp.windows.boxcar(50)
                # hop = 25
                fs = 50  # sampling frequency (Hz)
                # scale_to = 'psd'
                # SFT = sp.ShortTimeFFT(window, hop, fs=fs, scale_to = scale_to)
                # Sx = SFT.stft(u_fluct)  # Perform the STFT

                # f, Pxx_den = sp.periodogram(u_fluct, fs, scaling='density')
                # freqs = np.linspace(0.001, 100, 10000)
                ps = np.abs(np.fft.fft(u_fluct))**2
                timestep = 1/50
                freqs = np.fft.fftfreq(u_fluct.size, timestep)
                idx = np.argsort(freqs)

                ILS_guess = 0.015  # integral length scale, meters
                # u_avg = np.mean(u_timeseries)  # average velocity, m/s 
                u_avg = 0.10  # m/s
                fL = freqs * ILS_guess / u_avg # normalized frequencies
                vonkarman = (4*fL) / ((1 + 70.78*(fL**2))**(5/6)) 
                # plt.semilogy(f, Pxx_den)
                # plt.scatter(fL, Pxx_den * f / u_avg**2, label='modeled')
                plt.plot(ps / fs * freqs / np.var(u_fluct)**2, label='modeled')
                # plt.plot(freqs[idx], ps[idx] * freqs[idx] / np.var(u_fluct)**2, label='modeled')
                plt.plot(vonkarman, color='r', label='Von Karman fit')
                # plt.psd(u_timeseries, NFFT=150, Fs=fs, scale_by_freq=True, label='PSD')
                plt.yscale('log')
                plt.xscale('log')
                # plt.xlim(0, 100)
                # plt.ylim(10**(-10), 1)
                plt.xlabel('normalized frequency (Hz)')
                plt.ylabel('PSD [V**2/Hz]')
                plt.legend()
                plt.show()

                # df, dt = f[1] - f[0], t[1] - t[0]
                # psd = np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df
                # psd = np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df * dt
                # u_freq_power = np.square(fft(u_fluct))
                # fft_list.append(u_freq)
                
                # QC: time series of u data
                # plt.close()
                # fig, ax = plt.subplots(figsize=(12, 5))
                # plt.plot(u_fluct)
                # plt.savefig('ignore/tests/stft_u_timeseries473.png', dpi=300)

        # x_vals = np.linspace(0.0, N*T, N, endpoint=False)
        # x_freq = fftfreq(N, T)[:N//2]

        # if plot:

        #     # TEST PLOT FROM SCIPY EXAMPLE
        #     fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
        #     t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        #     ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
        #                 rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        #             ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
        #                 rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        #             xlim=(t_lo, t_hi))

        #     im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
        #                     extent=SFT.extent(N), cmap='viridis')
        #     # ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
        #     fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

        #     # Shade areas where window slices stick out to the side:
        #     for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
        #                     (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
        #         ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
        #     for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
        #         ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
        #     ax1.legend()
        #     fig1.tight_layout()
        #     plt.show()
            # plt.close()
            # plt.plot(psd)
            # plt.savefig('ignore/tests/psd_test.png', dpi=300)

            # plt.plot(x_freq, 2.0/N * np.abs(u_freq[:N//2]))
            # # plt.plot(x_freq, 2.0/N * np.abs(u_freq_power[:N//2]), alpha=0.5)
            # plt.xlim(0, 5)
            # plt.savefig('ignore/tests/fft_test473.png', dpi=300)
                

                


