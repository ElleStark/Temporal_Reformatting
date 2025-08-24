# Script for manuscript Figure 2: evolution of spectrum from continuous source
# Elle Stark August 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# load previously computed concentration spectrum
energies = loadmat('ignore/spectra_data/Cspectrum_det7.mat')['det7_CSpectrum']
cspectrum7 = energies[0]
energies = loadmat('ignore/spectra_data/Cspectrum_det1.mat')['det1_CSpectrum']
cspectrum1 = energies[0]

# Construct wavenumber axis
Lx, Ly = 0.1, 0.4
dx, dy = 0.0005, 0.0005
nx, ny = int(Lx/dx), int(Ly/dy)

kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi  # units: rad/m
ky = np.fft.fftfreq(ny, d=dy) * 2*np.pi
kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)

kxm, kym = np.meshgrid(kx, ky, indexing='ij')
k_mag = np.sqrt(kxm**2 + kym**2)
k_max = np.max(k_mag)

# estimated number of radial bins
box_radius_c = len(cspectrum7)

# === Generate isotropic k values for bins ===
k_vals_c = np.linspace(0, k_max, box_radius_c)
om_vals_c = k_vals_c*0.1 / (2*np.pi)

# === Normalize and plot ===
norm_power_c7 = (cspectrum7 - np.min(cspectrum7)) / (np.max(cspectrum7) - np.min(cspectrum7))
norm_power_c1 = (cspectrum1 - np.min(cspectrum1)) / (np.max(cspectrum1) - np.min(cspectrum1))
norm_power_c7 = cspectrum7
norm_power_c1 = cspectrum1


plt.semilogy(om_vals_c[0:20], norm_power_c7[0:20], label='Concentration spectrum', color='#B85B51', linestyle='solid', linewidth=2)
# plt.semilogy(om_vals_c[0:20], norm_power_c1[0:20], label='Concentration spectrum', color='#588D9D', linestyle='solid', linewidth=2)

plt.show()