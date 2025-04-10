# Save computation of DLL - the longitudinal component of the Eulerian 2nd order structure function - for possible future use
# Would likely need to parallelize & run on HPC resources - running the below code within the PairSeparations script did not finish in over 48 hrs on my laptop.


# find separation vectors for particle pairs
# x_diffs = particle_matrix[t0 + compute_t, 1, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 1, (t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]
# y_diffs = particle_matrix[t0 + compute_t, 2, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 2, (t*20 + 20*delta_release):(t*20 + 20*delta_release +20)]

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# PLOT ENERGY SPECTRUM ON 2PI OMEGA AXIS

energies = loadmat('ignore/spectra_data/Pspectrum_normalized_det3_total_omega.mat')['det3_flowSpectrum']
spectrum = energies[0]

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
box_radius = len(spectrum)

# === Generate isotropic k values for bins ===
k_vals = np.linspace(0, k_max, box_radius)
om_vals = k_vals*0.1 / (2*np.pi)

# === Normalize and plot ===
norm_power = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

plt.figure()
plt.title("Velocity Power Spectrum (streamwise)")
plt.xlabel(r"Frequency $\omega$ (Hz)")
plt.ylabel("Normalized Power")
plt.semilogy(om_vals[:50], norm_power[:50], 'k')
plt.grid(True)
plt.show()

# omegas = np.array(range(int(len(energies)))) * 0.1 * 2 * np.pi / Lx
# sensor = 1

# plt.semilogy(omegas, energies)
# plt.xlim((0, 12))
# plt.ylim((0.0001, 10))
# plt.show()

