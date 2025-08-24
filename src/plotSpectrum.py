# Save computation of DLL - the longitudinal component of the Eulerian 2nd order structure function - for possible future use
# Would likely need to parallelize & run on HPC resources - running the below code within the PairSeparations script did not finish in over 48 hrs on my laptop.


# find separation vectors for particle pairs
# x_diffs = particle_matrix[t0 + compute_t, 1, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 1, (t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]
# y_diffs = particle_matrix[t0 + compute_t, 2, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 2, (t*20 + 20*delta_release):(t*20 + 20*delta_release +20)]

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# PLOT ENERGY SPECTRUM ON 2PI OMEGA AXIS

energies = loadmat('ignore/spectra_data/Cspectrum_det7.mat')['det7_CSpectrum']
cspectrum = energies[0]

u_energies = loadmat('ignore/spectra_data/Pspectrum_normalized_det7_total_omega')['det7_flowSpectrum']
uspectrum = u_energies[0]


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
box_radius_u = len(uspectrum)

# === Generate isotropic k values for bins ===
k_vals_u = np.linspace(0, k_max, box_radius_u)
om_vals_u = k_vals_u*0.1 / (2*np.pi)


# estimated number of radial bins
box_radius_c = len(cspectrum)

# === Generate isotropic k values for bins ===
k_vals_c = np.linspace(0, k_max, box_radius_c)
om_vals_c = k_vals_c*0.1 / (2*np.pi)

# === Normalize and plot ===
norm_power_c = (cspectrum - np.min(cspectrum)) / (np.max(cspectrum) - np.min(cspectrum))
norm_power_u = (uspectrum - np.min(uspectrum)) / (np.max(uspectrum) - np.min(uspectrum))

# plt.figure()
# plt.title("Scalar Power Spectrum")
# plt.xlabel(r"frequency $\omega$ (rad/s)")
# plt.ylabel("Normalized Power")
# plt.semilogy(om_vals[:50], norm_power_c[:50], 'k')
# plt.semilogy(om_vals[:50], energies[0, :50])
# #plt.grid(True)
# plt.show()
# print(k_vals)

plt.figure()
plt.title("Scalar Power Spectrum, sensor 7")
plt.xlabel(r"wavenumber $k$ (rad/m)")
plt.ylabel("Normalized Power")
plt.loglog(k_vals_c[:len(norm_power_c)], norm_power_c[:], 'k')
plt.loglog(k_vals_c[6:45], 50000000*(k_vals_c[6:45]**(-3.3)), 'r--')
plt.loglog(k_vals_c[3:6], 8000*(k_vals_c[3:6]**(-5/3)), 'b--')
plt.loglog(k_vals_c[3:6], (k_vals_c[3:6]**(-1)), 'c--')
plt.loglog(k_vals_u[:], norm_power_u[:len(k_vals_u)], 'g--')
plt.ylim(0.000004, 10)
#plt.grid(True)
plt.savefig('ignore/plots/det7_Cvsu_spectrum_k.png', dpi=600)
plt.show()


# #omegas = np.array(range(int(len(energies)))) * 0.1 * 2 * np.pi / Lx
# sensor = 1
# omegas = k_vals*0.1 / (2*np.pi)
# plt.semilogy(omegas, energies[0, :], 'k-')
# plt.semilogy(omegas[6:15], 1.2*(omegas[6:15]**(-3)), 'r--')
# plt.semilogy(omegas[3:6], 0.55*(omegas[3:6]**(-5/3)), 'b--')
# plt.ylabel("Normalized Power")
# plt.xlabel(r"frequency $f$ (Hz)")
# plt.xlim((0, 14))
# plt.ylim((0.001, 10))
# #plt.savefig('ignore/plots/det1_spectrum_omega_combinedv2.png', dpi=600)
# plt.show()


# # compute expected turbulent dispersion term
# # max_freqs = np.max(omegas) * (2*np.pi)
# # freqs1 = np.linspace(0, max_freqs, len(omegas))
# #t_disp = np.exp(-((omegas)**2* 9.7*10**(-4)) / (2*0.01))
# #t_disp = np.exp(-((omegas)**2* 1.2*10**(-2)) / (2*0.01))
# #t_disp = np.exp(-((omegas)**2* 8.3*10**(-8))) # computed from cutoff factor eqn, using sensor 1 data for D=10E-8
# #t_disp = np.exp(-(omegas**2* 1.2*10**(-6))) # computed from cutoff factor eqn, using sensor 7 data for D=10E-8
# t_disp = np.exp(-((omegas)**2* 1.5*10**(-8)*0.4) / (0.01))
# #t_disp = t_disp[0:102]
# #t_disp = ((t_disp - np.min(t_disp)) / (np.max(t_disp) - np.min(t_disp)))*(1-0.001)+0.001
# plt.semilogy(omegas[0:len(t_disp)], t_disp)
# plt.semilogy(omegas[0:len(t_disp)], energies[0, 0:len(t_disp)])
# plt.ylim((0.001, 10))
# plt.xlim(0, 14)
# plt.show()

# # multiply turbulent dispersion and spectrum to get total batchelor spectrum
# #energies = (energies-np.min(energies)) / (np.max(energies)-np.min(energies))

# b_spectrum = t_disp[:] * energies[0, 0:len(t_disp)]
# #total_spec = (t_disp[:] + energies[0, 0:len(t_disp)]) / np.max(t_disp[:]+energies[0, 0:len(t_disp)])
# # Plotting
# sensor = 1
# #omegas = k_vals*0.1/(2*np.pi)
# plt.semilogy(omegas[0:len(b_spectrum)], b_spectrum[:], 'k-')
# #plt.semilogy(omegas[6:15], 1.2*(omegas[6:15]**(-3)), 'r--')
# #plt.semilogy(omegas[3:6], 0.55*(omegas[3:6]**(-5/3)), 'b--')
# plt.ylabel("Normalized Batchelor spectrum")
# plt.xlabel(r"frequency $f$ (Hz)")
# plt.xlim((0, 14))
# plt.ylim((0.001, 10))
# #plt.savefig('ignore/plots/det1_spectrum_omega_combinedv2.png', dpi=600)
# plt.show()


# compare to remap underlying spectrum
freqs = np.linspace(0, 24.95, 500)
#print(freqs[0:15])
# load mat file for low diffusivity
data_dm8 = loadmat('C:/Users/elles/OneDrive - UCB-O365/Fluids_Research/LCS Project/TemporalReformatting/bdiffPCsAnalysis/u_v_data/remap_dm8_v2.mat')['remap']
data_dm8 = np.abs(np.array(data_dm8))

#print(data_dm8.size)
#data_dm81 = np.mean(data_dm8[:, 200:499, 6], axis=1)
#data_dm81 = np.log10(data_dm8[0:250, 239, 6])
data_dm81 = data_dm8[0:250, 239, 6]
data_dm82 = np.mean(data_dm8[:, 400:500, 6], axis=1)
data_dm80 = np.mean(data_dm8[:, 0:10, 6], axis=1)
#data_dm82 = data_dm8[0:250, 0, 6]
#data_dm8 = np.log10(np.abs(data_dm8))
#plt.semilogy(freqs, data_dm81)
#plt.plot(freqs[0:len(data_dm81)], data_dm81)
# plt.semilogy(freqs[0:len(data_dm81)], data_dm81)
# plt.show()

# # normalization?
# # data_dm81 = data_dm81 / np.max(data_dm81)
# data_dm82 = (data_dm82-min(data_dm82)) / (np.max(data_dm82)-np.min(data_dm82))

# # convolution
# # conv = np.convolve(t_disp, energies[0, :], 'same')
# # #conv = (conv-min(conv[0:50])) / (max(conv[0:50])-min(conv[0:50]))
# # plt.semilogy(omegas, conv)
# # #plt.xlim((0, 14))
# # #plt.ylim((0.0001, 1))
# # plt.show()




#plt.semilogy(freqs[:len(data_dm81)], data_dm81*10000, label='12 Hz')
plt.semilogy(freqs[:len(data_dm82)], data_dm80*30000, label='data-driven, 20to25Hz', color='#F6AF50', linewidth=3)

#plt.semilogy(omegas[0:len(t_disp)], t_disp, label='dispersion')
plt.semilogy(om_vals_c[0:len(data_dm82)], norm_power_c[0:len(data_dm82)], label='Concentration spectrum', color='#B85B51', linestyle='dashed', linewidth=2)
#plt.semilogy(om_vals_u[0:len(data_dm82)], norm_power_u[0:len(data_dm82)], label='velocity spectrum', color='#588D9D')
# plt.semilogy(freqs[:len(data_dm80)], data_dm80*5000, 'k-', label='data-driven, 0Hz')
plt.semilogy(om_vals_c[0:len(data_dm82)], 8*om_vals_c[0:len(data_dm82)]**(-10/3), label='-3.3 power law', color = 'k', linestyle='dotted', linewidth=2)


# plt.semilogy(freqs[:len(data_dm80)], data_dm80/norm_power_u, label='0Hz / velocity spectrum')
# plt.semilogy(freqs[:len(data_dm80)], (data_dm80*10000-data_dm82*15000), label='data-driven, 0Hz-20to25Hz')
#plt.semilogy(omegas[0:len(t_disp)], (t_disp[:]+energies[0, 0:len(t_disp)])/2, label='average')
# plt.semilogy(omegas[0:len(t_disp)], b_spectrum[:], 'k-', label='physics-based')
# plt.semilogy(omegas[0:len(t_disp)], conv[:], 'k-', label='convolved')
plt.xlim((0, 14))
plt.ylim((0.0001, 10))
plt.legend()
plt.title('Sensor 7 comparison: remapping, concentration, and power law spectra')
# plt.savefig('ignore/plots/det7_dm8_ubc_plaw_spectrum.png', dpi=600)
plt.show()


# # Normalize
# data_dm80 = 0.65*(data_dm80-np.min(data_dm80))/(np.max(data_dm80[40:]) - np.min(data_dm80))
# data_dm82 = (data_dm82-np.min(data_dm82))/(np.max(data_dm82) - np.min(data_dm82))


# plt.figure()
# plt.plot(freqs[:len(data_dm82)], data_dm82, label='data-driven, 20to25Hz')

# #plt.semilogy(omegas[0:len(t_disp)], t_disp, label='dispersion')
# plt.plot(om_vals_c[0:len(data_dm82)], norm_power_c[0:len(data_dm82)], label='C_spectrum')
# plt.plot(om_vals_u[0:len(data_dm82)], norm_power_u[0:len(data_dm82)], label='vel_spectrum')
# plt.plot(freqs[:len(data_dm80)], data_dm80, label='data-driven, 0Hz')
# # plt.semilogy(freqs[:len(data_dm80)], data_dm80/norm_power_u, label='0Hz / velocity spectrum')
# plt.plot(freqs[:len(data_dm80)], (data_dm80-data_dm82), label='data-driven, low Hz - high Hz', linestyle='dashed')
# #plt.plot(freqs[:len(data_dm80)], (data_dm82/data_dm80), label='data-driven, high Hz / low Hz', linestyle='dashed')
# #plt.semilogy(omegas[0:len(t_disp)], (t_disp[:]+energies[0, 0:len(t_disp)])/2, label='average')
# # plt.semilogy(omegas[0:len(t_disp)], b_spectrum[:], 'k-', label='physics-based')
# # plt.semilogy(omegas[0:len(t_disp)], conv[:], 'k-', label='convolved')
# plt.xlim((0, 15))
# plt.ylim((-0.25, 2))
# plt.legend()
# plt.title('Sensor 1 comparison: velocity, concentration, and remap spectra')
# #plt.savefig('ignore/plots/det1_dm8_ubc_0m20Hz_spectrum.png', dpi=600)
# plt.show()
