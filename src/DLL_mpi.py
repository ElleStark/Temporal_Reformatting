# Save computation of DLL - the longitudinal component of the Eulerian 2nd order structure function - for possible future use
# Would likely need to parallelize & run on HPC resources - running the below code within the PairSeparations script did not finish in over 48 hrs on my laptop.


# find separation vectors for particle pairs
# x_diffs = particle_matrix[t0 + compute_t, 1, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 1, (t*20 + 20*delta_release):(t*20 + 20*delta_release + 20)]
# y_diffs = particle_matrix[t0 + compute_t, 2, (t*20-20):(t*20)] - particle_matrix[t0 + compute_t, 2, (t*20 + 20*delta_release):(t*20 + 20*delta_release +20)]


