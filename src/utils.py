# Utility functions for two-particle simulations for temporal reformatting study
# Elle Stark May 2024

import numpy as np
import scipy.io

# Script to save .npy as .mat with relevant metadata
data = np.load('ignore/tests/particleTracking_n1_fullsim.npy')

f_path = 'ignore/tests/ParticleTracking_MSPlumeSim_n1_t60s.mat'
scipy.io.savemat(f_path, {'data': data, 'meta':{'ParticleTrackingParams':{'num_particles': '50 seeded each frame', 'num_frames': '3000', 'dt': '0.02 sec', 'duration': '60 sec', 'diffusionCoefficient': '1.5E(-5)', 'gridResolution': '0.0005 meter', 'ParticleReleasePoint': '(0, 0)', 'NumericalAdvectionMethod': 'Improved Euler'}, 
                                                'FlowfieldSimulationInfo':{'description':'2D grid turbulence Comsol model', 'source': 'Fisher Plume manuscript Tootoonian et al., 2024', 'meanVelocity': '10 cm/s', 'xDomain': '[0, 0.5] meters', 'yDomain': '[-0.211, 0.211] meters'}, 
                                                'FileCreationInfo': {'creationDate': 'April 2024', 'createdBy': 'Elle Stark, EFD Lab, CU Boulder CEAE Dept', 'contact': 'elle.stark@colorado.edu or aaron.true@colorado.edu'}}})

