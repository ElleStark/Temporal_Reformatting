% Script to process & combine flow state datasets using MATLAB
% And test storage requirements for data structure options for .mat file
% Elle Stark, May 2024

cd('C:/Users/elles/Documents/CU_Boulder/Fluids_Research/TemporalReformatting/Code/TwoParticleSims_TempReform/')

filedata = load('ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_180to360s_D5em5_RK4method.mat');
simdata = filedata.data;

ParticleTracking = struct;
ParticleTracking.data = simdata;
metadata = struct;
params = struct;
siminfo = struct;
fileinfo = struct;

params.num_particles = '20 seeded each frame';
params.num_frames = '9000';
params.dt = '0.02 sec';
params.duration = '180 sec';
params.diffusionCoefficient = '1.5e-5 m^2/s';
params.gridResolution = '0.0005 meter';
params.ParticleReleasePoint = '(0,0)';
params.NumericalAdvectionMethod = 'RK4';

siminfo.description = '2D grid turbulence Comsol model';
siminfo.source = 'Expanded domain of Tootoonian et al., 2025 simulations';
siminfo.meanVelocity = '10 cm/s';
siminfo.xDomain = '[0, 0.75] meters';
siminfo.yDomain = '[-0.3, 0.3] meters';

fileinfo.creationDate = 'March 2025';
fileinfo.createdBy = 'Elle Stark, EFD Lab, CU Boulder CEAE Dept';
fileinfo.contact = 'elle.stark@colorado.edu or aaron.true@colorado.edu';

metadata.ParticleTrackingParams = params;
metadata.FlowfieldSimulationInfo = siminfo;
metadata.FileCreationInfo = fileinfo;

ParticleTracking.meta = metadata;

save('ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_180s_D1.5em5_RK4.mat', 'ParticleTracking', '-v7.3')

