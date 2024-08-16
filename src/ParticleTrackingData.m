% Script to process & combine flow state datasets using MATLAB
% And test storage requirements for data structure options for .mat file
% Elle Stark, May 2024

cd('C:/Users/elles/Documents/CU_Boulder/Fluids_Research/TemporalReformatting/Code/TwoParticleSims_TempReform/')

sim1data = load('ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_0to60s_D1.5_nanUpstream.mat');
sim2data = load('ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_60to120s_D1.5_nanUpstream.mat');
sim3data = load('ignore/ParticleTrackingData/ParticleTracking_sim_extended_n20_120to180s_D1.5_nanUpstream.mat');

% sim1matrix = sim1data.data(:);
% sim2matrix = sim2data.data(:);
% sim3matrix = sim3data.data(:);

% ParticleTracking(1) = sim1data;
% ParticleTracking(2) = sim2data;
% ParticleTracking(3) = sim3data;

combined_data = [sim1data.data(:, :, :); sim2data.data(:, :, :); sim3data.data(:, :, :)];
save('ExtendedSim_0to180sec_ParticleTracking_n20_D1.5.mat', 'combined_data', '-v7.3')

