function [MPS_roughness, roughness_Zwicker, sharpness_Zwicker, ...
          roughness_vassilakis, roughness_sethares, Loudness_SC_Hz ...
          time_MPS, time_Zwicker, time_vassilakis, time_sethares] = roughnessTimeSeries(inputPath)

% Check if input is a folder or a single file
if isfolder(inputPath)
    % If input is a folder, process all supported audio files
    audioFiles = dir(fullfile(inputPath, '*.wav'));
    audioFiles = [audioFiles; dir(fullfile(inputPath, '*.mp3'))];
    audioFiles = [audioFiles; dir(fullfile(inputPath, '*.flac'))];
    
    if isempty(audioFiles)
        error('No audio files found in folder: %s', inputPath);
    end
    
    wavfiles = fullfile({audioFiles.folder}, {audioFiles.name});
else
    % Input is assumed to be a file
    if ~isfile(inputPath)
        error('File not found: %s', inputPath);
    end
    [~, ~, ext] = fileparts(inputPath);
    if ~ismember(lower(ext), {'.wav', '.mp3', '.flac'})
        error('Unsupported file type: %s', ext);
    end
    wavfiles = {inputPath};
end

% Sort naturally
% wavfiles = sort_nat(wavfiles)';
numFiles = length(wavfiles);

% calib = 98; %calibration level


%%%%% features of spectrogram calculation through the MIR Toolbox %%%%%%%%
winsize = 0.04;
% hop = winsize/2;

%%%%% Define the frequency range for energy calculation %%%%
freqMin = 30;  % Lower bound of frequency (30 Hz)
freqMax = 150;  % Upper bound of frequency (150 Hz)

% Initialize arrays to store energy values and corresponding labels
% MPS_roughness = [];  % Total energy array
% roughness = [];
% sharpness = [];
energyLabels = {};  % Array to store the labels (sound names)

% Preallocate a vector to store total energy for each sound
MPS_roughness = cell(length(wavfiles), 1);
roughness_Zwicker = cell(length(wavfiles), 1);
sharpness_Zwicker = cell(length(wavfiles), 1);
roughness_vassilakis = cell(numFiles, 1);
roughness_sethares = cell(numFiles, 1);
time_vassilakis = cell(numFiles, 1);
time_sethares   = cell(numFiles, 1);
Loudness_SC_Hz = cell(numFiles, 1);


% Initialize an array to store the energies and their corresponding labels

 % Set the duration to extract (0.3 seconds)
startTime = 0.001;  % Start time in seconds
endTime = 0.25;    % End time in seconds

timeStep = 2e-3; % 2 ms hop size (default for ISO 532-1) - acousticLoudness
fs_mod = 1 / timeStep; % Compute modulation sampling rate
fs = 44100; % Sampling frequency

barkBins = 0.1:0.1:24;
hzBins = bark2hz(barkBins);
barkCenters = barkBins;

%%

for i = 1:length(wavfiles)
    % Combine folder path with file name
    filepath = wavfiles{i};  % Already a full path    
    
    % Check if the file exists
    if ~isfile(filepath)
        error('File not found: %s', filepath);
    end
    
    % Read the audio file
 
% Read the first ... seconds of the audio file
    % [a, fs] = audioread(filepath, [round(startTime * fs), round(endTime * fs)]);  
        [a, fs] = audioread(filepath);    


    % Check if the audio is stereo
if size(a, 2) > 1
    a = mean(a, 2); % Convert stereo to mono by averaging the channels
end
    

% Compute roughness using the MIR Toolbox
    mirR_1 = mirroughness(filepath, 'Frame', winsize, 's', 50 , '%', 'Vassilakis');  
    roughness_vassilakis_ = mirgetdata(mirR_1);

    hop_mir = winsize / 2; % 50% overlap assumed
    nFrames = length(roughness_vassilakis_);
    time_mir{i} = (0:nFrames-1) * hop_mir;

     mirR_2 = mirroughness(filepath, 'Frame', winsize, 's', 50 , '%', 'Sethares');  
    roughness_sethares_ = mirgetdata(mirR_2);

    hop_mir = winsize / 2; % 50% overlap
    nFrames_v = length(roughness_vassilakis_);
    nFrames_s = length(roughness_sethares_);

    time_vassilakis{i} = (0:nFrames_v-1) * hop_mir;
    time_sethares{i}   = (0:nFrames_s-1) * hop_mir;

%%%%%% calculate time varying specific loudness using the Zwicker Loudness model %%%%%
    [~,specificLoudness] = acousticLoudness(a,fs,'TimeVarying', true);
    roughnessVec = acousticRoughness(specificLoudness);
    sharpnessVec = acousticSharpness(specificLoudness,'TimeVarying',true);
    time_Zwicker{i} = (0:length(roughnessVec)-1) * timeStep;

    

%%%%%% define spectral resolution %%%%%%%
    %Number of spectral bins (Bark bands)
N_spectralBins = size(specificLoudness, 1);

% Define Bark scale frequency limits
f_low = hzBins(1); % Lower bound (Hz)
f_high = hzBins(end); % Upper bound (Hz)

% Convert to octaves relative to 1 kHz
low_oct = log2(f_low / 1000); 
high_oct = log2(f_high / 1000);

% Compute frequency range in octaves
octave_range = high_oct - low_oct;

% Compute frequency spacing in cycles/octave
cycles_per_octave = N_spectralBins / octave_range;

% Display result
% fprintf('Number of spectral bins: %d\n', N_spectralBins);
% fprintf('Total range: %.2f octaves\n', octave_range);
% fprintf('Frequency spacing: %.2f cycles/octave\n', cycles_per_octave);

    %%%%%%% mirspectrum from MIR Toolbox %%%%%%%%%%%%
    % a = mean(a, 2); % Convert to mono
    % 
    % s = mirspectrum(miraudio(a), 'Frame', overlap, hop, 'Min', 20, 'MinRes', 5, 'Terhardt', 'Normalinput', 'MinGate', 0.001);
    % f = get(s, 'Frequency'); 
    % f = f{1}{1}(:,1)';
    % d = get(s, 'Magnitude');
    % d = d{1}{1};
    % d = d / length(f); % Normalize (verify if necessary)
    % 
    % nt = length(d(1,:));
    % t = (winsize * overlap) * (0:(nt - 1));
    % [T, F] = meshgrid(t, f);
    
    %plot spectrograms
    % 

    [s1, s2] = size(specificLoudness);


% % Define time axis

t = 0:2e-3:2e-3*(size(specificLoudness,1)-1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% window2D = hamming(size(specificLoudness,1)) * hamming(size(specificLoudness,2))';
% % window2D = ones(size(specificLoudness)); % No windowing
% 
% specificLoudness = specificLoudness .* window2D;

%%%% Interpolate to a Uniform Frequency Grid
linearHz = linspace(min(hzBins), max(hzBins), length(hzBins)); % Keep same size
specificLoudness = interp1(hzBins, specificLoudness.', linearHz, 'spline', 'extrap').';


% linearHz = linspace(min(hzBins), max(hzBins), size(specificLoudness,1));
% specificLoudness = interp1(hzBins, specificLoudness.', linearHz, 'spline', 'extrap').';

  
windowSize = 20; % number of time points in the window (20 × 2ms = 40ms)
nTimePoints = size(specificLoudness, 1);
nWindows = floor(nTimePoints / windowSize);

totalE = zeros(1, nWindows); % preallocate

for w = 1:nWindows
    idx_start = (w-1)*windowSize + 1;
    idx_end = w*windowSize;
    
    % Select the loudness slice for this window
    loudness_window = specificLoudness(idx_start:idx_end,:);
    
    % Calculate MPS
    mps_local = fftshift(fft2(loudness_window'));
    mpsPower_local = abs(mps_local).^2;
    
    % Get number of modulation frequency bins
    numFreqBins = size(mpsPower_local, 2);
    modFreqAxisFFT = linspace(-fs_mod/2, fs_mod/2, numFreqBins);
    
    % Define frequency indices
    freqIndices = (modFreqAxisFFT >= freqMin) & (modFreqAxisFFT <= freqMax);
    
    % Select region
    mpsRegion = mpsPower_local(:, freqIndices);
    
    % Calculate mean energy in selected region
    totalE(w) = mean(mpsRegion(:));
end
 
% Create time vector in seconds for this file
frameDuration = windowSize * timeStep; % e.g., 20 * 0.002 = 0.04s
time_MPS{i} = (0:nWindows-1) * frameDuration;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the total energy of the MPS within 30-150 Hz 
% Only for positive spectral frequencies (cycles/octave)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Get number of modulation frequency bins
% numFreqBins = size(mpsPower, 2);
% 
% % Define the true FFT-based modulation frequency axis
% modFreqAxisFFT = linspace(-fs_mod/2, fs_mod/2, numFreqBins);
% 
% % Get indices for the modulation frequency range 30-150 Hz
% freqIndices = (modFreqAxisFFT >= freqMin) & (modFreqAxisFFT <= freqMax);
% 
% % take the 0-30 Hz part of the x-axis
% freqIndices2 = (modFreqAxisFFT >= 1) & (modFreqAxisFFT <= freqMin); %from 1 to remove DC
% 
% % Define spectral frequency axis (cycles/octave)
% numSpectralBins = size(mpsPower, 1);
% spectralAxis = linspace(-5, 5, numSpectralBins);  % Adjust range if needed
% 
% % Keep only positive spectral frequencies
% positiveIdx = spectralAxis >= 0;
% mpsPositive = mpsPower(positiveIdx, :);  % Select positive part
% 
%     % Extract the MPS power values within the selected frequency range
%     mpsRegion = mpsPower(:, freqIndices);
%      % mpsRegion = mpsPower_n(:, freqIndices);
%     mpsRegion2 = mpsPower(:, freqIndices2);
% 
% 
%     % Calculate the total energy (sum) within this region by summing over the y-axis (spectral axis)
%     MPS_roughness(i) = mean(mpsRegion(:));  % Sum all values within the region
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Calculate the spectral centroid based on loudness and Bark bands
[~,specificLoudness] = acousticLoudness(a,fs,'TimeVarying', true);
% Normalize loudness per frame to avoid biasing the centroid
frameEnergy = sum(specificLoudness, 2);  % total loudness per frame
validFrames = frameEnergy > 0;

% Preallocate
spectralCentroids = zeros(size(specificLoudness, 1), 1);
spectralCentroids_Hz = zeros(size(spectralCentroids)); 

% Compute spectral centroid in Bark
for k = 1:size(specificLoudness, 1)
    if validFrames(k)
        spectrum = specificLoudness(k, :);
        spectralCentroids(k) = sum(barkCenters .* spectrum) / sum(spectrum);
    end
end

% Convert from Bark to Hz where possible
% for k = 1:length(spectralCentroids)
%     z = spectralCentroids(k);
%     if ~isnan(z) && z >= 0 && z < 26.28
%         spectralCentroids_Hz(k) = 1960 * (z / (26.28 - z));
%     else
%         spectralCentroids_Hz(k) = 0;  % Keep it 0 for invalid or out-of-range values
%     end
% end
spectralCentroids_Hz = bark2hz(spectralCentroids);
    % … compute roughnessVec (1×T), sharpnessVec (1×T), totalE (1×T) …
    roughness_Zwicker{i} = roughnessVec';
    sharpness_Zwicker{i} = sharpnessVec';
    MPS_roughness{i} = totalE; 
    roughness_vassilakis{i} = roughness_vassilakis_;
    roughness_sethares{i} = roughness_sethares_;
    Loudness_SC_Hz{i} = spectralCentroids_Hz';




end