function [spectralCentroids_Hz, time_vector] = SC_Loudness_single(audioFilePath)
% Computes loudness-based spectral centroid (in Hz) from a single audio file
% using Zwicker's time-varying specific loudness model.

% INPUT:
%   audioFilePath - full path to a single .wav, .mp3, or .flac file
% OUTPUT:
%   spectralCentroids_Hz - column vector of spectral centroid in Hz
%   time_vector - time in seconds for each centroid frame

% Validate file
if ~isfile(audioFilePath)
    error('File not found: %s', audioFilePath);
end

% Read audio
[a, fs] = audioread(audioFilePath);

% Convert to mono if stereo
if size(a, 2) > 1
    a = mean(a, 2);
end

% Bark scale bins
barkBins = 0.1 : 0.1 : 24;
barkCenters = barkBins;

% Zwicker-specific loudness
[~, specificLoudness] = acousticLoudness(a, fs, 'TimeVarying', true);

% Time vector (1 frame per 2.5 ms)
frameDuration = 0.0025; % seconds
numFrames = size(specificLoudness, 1);
time_vector = (0:numFrames-1)' * frameDuration;

% Frame-wise normalization and spectral centroid in Bark
frameEnergy = sum(specificLoudness, 2);
validFrames = frameEnergy > 0;

spectralCentroids_Bark = zeros(numFrames, 1);

for k = 1:numFrames
    if validFrames(k)
        spectrum = specificLoudness(k, :);
        spectralCentroids_Bark(k) = sum(barkCenters .* spectrum) / sum(spectrum);
    end
end

% Convert Bark → Hz
spectralCentroids_Hz = bark2hz(spectralCentroids_Bark);

end