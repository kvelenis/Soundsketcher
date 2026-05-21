export const rawFeatureNames = [
    "spectral_centroid",
    "weighted_spectral_centroid",
    "crepe_f0",
    "yin_f0_librosa",
    "perceived_pitch_f0_or_SC_weighted",
    "loudness",
    "loudness_periodicity",
    "loudness_pitchConf",
    "sharpness",
    "mir_mps_roughness",
    "mir_sharpness_zwicker",
    "mir_roughness_vassilakis",
    "fullness",
    "yin_periodicity",
    "crepe_confidence",
];

export const visibleFeatureNames = [
    "Spectral Centroid",
    "Weighted Spectral Centroid",
    "F0 | Crepe",
    "Yin F0 | Librosa",
    "F0 Or SC weighted | CREPE - Periodicity",
    "Loudness",
    "Loudness-Periodicity",
    "Loudness-CREPE Conf",
    "Sharpness | Mosqito",
    "MIR: MPS Roughness",
    "MIR: Sharpness | Zwicker",
    "MIR: Roughness | Vassilakis",
    "Fullness: Alluri & Toiviainen",
    "Periodicity",
    "F0 | CREPE Conf",
];

export const pitchFeatureNames = [
    "spectral_centroid",
    "weighted_spectral_centroid",
    "crepe_f0",
    "yin_f0_librosa",
    "perceived_pitch_f0_or_SC_weighted",
];

export const allowedLogFeatures = [
    "spectral_centroid",
    "weighted_spectral_centroid",
    "weighted_spectral_centroid_bandwidth",
    "spectral_peak",
    "centroid_peak_bandwidth",
    "centroid_peak_bandwidth_prominence",
    "multipeak_centroid",
    "yin_f0_librosa",
    "yin_f0_aubio",
    "crepe_f0",
    "perceived_pitch",
    "perceived_pitch_librosa",
    "perceived_pitch_crepe_periodicity",
    "perceived_pitch_librosa_periodicity",
    "perceived_pitch_f0_candidates_periodicity",
    "perceived_pitch_f0_or_SC",
    "perceived_pitch_f0_or_SC_weighted",
];

export const percentages = {
    "0.0%": 0,
    "20.0%": 0.2,
    "25.0%": 0.25,
    "33.3%": 1 / 3,
    "40.0%": 0.4,
    "50.0%": 0.5,
    "60.0%": 0.6,
    "66.6%": 2 / 3,
    "75.0%": 0.75,
    "80.0%": 0.8,
};

export const featureConfig = {
    rawFeatureNames,
    visibleFeatureNames,
    pitchFeatureNames,
    allowedLogFeatures,
    percentages,
};

window.SoundSketcherFeatureConfig = featureConfig;
if (window.SoundSketcher) {
    window.SoundSketcher.featureConfig = featureConfig;
}
