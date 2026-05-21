function adjustPeriodicity(yin_periodicity, threshold = 0.85) {
    return yin_periodicity > threshold ? 1 : 0;
}

function adjustForSaturationPeriodicity(yin_periodicity, threshold = 0.3) {
    return yin_periodicity < threshold ? 0 : yin_periodicity;
}

function calibrateZCR(zcr, spectralCentroid, sampleRate = 44100, k = 1.0) {
    const nyquist = sampleRate / 2;
    const normalizedSC = spectralCentroid / nyquist;
    return zcr / (1 + k * normalizedSC);
}

function perceivedPitch(conf, f0, sc) {
    return f0 * conf + sc * (1 - conf) * 0.2;
}

function perceivedPitchLibrosa(conf, yinF0, sc) {
    return yinF0 * conf + sc * (1 - conf) * 0.2;
}

function perceivedPitchLibrosaPeriodicity(periodicity, yinF0, sc) {
    return yinF0 * periodicity + sc * (1 - periodicity) * 0.2;
}

function perceivedPitchCrepePeriodicity(periodicity, crepeF0, sc) {
    return crepeF0 * periodicity + sc * (1 - periodicity) * 0.2;
}

function perceivedPitchF0OrSC(periodicity, crepeF0, sc, threshold = 0.5, gamma = 1, division_value) {
    const g = Math.max(Number(gamma), 0.01);
    let pitchWeight = (threshold - periodicity) / threshold;
    pitchWeight = clamp(pitchWeight, 0, 1);
    pitchWeight = pitchWeight ** g;
    return crepeF0 * (1 - pitchWeight) + sc * pitchWeight * division_value;
}

function perceivedPitchF0Candidates(periodicity, f0Candidates, sc) {
    return f0Candidates * periodicity + sc * (1 - periodicity) * 0.2;
}

function spectralCentroidWithBandwidthWeight(centroidHz, bandwidthHz, beta = 1.0) {
    const eps = 1e-9;
    const centroid = Math.max(centroidHz, eps);
    return centroid / (1 + beta * (bandwidthHz / centroid));
}

function computeBlendedTonalY(localPeakHz, centroidHz, bandwidthHz) {
    const eps = 1e-9;
    const centroid = Math.max(centroidHz, eps);
    const bandwidth = Math.max(bandwidthHz, 0);
    const tonalness = centroid / (centroid + bandwidth + eps);
    return localPeakHz * tonalness + centroid * (1 - tonalness);
}

function computeTonalYWithProminenceDb({
    periodicity,
    crepeF0,
    spectralCentroidHz,
    spectralBandwidthHz,
    localPeakHz,
    prominenceDb,
}, options = {}) {
    const {
        prominenceCenter = 45,
        prominenceSlope = 0.8,
        toggleThreshold = 0.85,
    } = options;

    const eps = 1e-9;
    const centroid = Math.max(spectralCentroidHz, eps);
    const bandwidth = Math.max(spectralBandwidthHz, 0);
    const tonalness = centroid / (centroid + bandwidth + eps);
    const x = prominenceSlope * (prominenceDb - prominenceCenter);
    const prominenceWeight = 1 / (1 + Math.exp(-x));
    const trust = tonalness * prominenceWeight * Math.max(0, Math.min(periodicity, 1));
    const toggle = periodicity > toggleThreshold ? 1 : 0;
    const basePitch = crepeF0 * toggle + centroid * (1 - toggle) * 0.3;

    return localPeakHz * trust + basePitch * (1 - trust);
}
