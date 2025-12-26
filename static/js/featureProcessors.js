// featureProcessors.js

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

//! Some temp changes here
function perceivedPitchF0OrSC(periodicity,crepeF0,sc,threshold = 0.5,gamma = 1)
{
	// const toggle = adjustPeriodicity(periodicity,threshold);
	// return crepeF0 * toggle + sc * (1 - toggle) * 0.6;

	let P = (threshold - periodicity)/(threshold)
	P = clamp(P,0,1);
	P = P**gamma;
	return crepeF0*(1 - P) + sc*P*0.4
}

function perceivedPitchF0Candidates(periodicity, f0Candidates, sc) {
  return f0Candidates * periodicity + sc * (1 - periodicity) * 0.2;
}

function spectralCentroidWithBandwidthWeight(centroidHz, bandwidthHz, beta = 1.0) {
  const eps = 1e-9;
  const C = Math.max(centroidHz, eps);
  return C / (1 + beta * (bandwidthHz / C));
}

// function computeTonalness(centroidHz, bandwidthHz) {
//   const eps = 1e-9; // avoid division by zero
//   return centroidHz / (centroidHz + bandwidthHz + eps);
// }

function computeBlendedTonalY(localPeakHz, centroidHz, bandwidthHz) {
  const eps = 1e-9;
  const C = Math.max(centroidHz, eps);
  const B = Math.max(bandwidthHz, 0);

  const tonalness = C / (C + B + eps);
  const Y = localPeakHz * tonalness + C * (1 - tonalness);

  return Y;
}

// function computeTonalYWithProminenceDb(localPeakHz, centroidHz, bandwidthHz, prominenceDb, options = {}) {
//   const {
//     prominenceCenter = 18,  // dB where prominence weight = 0.5
//     prominenceSlope = 0.15   // how steep the sigmoid is
//   } = options;

//   const eps = 1e-9;
//   const C = Math.max(centroidHz, eps);
//   const B = Math.max(bandwidthHz, 0);

//   // Tonalness based on bandwidth
//   const tonalness = C / (C + B + eps);

//   // Sigmoid prominence weight from dB
//   const x = prominenceSlope * (prominenceDb - prominenceCenter);
//   const prominenceWeight = 1 / (1 + Math.exp(-x));

//   // Final combined confidence
//   //const trust = tonalness * prominenceWeight;
//   const trust = prominenceWeight;

//   // Blended Y-axis value
//   const Y = localPeakHz * trust + C * (1 - trust);
//   return Y;
// }

function computeTonalYWithProminenceDb({
  periodicity,
  crepeF0,
  spectralCentroidHz,
  spectralBandwidthHz,
  localPeakHz,
  prominenceDb
}, options = {}) {
  const {
    prominenceCenter = 45,   // center of sigmoid (dB)
    prominenceSlope = 0.8,  // sigmoid steepness
    toggleThreshold = 0.85   // periodicity threshold
  } = options;
  
  const eps = 1e-9;
  const C = Math.max(spectralCentroidHz, eps);
  const B = Math.max(spectralBandwidthHz, 0);

  // Step 1: Tonalness from centroid/bandwidth
  const tonalness = C / (C + B + eps);

  // Step 2: Sigmoid prominence weight
  const x = prominenceSlope * (prominenceDb - prominenceCenter);
  const prominenceWeight = 1 / (1 + Math.exp(-x));

  // Step 3: Final trust
  const trust = tonalness * prominenceWeight * Math.max(0, Math.min(periodicity, 1));

  // Step 4: Base pitch from F0 or centroid fallback
  const toggle = periodicity > toggleThreshold ? 1 : 0;
  const basePitch = crepeF0 * toggle + C * (1 - toggle) * 0.3;

  // Step 5: Final perceptual Y (blending local peak and base pitch)
  const Y = localPeakHz * trust + basePitch * (1 - trust);

  return Y;
}
