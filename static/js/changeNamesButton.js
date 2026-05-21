(function () {
    const originalNames = {
        spectral_centroid: "Spectral Centroid",
        spectral_flux: "Spectral Flux",
        spectral_deviation: "Spectral Standard Deviation",
        zerocrossingrate: "Zero Crossing Rate",
        amplitude: "Amplitude",
        yin_f0_librosa: "Yin F0",
        normalized_height: "(Spectral Centroid - Deviation)/2",
        none: "None",
    };

    const simplifiedNames = {
        spectral_centroid: "Center of Spectrum",
        spectral_flux: "Change in Spectrum",
        spectral_deviation: "Spectrum Variability",
        zerocrossingrate: "Noisiness",
        amplitude: "Volume Level",
        yin_f0_librosa: "Pitch Estimation (Yin)",
        normalized_height: "Normalized Spectral height",
        none: "No Feature",
    };

    window.SoundSketcherFeatureNameSets = {
        original: originalNames,
        simplified: simplifiedNames,
    };
})();
