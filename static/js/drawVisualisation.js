//! Moved this outside drawVisualization
function getMappedFeatureValue(feature,featureName,config,startRange,endRange,applySoftclip = false,softclipScale = 10,defaulValue = 0)
{
    if(featureName === "none")
    {
        return defaulValue;
    }

    const entry = config[featureName];
    if(!entry)
    {
        return 0;
    }

    const min = entry.min;
    const max = entry.max;

    const clamped = clamp(entry.val(feature),min,max);
    //! Softclip applies to same features as clamping
    if(applySoftclip && clampConfig[featureName])
    {
        const shift = (entry.median - min)/(max - min);
        return mapWithSoftClipping(clamped,min,max,startRange,endRange,shift,softclipScale)
    }
    else
    {
        return map(clamped,min,max,startRange,endRange);
    }
}

//! Light modifications
function getYAxisScaleFromConfig(featureConfig,featureName,scale = "linear") {

    const entry = featureConfig[featureName];
    if(!entry)
    {
        return null;
    }

    let min = entry.min;
    let max = entry.max;
    
    if(PITCH_FEATURES.has(featureName))
    {
        // min = Math.min(20,min); //! Added Math.min
        // max = Math.max(4000,max); //! Added Math.max
        // min = 20;
        // max = 4000;
    }

    const labelFormatter = scale !== "linear"
    ? (v) => (v < 10 ? v.toFixed(2) : v.toFixed(0))
    : (v) => v.toFixed(v < 10 ? 2 : 0);

    return { min, max, labelFormatter };
}

//! Merged computeYAxisValue & getMappedYAxisFallback
function computeYAxisValue(feature,featureName,config,height,padding,inverted,scale,min,max,applySoftclip = false,softclipScale = 10)
{
    let value = null;
    const entry = config[featureName];

    const mapFn = scale === "linear" ? mapToLinearScale :
                  scale === "log"    ? mapToLogScale    :
                  scale === "mel"    ? mapToMelScale    :
                  undefined; // or throw, or default
    
    if(!entry)
    {
        value = (min + max)/2;
        value = mapFn(value,min,max,height,padding,inverted);
    }
    else
    {
        const min_value = entry.min;
        const max_value = entry.max;
        value = clamp(entry.val(feature),min_value,max_value);
        if(applySoftclip && clampConfig[featureName])
        {    
            const shift = (entry.median - min_value)/(max_value - min_value);
            value = mapWithSoftClipping(value,min_value,max_value,min_value,max_value,shift,softclipScale)
        }
        value = mapFn(value,min,max,height,padding,inverted);
    }

    return value;

    //console.log("INSIDE computeYAxisValue:", "feature:", feature, "featureName:", featureName, "isLog:", isLog, "min:", min, "max:", max);

    // const mapFn = isLog ? mapToLogScale : mapToLinearScale;
    // const padded = (val) => mapFn(val,min,max,height,padding,inverted);

    //! Changed "Math.max(feature.X + 1, 1)" logic to "(feature.X >= 1 ? feature.X : 1)"
    // switch (featureName)
    // {
        // case "spectral_centroid":
        //     const safeCentroid = isLog ? (feature.spectral_centroid >= 1 ? feature.spectral_centroid : 1) : feature.spectral_centroid;
        //     return padded(safeCentroid);

        // case "weighted_spectral_centroid":
        //     const safeCentroidWeighted = isLog ? (feature.weighted_spectral_centroid >= 1 ? feature.weighted_spectral_centroid : 1) : feature.weighted_spectral_centroid;
        //     return padded(safeCentroidWeighted);
            
        // //! Added this
        // case "crepe_f0":
        //     const safeCrepeF0 = isLog ? (feature.crepe_f0 >= 1 ? feature.crepe_f0 : 1) : feature.crepe_f0;
        //     return padded(safeCrepeF0);

        // //! Modified this
        // case "perceived_pitch_f0_or_SC_weighted":
        //     const safePerceivedPitchWeighted = isLog ? (feature.perceived_pitch_f0_or_SC_weighted >= 1 ? feature.perceived_pitch_f0_or_SC_weighted : 1) : feature.perceived_pitch_f0_or_SC_weighted;
        //     return padded(safePerceivedPitchWeighted);
            // return padded(perceivedPitchF0OrSC(feature.yin_periodicity, feature.crepe_f0, feature.weighted_spectral_centroid));

        // case "multipeak_centroid":
        //     const safeMultipeakCentroid = isLog ? Math.max(feature.multipeak_centroid + 1, 1) : feature.multipeak_centroid;
        //     return padded(safeMultipeakCentroid);

        // case "spectral_peak":
        //     const safeSpectralPeak = isLog ? Math.max(feature.spectral_peak + 1, 1) : feature.spectral_peak;
        //     return padded(safeSpectralPeak);

        // case "weighted_spectral_centroid_bandwidth":
        //     const safeCentroidWeightedBandwidth = isLog ? Math.max(feature.weighted_spectral_centroid + 1, 1) : feature.weighted_spectral_centroid;
        //     return padded(spectralCentroidWithBandwidthWeight(feature.weighted_spectral_centroid, feature.spectral_bandwidth));
        
        // case "centroid_peak_bandwidth":
        //     return padded(computeBlendedTonalY(feature.spectral_peak, feature.weighted_spectral_centroid, feature.spectral_bandwidth));
        
        // case "centroid_peak_bandwidth_prominence":
        //     return padded(computeTonalYWithProminenceDb({
        //         periodicity: feature.yin_periodicity,
        //         crepeF0: feature.crepe_f0,
        //         spectralCentroidHz: feature.weighted_spectral_centroid,
        //         spectralBandwidthHz: feature.spectral_bandwidth,
        //         localPeakHz: feature.spectral_peak,
        //         prominenceDb: feature.spectral_peak_prominence_db
        //     }));
            
        // case "perceived_pitch":
        //     return padded(perceivedPitch(feature.crepe_confidence, feature.crepe_f0, feature.spectral_centroid));
    
        // case "perceived_pitch_librosa":
        //     return padded(perceivedPitchLibrosa(feature.crepe_confidence, feature.yin_f0_librosa, feature.spectral_centroid));
    
        // case "perceived_pitch_librosa_periodicity":
        //     return padded(perceivedPitchLibrosaPeriodicity(feature.yin_periodicity, feature.yin_f0_librosa, feature.spectral_centroid));
    
        // case "perceived_pitch_crepe_periodicity":
        //     return padded(perceivedPitchCrepePeriodicity(feature.yin_periodicity, feature.crepe_f0, feature.spectral_centroid));
    
        // case "perceived_pitch_f0_or_SC":
        //     return padded(perceivedPitchF0OrSC(feature.yin_periodicity, feature.crepe_f0, feature.spectral_centroid));

    
        // case "perceived_pitch_f0_candidates_periodicity":
        //     return padded(perceivedPitchF0Candidates(feature.yin_periodicity, feature.f0_candidates, feature.spectral_centroid));
    
        // default:
        //     return null; // fallback to config-based mapping
    // }
}

//! Merged with computeYAxisValue
// function getMappedYAxisFallback(featureName,config,canvasHeight,padding,isInverted,isLog,min,max,defaultValue = 400,fallbackFeature = "spectral_centroid")
// {
//     if (featureName === "none")
//     {
//         return defaultValue;
//     }
//     const entry = config[featureName] ?? config[fallbackFeature];
//     const value = clamp(entry.val(),min,max);

//     const mapFn = isLog ? mapToLogScale : mapToLinearScale;
//     return mapFn(value,min,max,canvasHeight,padding,isInverted);
// }
  
function computeRobustStats(data,lowerPercentile = 5,upperPercentile = 95)
{
    const length = data.length
    if(!length)
    {
        return { min: 0, max: 1 };
    }
    const sorted = [...data].sort((a,b) => a - b);

    let lowerIndex,upperIndex;
    if(lowerPercentile == upperPercentile)
    {
        const index = Math.round((lowerPercentile/100)*(length - 1));
        lowerIndex = index;
        upperIndex = index;
    }
    else
    {
        lowerIndex = Math.floor((lowerPercentile/100)*(length - 1));
        upperIndex = Math.ceil((upperPercentile/100)*(length - 1));
    }

    let mid = Math.floor(length/2)
    const median = length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid])/2

    const clamped = sorted.slice(lowerIndex,upperIndex + 1);
    const clamped_length = clamped.length;
    mid = Math.floor(clamped_length/2)
    const clamped_median = clamped_length % 2 !== 0 ? clamped[mid] : (clamped[mid - 1] + clamped[mid])/2

    return {
        min: sorted[lowerIndex],
        max: sorted[upperIndex],
        median: median,
        clamped_median: clamped_median
    };
}
  
/**
 * Compute robust min/max for raw and derived audio features.
 * @param {Object[]} features - Your audioData.features array.
 * @param {string[]} rawFeatureNames - Keys that exist directly on each feature object.
 * @param {Object} derivedFeatureFuncs - { name: function(f) => value }
 * @param {number} [lower=5] - Lower percentile.
 * @param {number} [upper=95] - Upper percentile.
 * @returns {Object} - { featureName: { min, max } }
 */
function prepareRobustStats(features,rawFeatureNames,derivedFeatureFuncs,lower = 5,upper = 95)
{
    const robustStats = {};
  
    // Handle raw features
    rawFeatureNames.forEach(name =>
    {
        // const values = features.map(f => f[name]).filter(v => isFinite(v));
        const values = features.map(f => f[name]).filter((v, i) => isFinite(v) && features[i]["loudness"] !== 0);
        robustStats[name] = computeRobustStats(values,lower,upper);
    });
  
    // Handle derived features
    for (const [name,func] of Object.entries(derivedFeatureFuncs))
    {
        // const values = features.map(func).filter(v => isFinite(v));
        const values = features.map(func).filter((v, i) => isFinite(v) && features[i]["loudness"] !== 0);
        robustStats[name] = computeRobustStats(values,lower,upper);
    }
  
    return robustStats;
}
  
// ✅ Clamp config lives outside
let clampConfig = {};

// ✅ Update function lives outside
function updateClampConfig()
{
    const selected = Array.from(document.getElementById("clamp-feature-select").selectedOptions).map(opt => opt.value);
    rawFeatureNames.forEach(name =>
    {
        clampConfig[name] = selected.includes(name);
    });
}
document.getElementById("clamp-feature-select").addEventListener("change",updateClampConfig);
updateClampConfig();

// Initialize an array to store dot positions
// let pointsArray = [];
//! Modifications here & there
// function drawVisualization(audioData, svgContainer, canvasWidth, maxDuration, fileIndex, baseHue, isObjectifyEnabled)
function drawVisualization()
{
    const configurations = [];
    let minValue = Number.MAX_SAFE_INTEGER;
    let maxValue = Number.MIN_SAFE_INTEGER;
    let yAxisFormatter;

    const threshold_value = parseFloat(document.getElementById("tmp-slider").noUiSlider.get());
    const gamma_value = parseFloat(document.getElementById("gamma-slider").noUiSlider.get());
    const periodicity_selector = globalAudioData.data[0].features[0].raw_periodicity !== undefined ? "raw_periodicity" : "yin_periodicity";

    // Get the selected feature and characteristic
    const selectedFeature1 = featureSelect1.value;
    const selectedFeature2 = featureSelect2.value;
    const selectedFeature3 = featureSelect3.value;
    const selectedFeature4 = featureSelect4.value;
    const selectedFeature5 = featureSelect5.value;
    const selectedFeature6 = featureSelect6.value;
    const selectedFeature7 = featureSelect7.value;
    
    // Get the current state of the checkbox and the slider value
    // const isThresholdEnabled = thresholdToggle.checked;
    // const zeroCrossingThreshold = parseFloat(thresholdSlider.value);

    const isThresholdCircleEnabled = thresholdCircle.checked;
    const isJoinPathsEnabled = joinDataPoints.checked;

    //! Changed this
    const isLineSketchingEnabled = !document.getElementById("linePolygonMode").checked
    const isPolygonEnabled = !isLineSketchingEnabled
    // const isLineSketchingEnabled = lineSketchingToggle.checked;
    // const isPolygonEnabled = polygonToggle.checked;

    const isGlobalClampEnabled = document.getElementById("toggleClamp").checked;
    const clampSlider = document.getElementById("slider-clamp");
    const [lowerClampBound,upperClampBound] = clampSlider.noUiSlider.get().map(parseFloat);

    const isSoftclipEnabled = document.getElementById("toggleSoftclip").checked;
    const softclipSlider = document.getElementById("slider-softclip");
    const softclipScale = parseFloat(softclipSlider.noUiSlider.get());

    const isLogSelected = document.getElementById("log-linear").checked;
    const isMelSelected = document.getElementById("mel-scale").checked;

    //! Tried to move filtering to frontend, didn't get good results
    // const applyFiltering = document.getElementById("filter_button").checked;
    // const overlap = percentages[document.getElementById("window_overlap_selector").value];
    // const kernelSize = calculate_optimal_length(overlap);

    const scale = isMelSelected ? "mel" :
                  isLogSelected ? "log" :
                  "linear";

    // Y AXIS
    const isInverted_y_axis = document.getElementById("invertMapping-5")?.checked;

    // LINE LENGTH
    const slider1 = document.getElementById("slider-1");
    const isInverted_lineLength = document.getElementById("invertMapping-1")?.checked;
    const {startRange_lineLength,endRange_lineLength} = calculateDynamicRange(slider1,isInverted_lineLength,"startRange_lineLength","endRange_lineLength");

    // LINE WIDTH
    const slider2 = document.getElementById("slider-2");
    const isInverted_lineWidth = document.getElementById("invertMapping-2")?.checked;
    const {startRange_lineWidth,endRange_lineWidth } = calculateDynamicRange(slider2,isInverted_lineWidth,"startRange_lineWidth","endRange_lineWidth");

    // COLOR SATURATION
    const slider3 = document.getElementById("slider-3");
    const isInverted_colorSaturation = document.getElementById("invertMapping-3")?.checked;
    const {startRange_colorSaturation,endRange_colorSaturation} = calculateDynamicRange(slider3,isInverted_colorSaturation,"startRange_colorSaturation","endRange_colorSaturation");

    // COLOR LIGHTNESS
    const slider6 = document.getElementById("slider-6");
    const isInverted_colorLightness = document.getElementById("invertMapping-6")?.checked;
    const {startRange_colorLightness,endRange_colorLightness} = calculateDynamicRange(slider6,isInverted_colorLightness,"startRange_colorLightness","endRange_colorLightness");

    // ANGLE
    const slider4 = document.getElementById("slider-4");
    const isInverted_angle = document.getElementById("invertMapping-4")?.checked;
    const {startRange_angle,endRange_angle} = calculateDynamicRange(slider4,isInverted_angle,"startRange_angle","endRange_angle");

    // DASH ARRAY
    const slider7 = document.getElementById("slider-7");
    const isInverted_dashArray = document.getElementById("invertMapping-7")?.checked;
    const {startRange_dashArray,endRange_dashArray} = calculateDynamicRange(slider7,isInverted_dashArray,"startRange_dashArray","endRange_dashArray");

    //! Moved derived feature calculations to backend #tmpf0
    // const derivedFeatureFuncs = {};
    const derivedFeatureFuncs = { perceived_pitch_f0_or_SC_weighted: f => perceivedPitchF0OrSC(f[periodicity_selector],f.crepe_f0,f.weighted_spectral_centroid,threshold_value,gamma_value) };
    // const derivedFeatureFuncs = { perceived_pitch_f0_or_SC_weighted: f => perceivedPitchF0OrSC(f.crepe_confidence,f.crepe_f0,f.spectral_centroid,document.getElementById("tmp-slider").value) }; // Tmp just for testing the slider
    // const derivedFeatureFuncs = {
    //     perceived_pitch: f => perceivedPitch(f.crepe_confidence, f.crepe_f0, f.spectral_centroid),
    //     perceived_pitch_librosa: f => perceivedPitchLibrosa(f.crepe_confidence, f.yin_f0_librosa, f.spectral_centroid),
    //     perceived_pitch_librosa_periodicity: f => perceivedPitchLibrosaPeriodicity(f.yin_periodicity, f.yin_f0_librosa, f.spectral_centroid),
    //     perceived_pitch_crepe_periodicity: f => perceivedPitchCrepePeriodicity(f.yin_periodicity, f.crepe_f0, f.spectral_centroid),
    //     perceived_pitch_f0_or_SC: f => perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.spectral_centroid),
    //     perceived_pitch_f0_or_SC_weighted: f => perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.weighted_spectral_centroid), //! Added this
    //     perceived_pitch_f0_candidates_periodicity: f => perceivedPitchF0Candidates(f.yin_periodicity, f.f0_candidates, f.spectral_centroid),
    //     loudness_zcr: f => f.loudness * f.zerocrossingrate,
    //     loudness_periodicity: f => f.loudness * (1 - f.yin_periodicity),
    //     loudness_pitchConf: f => f.loudness * (1 - f.crepe_confidence)
    // };

    const svgContainer = document.getElementById("svgCanvas");
    const canvasWidth = svgContainer.getAttribute("width");
    const canvasHeight = svgContainer.getAttribute("height");
    const padding = parseInt(svgContainer.getAttribute("padding"));
    const maxDuration = Math.max(...globalAudioData.data.map(file => file.features[file.features.length - 1]?.timestamp || 0));

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    svgContainer.appendChild(defs);

    const isObjectifyEnabled = document.getElementById("objectifierMode").checked

    //! Loop through files
    globalAudioData.data.forEach((audioData,fileIndex) =>
    {
        // console.log(`File ${fileIndex}: ${audioData.filename}`);

        //! Added this [#update: moved this calculations to backend (run_serial_extraction)]
        // for (let i = 0; i < audioData.features.length; i++)
        // {
        //     const feature = audioData.features[i];
        //     feature["yin_periodicity"] = adjustForSaturationPeriodicity(feature["yin_periodicity"]);
        // }

        //! Tried to move filtering to frontend, didn't get good results
        // const audioData = JSON.parse(JSON.stringify(audioDataRef));
        // if(applyFiltering)
        // {
        //     rawFeatureNames.forEach(name =>
        //     {
        //         if(clampConfig[name])
        //         {
        //             let values = audioData.features.map(f => f[name]);
        //             values = medfilt(values,kernelSize);
        //             for (let i = 0; i < audioData.features.length; i++)
        //             {
        //                 audioData.features[i][name] = values[i];
        //             }
        //         }
        //     });
        // }
        
        const robustStats = prepareRobustStats(audioData.features,rawFeatureNames,derivedFeatureFuncs,lowerClampBound,upperClampBound);
        //console.log("clampConfig:", clampConfig);
        //console.log("✔️ robustStats:", robustStats);

        //! Introducing dynamic calculations for all features
        const maxFeatureValues = Object.fromEntries(rawFeatureNames.map(key => [key,Number.MIN_SAFE_INTEGER]));
        const minFeatureValues = Object.fromEntries(rawFeatureNames.map(key => [key,Number.MAX_SAFE_INTEGER]));
        for (let i = 0; i < audioData.features.length; i++)
        {
            const feature = audioData.features[i];

            //! tmpf0
            feature["perceived_pitch_f0_or_SC_weighted"] = perceivedPitchF0OrSC(feature[periodicity_selector],feature.crepe_f0,feature.weighted_spectral_centroid,threshold_value,gamma_value); // tmp just for testing the slider

            // Skip silent frames
            if (feature["loudness"] === 0) continue;

            rawFeatureNames.forEach(name =>
            {
                if(feature[name] < minFeatureValues[name])
                {
                    minFeatureValues[name] = feature[name];
                }
                if(feature[name] > maxFeatureValues[name])
                {
                    maxFeatureValues[name] = feature[name];
                }
            });
        }
        // Initialize maxAmplitude to the smallest possible value
        // let maxSpectralCentroid = Number.MIN_SAFE_INTEGER;
        // let maxSpectralCentroidWeigthed = Number.MIN_SAFE_INTEGER;
        // let maxF0Crepe = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchF0OrSC_weighted = Number.MIN_SAFE_INTEGER;
        // let maxLoudness = Number.MIN_SAFE_INTEGER;
        // let maxLoudnessPeriodicity = Number.MIN_SAFE_INTEGER;
        // let maxLoudnessCrepeConfidence = Number.MIN_SAFE_INTEGER;
        // let maxSharpness = Number.MIN_SAFE_INTEGER;
        // let maxMirMpsRoughness = Number.MIN_SAFE_INTEGER;
        // let maxMirSharpnessZwicker = Number.MIN_SAFE_INTEGER;
        // let maxMirRoughnessVassilakis = Number.MIN_SAFE_INTEGER;
        // let maxPeriodicity = Number.MIN_SAFE_INTEGER;
        // let maxF0CrepeConfidence = Number.MIN_SAFE_INTEGER;
        // let maxSpectralFlux = Number.MIN_SAFE_INTEGER;
        // let maxSpectralFlatness = Number.MIN_SAFE_INTEGER;
        // let maxSpectralBandwidth = Number.MIN_SAFE_INTEGER;
        // let maxSpectralPeak = Number.MIN_SAFE_INTEGER;
        // let maxSpectralPeakProminenceDb = Number.MIN_SAFE_INTEGER;
        // let maxSpectralPeakProminenceNorm = Number.MIN_SAFE_INTEGER;
        // let maxMultipeakCentroid = Number.MIN_SAFE_INTEGER;
        // let maxAmplitude = Number.MIN_SAFE_INTEGER;
        // let maxZerocrossingrate = Number.MIN_SAFE_INTEGER;
        // let maxYinF0Librosa = Number.MIN_SAFE_INTEGER;
        // let maxYinF0Aubio = Number.MIN_SAFE_INTEGER;
        // let maxStandardDeviation = Number.MIN_SAFE_INTEGER;
        // let maxBrightness = Number.MIN_SAFE_INTEGER;
        // let maxF0Candidates = Number.MIN_SAFE_INTEGER;
        // let maxMirRoughnessZwicker = Number.MIN_SAFE_INTEGER;
        // let maxMirRoughnessSethares = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitch = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchLibrosa = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchLibrosaPeriodicity = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchCrepePeriodicity = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchF0OrSC = Number.MIN_SAFE_INTEGER;
        // let maxPerceivedPitchF0Candidates = Number.MIN_SAFE_INTEGER;
        // let maxSpecralCentroidWeigthedBandwidth = Number.MIN_SAFE_INTEGER;
        // let maxCentroidPeakBandwidth = Number.MIN_SAFE_INTEGER;
        // let maxCentroidPeakBandwidthProminence = Number.MIN_SAFE_INTEGER;
        // let maxSpectralCentroidMinusStandardDeviation = Number.MIN_SAFE_INTEGER;

        // Threshold for zero crossing rate
        //const zeroCrossingThreshold = 30;
        // Calculate maxAmplitude and maxZerocrossingrate
    
        // for (let i = 0; i < audioData.features.length; i++)
        // {
            // const f = audioData.features[i];

            // // Skip silent frames
            // if (f.loudness === 0) continue;
        
            // if (f.spectral_centroid > maxSpectralCentroid) maxSpectralCentroid = f.spectral_centroid;
            // if (f.weighted_spectral_centroid > maxSpectralCentroidWeigthed) maxSpectralCentroidWeigthed = f.weighted_spectral_centroid;
            // if (f.crepe_f0 > maxF0Crepe) maxF0Crepe = f.crepe_f0;
            // if (f.perceived_pitch_f0_or_SC_weighted > maxPerceivedPitchF0OrSC_weighted) maxPerceivedPitchF0OrSC_weighted = f.perceived_pitch_f0_or_SC_weighted;
            // if (f.loudness > maxLoudness) maxLoudness = f.loudness;
            // if (f.loudness_periodicity > maxLoudnessPeriodicity) maxLoudnessPeriodicity = f.loudness_periodicity;
            // if (f.loudness_pitchConf > maxLoudnessCrepeConfidence) maxLoudnessCrepeConfidence = f.loudness_pitchConf;
            // if (f.sharpness > maxSharpness) maxSharpness = f.sharpness;
            // if (f.mir_mps_roughness > maxMirMpsRoughness) maxMirMpsRoughness = f.mir_mps_roughness;
            // if (f.mir_sharpness_zwicker > maxMirSharpnessZwicker) maxMirSharpnessZwicker = f.mir_sharpness_zwicker;
            // if (f.mir_roughness_vassilakis > maxMirRoughnessVassilakis) maxMirRoughnessVassilakis = f.mir_roughness_vassilakis;
            // if (f.yin_periodicity > maxPeriodicity) maxPeriodicity = f.yin_periodicity;
            // if (f.crepe_confidence > maxF0CrepeConfidence) maxF0CrepeConfidence = f.crepe_confidence;

            // if (f.amplitude > maxAmplitude) maxAmplitude = f.amplitude;
            // if (f.zerocrossingrate > maxZerocrossingrate) maxZerocrossingrate = f.zerocrossingrate;
            // if (f.spectral_bandwidth > maxSpectralBandwidth) maxSpectralBandwidth = f.spectral_bandwidth;
            // if (f.spectral_peak > maxSpectralPeak) maxSpectralPeak = f.spectral_peak;
            // if (f.spectral_peak_prominence_db > maxSpectralPeakProminenceDb) maxSpectralPeakProminenceDb = f.spectral_peak_prominence_db;
            // if (f.spectral_peak_prominence_norm > maxSpectralPeakProminenceNorm) maxSpectralPeakProminenceNorm = f.spectral_peak_prominence_norm;
            // if (f.multipeak_centroid > maxMultipeakCentroid) maxMultipeakCentroid = f.multipeak_centroid;
            // if (f.spectral_flatness > maxSpectralFlatness) maxSpectralFlatness = f.spectral_flatness;
            // if (f.spectral_flux > maxSpectralFlux) maxSpectralFlux = f.spectral_flux;
            // if (f.yin_f0_librosa > maxYinF0Librosa) maxYinF0Librosa = f.yin_f0_librosa;
            // if (f.yin_f0_aubio > maxYinF0Aubio) maxYinF0Aubio = f.yin_f0_aubio;
            // if (f.standard_deviation > maxStandardDeviation) maxStandardDeviation = f.standard_deviation;
            // const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
            // if (centroidMinusSD > maxSpectralCentroidMinusStandardDeviation)
            // {
            //     maxSpectralCentroidMinusStandardDeviation = centroidMinusSD;
            // }
            // if (f.brightness > maxBrightness) maxBrightness = f.brightness;
            // if (f.mir_roughness_zwicker > maxMirRoughnessZwicker) maxMirRoughnessZwicker = f.mir_roughness_zwicker;
            // if (f.mir_roughness_sethares > maxMirRoughnessSethares) maxMirRoughnessSethares = f.mir_roughness_sethares;

            // const pPitch = perceivedPitch(f.crepe_confidence, f.crepe_f0, f.spectral_centroid);
            // if (pPitch > maxPerceivedPitch) maxPerceivedPitch = pPitch;

            // const pPitchLibrosa = perceivedPitchLibrosa(f.crepe_confidence, f.yin_f0_librosa, f.spectral_centroid);
            // if (pPitchLibrosa > maxPerceivedPitchLibrosa) maxPerceivedPitchLibrosa = pPitchLibrosa;

            // const pPitchLibrosaPeriodicity = perceivedPitchLibrosaPeriodicity(f.yin_periodicity, f.yin_f0_librosa, f.spectral_centroid);
            // if (pPitchLibrosaPeriodicity > maxPerceivedPitchLibrosaPeriodicity) maxPerceivedPitchLibrosaPeriodicity = pPitchLibrosaPeriodicity;

            // const pPitchCrepePeriodicity = perceivedPitchCrepePeriodicity(f.yin_periodicity, f.crepe_f0, f.spectral_centroid);
            // if (pPitchCrepePeriodicity > maxPerceivedPitchCrepePeriodicity) maxPerceivedPitchCrepePeriodicity = pPitchCrepePeriodicity;

            // const pPitchF0OrSC = perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.spectral_centroid);
            // if (pPitchF0OrSC > maxPerceivedPitchF0OrSC) maxPerceivedPitchF0OrSC = pPitchF0OrSC;
        
            // const pPitchF0OrSC_weighted = perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.weighted_spectral_centroid);
            // if (pPitchF0OrSC_weighted > maxPerceivedPitchF0OrSC_weighted) maxPerceivedPitchF0OrSC_weighted = pPitchF0OrSC_weighted;

            // const pPitchF0Candidates = perceivedPitchF0Candidates(f.yin_periodicity, f.f0_candidates, f.spectral_centroid);
            // if (pPitchF0Candidates > maxPerceivedPitchF0Candidates) maxPerceivedPitchF0Candidates = pPitchF0Candidates;

            // const pCentroidBandwidthWeight = spectralCentroidWithBandwidthWeight(f.weighted_spectral_centroid, f.spectral_bandwidth);
            // if (pCentroidBandwidthWeight > maxSpecralCentroidWeigthedBandwidth) maxSpecralCentroidWeigthedBandwidth = pCentroidBandwidthWeight;

            // const pCentroidPeakBandwidth = computeBlendedTonalY(f.spectral_peak, f.weighted_spectral_centroid, f.spectral_bandwidth);
            // if (pCentroidPeakBandwidth > maxCentroidPeakBandwidth) maxCentroidPeakBandwidth = pCentroidPeakBandwidth;

            // const pCentroidPeakBandwidthProminence = computeTonalYWithProminenceDb({
            //     periodicity: f.yin_periodicity,
            //     crepeF0: f.crepe_f0,
            //     spectralCentroidHz: f.weighted_spectral_centroid,
            //     spectralBandwidthHz: f.spectral_bandwidth,
            //     localPeakHz: f.spectral_peak,
            //     prominenceDb: f.spectral_peak_prominence_db
            //   });
            // if (pCentroidPeakBandwidthProminence > maxCentroidPeakBandwidthProminence) maxCentroidPeakBandwidthProminence = pCentroidPeakBandwidthProminence;
        // }

        // Initialize min values to the largest possible value
        // let minSpectralCentroid = Number.MAX_SAFE_INTEGER;
        // let minSpectralCentroidWeigthed = Number.MAX_SAFE_INTEGER;
        // let minF0Crepe = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchF0OrSC_weighted = Number.MAX_SAFE_INTEGER;
        // let minLoudness = Number.MAX_SAFE_INTEGER;
        // let minLoudnessPeriodicity = Number.MAX_SAFE_INTEGER;
        // let minLoudnessCrepeConfidence = Number.MAX_SAFE_INTEGER;
        // let minSharpness = Number.MAX_SAFE_INTEGER;
        // let minMirMpsRoughness = Number.MAX_SAFE_INTEGER;
        // let minMirSharpnessZwicker = Number.MAX_SAFE_INTEGER;
        // let minMirRoughnessVassilakis = Number.MAX_SAFE_INTEGER;
        // let minF0CrepeConfidence = Number.MAX_SAFE_INTEGER;
        // let minPeriodicity = Number.MAX_SAFE_INTEGER;
        // let minSpectralFlux = Number.MAX_SAFE_INTEGER;
        // let minSpectralFlatness = Number.MAX_SAFE_INTEGER;
        // let minSpectralBandwidth = Number.MAX_SAFE_INTEGER;
        // let minSpectralPeak = Number.MAX_SAFE_INTEGER;
        // let minSpectralPeakProminenceDb = Number.MAX_SAFE_INTEGER;
        // let minSpectralPeakProminenceNorm = Number.MAX_SAFE_INTEGER;
        // let minMultipeakCentroid = Number.MAX_SAFE_INTEGER;
        // let minAmplitude = Number.MAX_SAFE_INTEGER;
        // let minZerocrossingrate = Number.MAX_SAFE_INTEGER;
        // let minYinF0Librosa = Number.MAX_SAFE_INTEGER;
        // let minYinF0Aubio = Number.MAX_SAFE_INTEGER;
        // let minStandardDeviation = Number.MAX_SAFE_INTEGER;
        // let minBrightness = Number.MAX_SAFE_INTEGER;
        // let minF0Candidates = Number.MAX_SAFE_INTEGER;
        // let minMirRoughnessZwicker = Number.MAX_SAFE_INTEGER;
        // let minMirRoughnessSethares = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitch = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchLibrosa = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchLibrosaPeriodicity = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchCrepePeriodicity = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchF0OrSC = Number.MAX_SAFE_INTEGER;
        // let minPerceivedPitchF0Candidates = Number.MAX_SAFE_INTEGER;
        // let minSpecralCentroidWeigthedBandwidth = Number.MAX_SAFE_INTEGER;
        // let minCentroidPeakBandwidth = Number.MAX_SAFE_INTEGER;
        // let minCentroidPeakBandwidthProminence = Number.MAX_SAFE_INTEGER;
        // let minSpectralCentroidMinusStandardDeviation = Number.MAX_SAFE_INTEGER;

        // for (let i = 0; i < audioData.features.length; i++)
        // {
        //     const f = audioData.features[i];
        
        //     // Skip silent frames
        //     if (f.loudness === 0) continue;
        
        //     if (f.spectral_centroid < minSpectralCentroid) minSpectralCentroid = f.spectral_centroid;
        //     if (f.weighted_spectral_centroid < minSpectralCentroidWeigthed) minSpectralCentroidWeigthed = f.weighted_spectral_centroid;
        //     if (f.crepe_f0 < minF0Crepe) minF0Crepe = f.crepe_f0;
        //     if (f.perceived_pitch_f0_or_SC_weighted < minPerceivedPitchF0OrSC_weighted) minPerceivedPitchF0OrSC_weighted = f.perceived_pitch_f0_or_SC_weighted;
        //     if (f.loudness < minLoudness) minLoudness = f.loudness;
        //     if (f.loudness_periodicity < minLoudnessPeriodicity) minLoudnessPeriodicity = f.loudness_periodicity;
        //     if (f.loudness_pitchConf < minLoudnessCrepeConfidence) minLoudnessCrepeConfidence = f.loudness_pitchConf;
        //     if (f.sharpness < minSharpness) minSharpness = f.sharpness;
        //     if (f.mir_mps_roughness < minMirMpsRoughness) minMirMpsRoughness = f.mir_mps_roughness;
        //     if (f.mir_sharpness_zwicker < minMirSharpnessZwicker) minMirSharpnessZwicker = f.mir_sharpness_zwicker;
        //     if (f.mir_roughness_vassilakis < minMirRoughnessVassilakis) minMirRoughnessVassilakis = f.mir_roughness_vassilakis;
        //     if (f.yin_periodicity < minPeriodicity) minPeriodicity = f.yin_periodicity;
        //     if (f.crepe_confidence < minF0CrepeConfidence) minF0CrepeConfidence = f.crepe_confidence;
            // if (f.amplitude < minAmplitude) minAmplitude = f.amplitude;
            // if (f.zerocrossingrate < minZerocrossingrate) minZerocrossingrate = f.zerocrossingrate;
            // if (f.spectral_bandwidth < minSpectralBandwidth) minSpectralBandwidth = f.spectral_bandwidth;
            // if (f.spectral_peak < minSpectralPeak) minSpectralPeak = f.spectral_peak;
            // if (f.spectral_peak_prominence_db < minSpectralPeakProminenceDb) minSpectralPeakProminenceDb = f.spectral_peak_prominence_db;
            // if (f.spectral_peak_prominence_norm < minSpectralPeakProminenceNorm) minSpectralPeakProminenceNorm = f.spectral_peak_prominence_norm;
            // if (f.multipeak_centroid < minMultipeakCentroid) minMultipeakCentroid = f.multipeak_centroid;
            // if (f.spectral_flatness < minSpectralFlatness) minSpectralFlatness = f.spectral_flatness;
            // if (f.spectral_flux < minSpectralFlux) minSpectralFlux = f.spectral_flux;
            // if (f.yin_f0_librosa < minYinF0Librosa) minYinF0Librosa = f.yin_f0_librosa;
            // if (f.yin_f0_aubio < minYinF0Aubio) minYinF0Aubio = f.yin_f0_aubio;
            // if (f.standard_deviation < minStandardDeviation) minStandardDeviation = f.standard_deviation;
            // const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
            // if (centroidMinusSD < minSpectralCentroidMinusStandardDeviation) {
            //     minSpectralCentroidMinusStandardDeviation = centroidMinusSD;
            // }
            // if (f.brightness < minBrightness) minBrightness = f.brightness;
            // if (f.mir_roughness_zwicker < minMirRoughnessZwicker) minMirRoughnessZwicker = f.mir_roughness_zwicker;
            // if (f.mir_roughness_sethares < minMirRoughnessSethares) minMirRoughnessSethares = f.mir_roughness_sethares;
            
            // const pPitch = perceivedPitch(f.crepe_confidence, f.crepe_f0, f.spectral_centroid);
            // if (pPitch < minPerceivedPitch) minPerceivedPitch = pPitch;

            // const pPitchLibrosa = perceivedPitchLibrosa(f.crepe_confidence, f.yin_f0_librosa, f.spectral_centroid);
            // if (pPitchLibrosa < minPerceivedPitchLibrosa) minPerceivedPitchLibrosa = pPitchLibrosa;

            // const pPitchLibrosaPeriodicity = perceivedPitchLibrosaPeriodicity(f.yin_periodicity, f.yin_f0_librosa, f.spectral_centroid);
            // if (pPitchLibrosaPeriodicity < minPerceivedPitchLibrosaPeriodicity) minPerceivedPitchLibrosaPeriodicity = pPitchLibrosaPeriodicity;

            // const pPitchCrepePeriodicity = perceivedPitchCrepePeriodicity(f.yin_periodicity, f.crepe_f0, f.spectral_centroid);
            // if (pPitchCrepePeriodicity < minPerceivedPitchCrepePeriodicity) minPerceivedPitchCrepePeriodicity = pPitchCrepePeriodicity;

            // const pPitchF0OrSC = perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.spectral_centroid);
            // if (pPitchF0OrSC < minPerceivedPitchF0OrSC) minPerceivedPitchF0OrSC = pPitchF0OrSC;

            // const pPitchF0OrSC_weighted = perceivedPitchF0OrSC(f.yin_periodicity, f.crepe_f0, f.weighted_spectral_centroid);
            // if (pPitchF0OrSC_weighted < minPerceivedPitchF0OrSC_weighted) minPerceivedPitchF0OrSC_weighted = pPitchF0OrSC_weighted;

            // const pPitchF0Candidates = perceivedPitchF0Candidates(f.yin_periodicity, f.f0_candidates, f.spectral_centroid);
            // if (pPitchF0Candidates < minPerceivedPitchF0Candidates) minPerceivedPitchF0Candidates = pPitchF0Candidates;

            // const pCentroidBandwidthWeight = spectralCentroidWithBandwidthWeight(f.weighted_spectral_centroid, f.spectral_bandwidth);
            // if (pCentroidBandwidthWeight < minSpecralCentroidWeigthedBandwidth) minSpecralCentroidWeigthedBandwidth = pCentroidBandwidthWeight;

            // const pCentroidPeakBandwidth = computeBlendedTonalY(f.spectral_peak, f.weighted_spectral_centroid, f.spectral_bandwidth);
            // if (pCentroidPeakBandwidth < minCentroidPeakBandwidth) minCentroidPeakBandwidth = pCentroidPeakBandwidth;

            // const pCentroidPeakBandwidthProminence = computeTonalYWithProminenceDb({
            //     periodicity: f.yin_periodicity,
            //     crepeF0: f.crepe_f0,
            //     spectralCentroidHz: f.weighted_spectral_centroid,
            //     spectralBandwidthHz: f.spectral_bandwidth,
            //     localPeakHz: f.spectral_peak,
            //     prominenceDb: f.spectral_peak_prominence_db
            //   });
            // if (pCentroidPeakBandwidthProminence < minCentroidPeakBandwidthProminence) minCentroidPeakBandwidthProminence = pCentroidPeakBandwidthProminence;
        // }

        //! Not used anywhere
        // for (let i = 0; i < audioData.features.length; i++)
        // {
        //     const f = audioData.features[i];
        
        //     f.normalizedSpectralCentroid = f.spectral_centroid / maxSpectralCentroid || 0;
        //     f.normalizedSpectralCentroidWeighted = f.weighted_spectral_centroid / maxSpectralCentroid || 0;
        //     f.normalizedF0Crepe = f.crepe_f0 / maxF0Crepe || 0;
        //     f.normalizedLoudness = f.loudness / maxLoudness || 0;
        //     f.normalizedLoudnessPeriodicity = f.loudness_periodicity / maxLoudnessPeriodicity || 0;
        //     f.normalizedLoudnessCrepeConfidence = f.loudness_pitchConf / maxLoudnessCrepeConfidence || 0;
        //     f.normalizedSharpness = f.sharpness / maxSharpness || 0;
        //     f.normalizedMirMpsRoughness = f.mir_mps_roughness / maxMirMpsRoughness || 0;
        //     f.normalizedMirSharpnessZwicker = f.mir_sharpness_zwicker / maxMirSharpnessZwicker || 0;
        //     f.normalizedMirRoughnessVassilakis = f.mir_roughness_vassilakis / maxMirRoughnessVassilakis || 0;
        //     f.normalizedPeriodicity = f.yin_periodicity / maxPeriodicity || 0;
        //     f.normalizedF0CrepeConfidence = f.crepe_confidence / maxF0CrepeConfidence || 0;
        //     // f.normalizedAmplitude = f.amplitude / maxAmplitude || 0;
        //     // f.normalizedZerocrossingrate = f.zerocrossingrate / maxZerocrossingrate || 0;
        //     // f.normalizedSpectralFlux = f.spectral_flux / maxSpectralFlux || 0;
        //     // f.normalizedYinF0Librosa = f.yin_f0_librosa / maxYinF0Librosa || 0;
        //     // f.normalizedYinF0Aubio = f.yin_f0_aubio / maxYinF0Aubio || 0;
        //     // f.normalizedStandardDeviation = f.standard_deviation / maxStandardDeviation || 0;
        
        //     // const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
        //     // f.normalizedSpectralCentroidMinusStandardDeviation =
        //     //     centroidMinusSD / maxSpectralCentroidMinusStandardDeviation || 0;
        
        //     // f.normalizedBrightness = f.brightness / maxBrightness || 0;
        
        //     // f.normalizedMirRoughnessZwicker = f.mir_roughness_zwicker / maxMirRoughnessZwicker || 0;
        //     // f.normalizedMirRoughnessSethares = f.mir_roughness_sethares / maxMirRoughnessSethares || 0;
        // }
        
        // let maxLogSpectralCentroid = Math.log10(maxSpectralCentroid + 1);
        // let minLogSpectralCentroid = Math.log10(1);

        // Add padding (in pixels) to the top and adjust the bottom range

        //! Create config dynamically
        const featureConfig = {};
        rawFeatureNames.forEach(name =>
        {
            featureConfig[name] =
            {
                val: (feature) => feature[name],
                min: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].min : minFeatureValues[name],
                max: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].max : maxFeatureValues[name],
                median: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].clamped_median : robustStats[name].median,
            }
        });
        //! Temporarily adding this #tmpf0
        featureConfig["perceived_pitch_f0_or_SC_weighted"].val = (feature) => perceivedPitchF0OrSC(feature[periodicity_selector],feature.crepe_f0,feature.weighted_spectral_centroid,threshold_value,gamma_value); // tmp just for testing the slider

        const scaleInfo = getYAxisScaleFromConfig(featureConfig,selectedFeature5,scale);
        if(scaleInfo)
        {
            const {min,max,labelFormatter} = scaleInfo;
            if(min < minValue)
            {
                minValue = min;
            }
            if(max > maxValue)
            {
                maxValue = max;
            }
            yAxisFormatter = labelFormatter;
        }
        else
        {
            console.warn(`Feature ${selectedFeature5} not found in config`);
        }

        //! Save feature configurations
        configurations.push(featureConfig);

    });

    pathData = {};
    globalAudioData.data.forEach((audioData,fileIndex) =>
    {
        let previousDots = []; // To store the scattered dots of the previous data point
        pathData[fileIndex] = []; // Array to store attributes for each path

        const featureConfig = configurations[fileIndex]

        //! Added this for "deterministic randomness"
        rng = random_engine(0); // rng() is a global function (dont add 'const' or anything)
        
        //! Added this
        const isHidden = !document.getElementById(`toggle-${fileIndex}`).isActive;
        const pathGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        pathGroup.id = `audio-path-${fileIndex}`; // Add unique id
        pathGroup.style.display = isHidden ? "none" : "inline";
        // pathGroup.classList.add(`audio-path-${fileIndex}`); // Add unique class
        
        //let colorHue = hexToHSL(huePicker.value); // Convert hex to HSL
        // const colorHue = getFileBaseHue(fileIndex);
        const colorHue = Number(document.getElementById(`color-slider-${fileIndex}`).value);
        //! Added these
        const hue1 = ((colorHue - 30) + 360) % 360; // analogous color
        const hue2 = ((colorHue + 30) + 360) % 360; // analogous color

        for (let i = 0; i < audioData.features.length; i++)
        {
            const feature = audioData.features[i];

            const xAxis = map(feature["timestamp"],0,maxDuration,0,canvasWidth);
            
            // const featureConfig = {
            //     spectral_centroid: {
            //       val: () => feature.spectral_centroid,
            //       min: (isGlobalClampEnabled && clampConfig["spectral_centroid"]) ? robustStats["spectral_centroid"].min : minSpectralCentroid,
            //       max: (isGlobalClampEnabled && clampConfig["spectral_centroid"]) ? robustStats["spectral_centroid"].max : maxSpectralCentroid,
            //       softclipping: isSoftclipEnabled
            //     },
            //     weighted_spectral_centroid: {
            //       val: () => feature.weighted_spectral_centroid,
            //       min: (isGlobalClampEnabled && clampConfig["weighted_spectral_centroid"]) ? robustStats["weighted_spectral_centroid"].min : minSpectralCentroidWeigthed,
            //       max: (isGlobalClampEnabled && clampConfig["weighted_spectral_centroid"]) ? robustStats["weighted_spectral_centroid"].max : maxSpectralCentroidWeigthed,
            //       softclipping: isSoftclipEnabled
            //     },
            //     crepe_f0: {
            //       val: () => feature.crepe_f0,
            //       min: (isGlobalClampEnabled && clampConfig["crepe_f0"]) ? robustStats["crepe_f0"].min : minF0Crepe,
            //       max: (isGlobalClampEnabled && clampConfig["crepe_f0"]) ? robustStats["crepe_f0"].max : maxF0Crepe,
            //       softclipping: isSoftclipEnabled
            //     },
            //     perceived_pitch_f0_or_SC_weighted: {
            //       val: () => feature.perceived_pitch_f0_or_SC_weighted,
            //       min: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_or_SC_weighted"]) ? robustStats["perceived_pitch_f0_or_SC_weighted"].min : minPerceivedPitchF0OrSC_weighted,
            //       max: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_or_SC_weighted"]) ? robustStats["perceived_pitch_f0_or_SC_weighted"].max : maxPerceivedPitchF0OrSC_weighted,
            //       softclipping: isSoftclipEnabled
            //     },
            //     loudness: {
            //       val: () => feature.loudness,
            //       min: (isGlobalClampEnabled && clampConfig["loudness"]) ? robustStats["loudness"].min : minLoudness,
            //       max: (isGlobalClampEnabled && clampConfig["loudness"]) ? robustStats["loudness"].max : maxLoudness,
            //       softclipping: isSoftclipEnabled
            //     },
            //     loudness_periodicity: {
            //       val: () => feature.loudness_periodicity,
            //       min: (isGlobalClampEnabled && clampConfig["loudness_periodicity"]) ? robustStats["loudness_periodicity"].min : minLoudnessPeriodicity,
            //       max: (isGlobalClampEnabled && clampConfig["loudness_periodicity"]) ? robustStats["loudness_periodicity"].max : maxLoudnessPeriodicity,
            //       softclipping: isSoftclipEnabled
            //     },
            //     loudness_pitchConf: {
            //       val: () => feature.loudness_pitchConf,
            //       min: (isGlobalClampEnabled && clampConfig["loudness_pitchConf"]) ? robustStats["loudness_pitchConf"].min : minLoudnessCrepeConfidence,
            //       max: (isGlobalClampEnabled && clampConfig["loudness_pitchConf"]) ? robustStats["loudness_pitchConf"].max : maxLoudnessCrepeConfidence,
            //       softclipping: isSoftclipEnabled
            //     },
            //     sharpness: {
            //       val: () => feature.sharpness,
            //       min: (isGlobalClampEnabled && clampConfig["sharpness"]) ? robustStats["sharpness"].min : minSharpness,
            //       max: (isGlobalClampEnabled && clampConfig["sharpness"]) ? robustStats["sharpness"].max : maxSharpness,
            //       softclipping: isSoftclipEnabled
            //     },
            //     mir_mps_roughness: {
            //       val: () => feature.mir_mps_roughness,
            //       min: (isGlobalClampEnabled && clampConfig["mir_mps_roughness"]) ? robustStats["mir_mps_roughness"].min : minMirMpsRoughness,
            //       max: (isGlobalClampEnabled && clampConfig["mir_mps_roughness"]) ? robustStats["mir_mps_roughness"].max : maxMirMpsRoughness,
            //       softclipping: isSoftclipEnabled
            //     },
            //     mir_sharpness_zwicker: {
            //       val: () => feature.mir_sharpness_zwicker,
            //       min: (isGlobalClampEnabled && clampConfig["mir_sharpness_zwicker"]) ? robustStats["mir_sharpness_zwicker"].min : minMirSharpnessZwicker,
            //       max: (isGlobalClampEnabled && clampConfig["mir_sharpness_zwicker"]) ? robustStats["mir_sharpness_zwicker"].max : maxMirSharpnessZwicker,
            //       softclipping: isSoftclipEnabled
            //     },
            //     mir_roughness_vassilakis: {
            //       val: () => feature.mir_roughness_vassilakis,
            //       min: (isGlobalClampEnabled && clampConfig["mir_roughness_vassilakis"]) ? robustStats["mir_roughness_vassilakis"].min : minMirRoughnessVassilakis,
            //       max: (isGlobalClampEnabled && clampConfig["mir_roughness_vassilakis"]) ? robustStats["mir_roughness_vassilakis"].max : maxMirRoughnessVassilakis,
            //       softclipping: isSoftclipEnabled
            //     },
            //     yin_periodicity: {
            //       val: () => feature.yin_periodicity,
            //       min: (isGlobalClampEnabled && clampConfig["yin_periodicity"]) ? robustStats["yin_periodicity"].min : minPeriodicity,
            //       max: (isGlobalClampEnabled && clampConfig["yin_periodicity"]) ? robustStats["yin_periodicity"].max : maxPeriodicity,
            //       softclipping: isSoftclipEnabled
            //     },
            //     crepe_confidence: {
            //       val: () => feature.crepe_confidence,
            //       min: (isGlobalClampEnabled && clampConfig["crepe_confidence"]) ? robustStats["crepe_confidence"].min : minF0CrepeConfidence,
            //       max: (isGlobalClampEnabled && clampConfig["crepe_confidence"]) ? robustStats["crepe_confidence"].max : maxF0CrepeConfidence,
            //       softclipping: isSoftclipEnabled
            //     }
                // amplitude: {
                //   val: () => feature.amplitude,
                //   min: (isGlobalClampEnabled && clampConfig["amplitude"]) ? robustStats["amplitude"].min : minAmplitude,
                //   max: (isGlobalClampEnabled && clampConfig["amplitude"]) ? robustStats["amplitude"].max : maxAmplitude,
                //   softclipping: isSoftclipEnabled
                // },
                // zerocrossingrate: {
                //   val: () => feature.zerocrossingrate,
                //   min: (isGlobalClampEnabled && clampConfig["zerocrossingrate"]) ? robustStats["zerocrossingrate"].min : minZerocrossingrate,
                //   max: (isGlobalClampEnabled && clampConfig["zerocrossingrate"]) ? robustStats["zerocrossingrate"].max : maxZerocrossingrate,
                //   softclipping: isSoftclipEnabled
                // },
                // calibrated_zerocrossingrate: {
                //   val: () => calibrateZCR(feature.zerocrossingrate, feature.spectral_centroid, 44100),
                //   min: (isGlobalClampEnabled && clampConfig["calibrated_zerocrossingrate"]) ? robustStats["zerocrossingrate"].min : minZerocrossingrate,
                //   max: (isGlobalClampEnabled && clampConfig["calibrated_zerocrossingrate"]) ? robustStats["zerocrossingrate"].max : maxZerocrossingrate,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_flux: {
                //   val: () => feature.spectral_flux,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_flux"]) ? robustStats["spectral_flux"].min : minSpectralFlux,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_flux"]) ? robustStats["spectral_flux"].max : maxSpectralFlux,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_bandwidth: {
                //   val: () => feature.spectral_bandwidth,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_bandwidth"]) ? robustStats["spectral_bandwidth"].min : minSpectralBandwidth,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_bandwidth"]) ? robustStats["spectral_bandwidth"].max : maxSpectralBandwidth,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_peak: {
                //   val: () => feature.spectral_peak,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_peak"]) ? robustStats["spectral_peak"].min : minSpectralPeak,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_peak"]) ? robustStats["spectral_peak"].max : maxSpectralPeak,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_peak_prominence_db: {
                //   val: () => feature.spectral_peak_prominence_db,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_peak_prominence_db"]) ? robustStats["spectral_peak_prominence_db"].min : minSpectralPeakProminenceDb,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_peak_prominence_db"]) ? robustStats["spectral_peak_prominence_db"].max : maxSpectralPeakProminenceDb,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_peak_prominence_norm: {
                //   val: () => feature.spectral_peak_prominence_norm,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_peak_prominence_norm"]) ? robustStats["spectral_peak_prominence_norm"].min : minSpectralPeakProminenceNorm,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_peak_prominence_norm"]) ? robustStats["spectral_peak_prominence_norm"].max : maxSpectralPeakProminenceNorm,
                //   softclipping: isSoftclipEnabled
                // },
                // multipeak_centroid: {
                //     val: () => safeLogInput(feature.multipeak_centroid),
                //     min: (isGlobalClampEnabled && clampConfig["multipeak_centroid"]) ? robustStats["multipeak_centroid"].min + 1 : minMultipeakCentroid + 1,
                //     max: (isGlobalClampEnabled && clampConfig["multipeak_centroid"]) ? robustStats["multipeak_centroid"].max + 1 : maxMultipeakCentroid + 1,
                //     softclipping: isSoftclipEnabled
                //   },
                // spectral_flatness: {
                //   val: () => feature.spectral_flatness,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_flatness"]) ? robustStats["spectral_flatness"].min : minSpectralFlatness,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_flatness"]) ? robustStats["spectral_flatness"].max : maxSpectralFlatness,
                //   softclipping: isSoftclipEnabled
                // },
                // yin_f0_librosa: {
                //   val: () => feature.yin_f0_librosa,
                //   min: (isGlobalClampEnabled && clampConfig["yin_f0_librosa"]) ? robustStats["yin_f0_librosa"].min : minYinF0Librosa,
                //   max: (isGlobalClampEnabled && clampConfig["yin_f0_librosa"]) ? robustStats["yin_f0_librosa"].max : maxYinF0Librosa,
                //   softclipping: isSoftclipEnabled
                // },
                // yin_f0_aubio: {
                //   val: () => feature.yin_f0_aubio,
                //   min: (isGlobalClampEnabled && clampConfig["yin_f0_aubio"]) ? robustStats["yin_f0_aubio"].min : minYinF0Aubio,
                //   max: (isGlobalClampEnabled && clampConfig["yin_f0_aubio"]) ? robustStats["yin_f0_aubio"].max : maxYinF0Aubio,
                //   softclipping: isSoftclipEnabled
                // },
                // spectral_deviation: {
                //   val: () => feature.standard_deviation,
                //   min: (isGlobalClampEnabled && clampConfig["spectral_deviation"]) ? robustStats["standard_deviation"].min : minStandardDeviation,
                //   max: (isGlobalClampEnabled && clampConfig["spectral_deviation"]) ? robustStats["standard_deviation"].max : maxStandardDeviation,
                //   softclipping: isSoftclipEnabled
                // },
                // brightness: {
                //   val: () => feature.brightness,
                //   min: (isGlobalClampEnabled && clampConfig["brightness"]) ? robustStats["brightness"].min : minBrightness,
                //   max: (isGlobalClampEnabled && clampConfig["brightness"]) ? robustStats["brightness"].max : maxBrightness,
                //   softclipping: isSoftclipEnabled
                // },
                // mir_roughness_zwicker: {
                //   val: () => feature.mir_roughness_zwicker,
                //   min: (isGlobalClampEnabled && clampConfig["mir_roughness_zwicker"]) ? robustStats["mir_roughness_zwicker"].min : minMirRoughnessZwicker,
                //   max: (isGlobalClampEnabled && clampConfig["mir_roughness_zwicker"]) ? robustStats["mir_roughness_zwicker"].max : maxMirRoughnessZwicker,
                //   softclipping: isSoftclipEnabled
                // },
                // mir_roughness_sethares: {
                //   val: () => feature.mir_roughness_sethares,
                //   min: (isGlobalClampEnabled && clampConfig["mir_roughness_sethares"]) ? robustStats["mir_roughness_sethares"].min : minMirRoughnessSethares,
                //   max: (isGlobalClampEnabled && clampConfig["mir_roughness_sethares"]) ? robustStats["mir_roughness_sethares"].max : maxMirRoughnessSethares,
                //   softclipping: isSoftclipEnabled
                // },
                // weighted_spectral_centroid_bandwidth: {
                //   val: () => spectralCentroidWithBandwidthWeight(feature.weighted_spectral_centroid, feature.spectral_bandwidth),
                //   min: (isGlobalClampEnabled && clampConfig["weighted_spectral_centroid_bandwidth"]) ? robustStats["weighted_spectral_centroid_bandwidth"].min : minSpecralCentroidWeigthedBandwidth,
                //   max: (isGlobalClampEnabled && clampConfig["weighted_spectral_centroid_bandwidth"]) ? robustStats["weighted_spectral_centroid_bandwidth"].max : maxSpecralCentroidWeigthedBandwidth,
                //   softclipping: isSoftclipEnabled
                // },
                // centroid_peak_bandwidth: {
                //   val: () => computeBlendedTonalY(feature.spectral_peak, feature.weighted_spectral_centroid, feature.spectral_bandwidth),
                //   min: (isGlobalClampEnabled && clampConfig["centroid_peak_bandwidth"]) ? robustStats["centroid_peak_bandwidth"].min : minCentroidPeakBandwidth,
                //   max: (isGlobalClampEnabled && clampConfig["centroid_peak_bandwidth"]) ? robustStats["centroid_peak_bandwidth"].max : maxCentroidPeakBandwidth,
                //   softclipping: isSoftclipEnabled
                // },
                // centroid_peak_bandwidth_prominence: {
                //     val: () => computeTonalYWithProminenceDb({
                //         periodicity: feature.yin_periodicity,
                //         crepeF0: feature.crepe_f0,
                //         spectralCentroidHz: feature.weighted_spectral_centroid,
                //         spectralBandwidthHz: feature.spectral_bandwidth,
                //         localPeakHz: feature.spectral_peak,
                //         prominenceDb: feature.spectral_peak_prominence_db
                //       }),
                //     min: (isGlobalClampEnabled && clampConfig["centroid_peak_bandwidth_prominence"]) ? robustStats["centroid_peak_bandwidth_prominence"].min : minCentroidPeakBandwidthProminence,
                //     max: (isGlobalClampEnabled && clampConfig["centroid_peak_bandwidth_prominence"]) ? robustStats["centroid_peak_bandwidth_prominence"].max : maxCentroidPeakBandwidthProminence,
                //     softclipping: isSoftclipEnabled
                //   },
                // perceived_pitch: {
                //   val: () => perceivedPitch(feature.crepe_confidence, feature.crepe_f0, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch"]) ? robustStats["perceived_pitch"].min : minPerceivedPitch,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch"]) ? robustStats["perceived_pitch"].max : maxPerceivedPitch,
                //   softclipping: isSoftclipEnabled
                // },
                // perceived_pitch_librosa: {
                //   val: () => perceivedPitchLibrosa(feature.crepe_confidence, feature.yin_f0_librosa, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch_librosa"]) ? robustStats["perceived_pitch_librosa"].min : minPerceivedPitchLibrosa,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch_librosa"]) ? robustStats["perceived_pitch_librosa"].max : maxPerceivedPitchLibrosa,
                //   softclipping: isSoftclipEnabled
                // },
                // perceived_pitch_librosa_periodicity: {
                //   val: () => perceivedPitchLibrosaPeriodicity(feature.yin_periodicity, feature.yin_f0_librosa, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch_librosa_periodicity"]) ? robustStats["perceived_pitch_librosa_periodicity"].min : minPerceivedPitchLibrosaPeriodicity,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch_librosa_periodicity"]) ? robustStats["perceived_pitch_librosa_periodicity"].max : maxPerceivedPitchLibrosaPeriodicity,
                //   softclipping: isSoftclipEnabled
                // },
                // perceived_pitch_crepe_periodicity: {
                //   val: () => perceivedPitchCrepePeriodicity(feature.yin_periodicity, feature.crepe_f0, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch_crepe_periodicity"]) ? robustStats["perceived_pitch_crepe_periodicity"].min : minPerceivedPitchCrepePeriodicity,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch_crepe_periodicity"]) ? robustStats["perceived_pitch_crepe_periodicity"].max : maxPerceivedPitchCrepePeriodicity,
                //   softclipping: isSoftclipEnabled
                // },
                // perceived_pitch_f0_or_SC: {
                //   val: () => perceivedPitchF0OrSC(feature.yin_periodicity, feature.crepe_f0, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_or_SC"]) ? robustStats["perceived_pitch_f0_or_SC"].min : minPerceivedPitchF0OrSC,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_or_SC"]) ? robustStats["perceived_pitch_f0_or_SC"].max : maxPerceivedPitchF0OrSC,
                //   softclipping: isSoftclipEnabled
                // },
                // perceived_pitch_f0_candidates_periodicity: {
                //   val: () => perceivedPitchF0Candidates(feature.yin_periodicity, feature.f0_candidates, feature.spectral_centroid),
                //   min: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_candidates_periodicity"]) ? robustStats["perceived_pitch_f0_candidates_periodicity"].min : minPerceivedPitchF0Candidates,
                //   max: (isGlobalClampEnabled && clampConfig["perceived_pitch_f0_candidates_periodicity"]) ? robustStats["perceived_pitch_f0_candidates_periodicity"].max : maxPerceivedPitchF0Candidates,
                //   softclipping: isSoftclipEnabled
                // },
                // loudness_zcr: {
                //   val: () => feature.loudness * feature.zerocrossingrate,
                //   min: (isGlobalClampEnabled && clampConfig["loudness_zcr"]) ? robustStats["loudness_zcr"].min : 0,
                //   max: (isGlobalClampEnabled && clampConfig["loudness_zcr"]) ? robustStats["loudness_zcr"].max : maxLoudness * maxZerocrossingrate,
                //   softclipping: isSoftclipEnabled
                // },
            //   };
            
            // Y AXIS
            // const slider5 = document.getElementById("slider-5");
            // const isInverted_y_axis = document.getElementById("invertMapping-5")?.checked;
            // const {startRange_y_axis,endRange_y_axi} = calculateDynamicRange(slider5,isInverted_y_axis,"startRange_y_axis","endRange_y_axis");
            // const scaleInfo = getYAxisScaleFromConfig(featureConfig,selectedFeature5,isLogSelected);
            // let minValue,maxValue;
            // if(scaleInfo)
            // {
            //     const {min,max,labelFormatter} = scaleInfo;
            //     minValue = min;
            //     maxValue = max;
            //     drawYAxisScale(canvasHeight,minValue,maxValue,isLogSelected,padding,labelFormatter,isInverted_y_axis);
            // }
            // else
            // {
            //     console.warn(`Feature ${selectedFeature5} not found in config`);
            // }
            let yAxis = computeYAxisValue(feature,selectedFeature5,featureConfig,canvasHeight,padding,isInverted_y_axis,scale,minValue,maxValue,isSoftclipEnabled,softclipScale);
            // if(yAxis === null)
            // {
            //     yAxis = getMappedYAxisFallback(selectedFeature5,featureConfig,canvasHeight,padding,isInverted_y_axis,isLogSelected,minValue,maxValue,400,"spectral_centroid");
            // }
            
            // LINE LENGTH
            // const slider1 = document.getElementById("slider-1");
            // const isInverted_lineLength = document.getElementById("invertMapping-1")?.checked;
            // const {startRange_lineLength,endRange_lineLength} = calculateDynamicRange(slider1,isInverted_lineLength,"startRange_lineLength","endRange_lineLength");
            const lineLength = getMappedFeatureValue(feature,selectedFeature1,featureConfig,startRange_lineLength,endRange_lineLength,isSoftclipEnabled,softclipScale,1);

            // if (selectedFeature1 === "none")
            // {
            //     lineLength = 1;
            // }
            // else if (selectedFeature1 === "loudness-zcr") {
            // const loud = map(feature.loudness, 0, maxLoudness, 0, endRange_lineLength);
            // const zcrRatio = feature.zerocrossingrate / maxZerocrossingrate;
            // lineLength = startRange_lineLength + loud * zcrRatio;
            
            // } else if (selectedFeature1 === "loudness-periodicity") {
            // const loud = map(feature.loudness, minLoudness, maxLoudness, 0, endRange_lineLength);
            // const periodicityInverted = 1 - (feature.yin_periodicity / maxPeriodicity);
            // lineLength = startRange_lineLength + loud * periodicityInverted;
            
            // } else if (selectedFeature1 === "loudness-pitchConf") {
            // const loud = map(feature.loudness, minLoudness, maxLoudness, 0, endRange_lineLength);
            // const pitchConfInverted = 1 - (feature.crepe_confidence / maxF0CrepeConfidence);
            // lineLength = startRange_lineLength + loud * pitchConfInverted;
            // }
            // else if (featureConfig[selectedFeature1])
            // {
            //     const { val, min, max, softclipping } = featureConfig[selectedFeature1];

            //     const clamped = isGlobalClampEnabled
            //       ? Math.max(min, Math.min(val(), max))
            //       : val();
            
            //     lineLength = softclipping
            //       ? mapWithSoftClipping(clamped, min, max, startRange_lineLength, endRange_lineLength, softclipScale)
            //       : map(clamped, min, max, startRange_lineLength, endRange_lineLength);
            
            // }
            // else
            // {
            //     // fallback
            //     lineLength = map(feature.amplitude, 0, maxAmplitude, startRange_lineLength, endRange_lineLength);
            // }

            // LINE WIDTH
            // const slider2 = document.getElementById("slider-2");
            // const isInverted_lineWidth = document.getElementById("invertMapping-2")?.checked;
            // const {startRange_lineWidth,endRange_lineWidth } = calculateDynamicRange(slider2,isInverted_lineWidth,"startRange_lineWidth","endRange_lineWidth");
            const lineWidth = getMappedFeatureValue(feature,selectedFeature2,featureConfig,startRange_lineWidth,endRange_lineWidth,isSoftclipEnabled,softclipScale,1);

            // COLOR SATURATION
            // const slider3 = document.getElementById("slider-3");
            // const isInverted_colorSaturation = document.getElementById("invertMapping-3")?.checked;
            // const {startRange_colorSaturation,endRange_colorSaturation} = calculateDynamicRange(slider3,isInverted_colorSaturation,"startRange_colorSaturation","endRange_colorSaturation");
            let colorSaturation = getMappedFeatureValue(feature,selectedFeature3,featureConfig,startRange_colorSaturation,endRange_colorSaturation,isSoftclipEnabled,softclipScale,0);
            colorSaturation = Math.floor(colorSaturation);

            // COLOR LIGHTNESS
            // const slider6 = document.getElementById("slider-6");
            // const isInverted_colorLightness = document.getElementById("invertMapping-6")?.checked;
            // const {startRange_colorLightness,endRange_colorLightness} = calculateDynamicRange(slider6,isInverted_colorLightness,"startRange_colorLightness","endRange_colorLightness");
            let colorLightness = getMappedFeatureValue(feature,selectedFeature6,featureConfig,startRange_colorLightness,endRange_colorLightness,isSoftclipEnabled,softclipScale,50);
            colorLightness = Math.floor(colorLightness);
            
            // ANGLE
            // const slider4 = document.getElementById("slider-4");
            // const isInverted_angle = document.getElementById("invertMapping-4")?.checked;
            // const {startRange_angle,endRange_angle} = calculateDynamicRange(slider4,isInverted_angle,"startRange_angle","endRange_angle");
            let angleRange = getMappedFeatureValue(feature,selectedFeature4,featureConfig,startRange_angle,endRange_angle,isSoftclipEnabled,softclipScale,0);
            const angleLowerBound = (isInverted_angle ? endRange_angle : startRange_angle)
            if(angleRange >= angleLowerBound)
            {
                angleRange = map(rng(),0,1,angleLowerBound,angleRange);
            }
            let angleDegrees = 90 + getRandomSign()*angleRange;
            angle = (angleDegrees*Math.PI)/180;

            // DASH ARRAY
            // const slider7 = document.getElementById("slider-7");
            // const isInverted_dashArray = document.getElementById("invertMapping-7")?.checked;
            // const {startRange_dashArray,endRange_dashArray} = calculateDynamicRange(slider7,isInverted_dashArray,"startRange_dashArray","endRange_dashArray");
            let dashArray = getMappedFeatureValue(feature,selectedFeature7,featureConfig,startRange_dashArray,endRange_dashArray,isSoftclipEnabled,softclipScale,0);
            // const dashLowerBound = (isInverted_dashArray ? endRange_dashArray : startRange_dashArray)
            // if(dashArray >= dashLowerBound)
            // {
            //     dashArray = map(rng(),0,1,dashLowerBound,dashArray);
            // }

            // // Special handling for composite case
            // if (selectedFeature7 === "loudness-zcr")
            // {
            //     const MaxZCR = maxZerocrossingrate;

            //     const ZCRLoudnessEffect = map(
            //         feature.loudness,
            //         minLoudness,
            //         maxLoudness,
            //         startRange_dashArray,
            //         endRange_dashArray
            //     );

            //     const ZCRInfluence = feature.zerocrossingrate / MaxZCR;

            //     dashArray = startRange_dashArray + (ZCRLoudnessEffect * ZCRInfluence);
            // }
            // else
            // {
            //     dashArray = getMappedFeatureValue(
            //     selectedFeature7,
            //     featureConfig,
            //     startRange_dashArray,
            //     endRange_dashArray,
            //     0, // default
            //     "zerocrossingrate" // fallback
            //     );
            // }

            const max_loudness = featureConfig["loudness"].max;
            const normalized_loudness = clamp(feature["loudness"],0,max_loudness)/max_loudness;
            feature["normalized_loudness"] = normalized_loudness;

            // Objectifier Data
            feature["visual"] = {
                xAxis,
                yAxis,
                lineLength,
                lineWidth,
                colorSaturation,
                colorLightness,
                angle,
                dashArray,
            };

            if(!isObjectifyEnabled)
            {
                const loudness_threshold = document.getElementById("slider-gate").noUiSlider.get()/100;
                if (normalized_loudness > loudness_threshold)
                {
                    // Sonificators Data
                    const timestamp = feature["timestamp"];
                    pathData[fileIndex].push(
                    {
                        timestamp,
                        yAxis,
                        lineLength,
                        lineWidth,
                        colorHue,
                        colorSaturation,
                        colorLightness,
                        angle,
                        dashArray,
                    });

                    //! Dynamically creating description
                    let featureDescription = `Timestamp: ${feature["timestamp"].toFixed(2)}s<br>`;
                    for(let i = 0; i < rawFeatureNames.length; i++)
                    {
                        let entry = featureConfig[rawFeatureNames[i]];
                        const min_value = entry.min;
                        const max_value = entry.max;
                        let value = clamp(entry.val(feature),min_value,max_value)
                        if(isSoftclipEnabled && clampConfig[rawFeatureNames[i]])
                        {    
                            const shift = (entry.median - min_value)/(max_value - min_value);
                            value = mapWithSoftClipping(value,min_value,max_value,min_value,max_value,shift,softclipScale)
                        }
                        featureDescription += `${visibleFeatureNames[i]}: ${value.toFixed(2)}<br>`;
                    }
                    
                    if(isLineSketchingEnabled)
                    {
                        // Calculate end points of the line
                        let x1 = xAxis - (lineLength / 2) * Math.cos(angle);
                        let y1 = yAxis - (lineLength / 2) * Math.sin(angle);
                        let x2 = xAxis + (lineLength / 2) * Math.cos(angle);
                        let y2 = yAxis + (lineLength / 2) * Math.sin(angle);      
    
                        // Store the dot position for polyline
                        // pointsArray.push(`${xAxis},${yAxis}`);
    
                        let path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    
                        // if (isThresholdEnabled && feature.zerocrossingrate > zeroCrossingThreshold)
                        // {
                        //     // Draw multiple lines at different y heights (spectral centroids)
                        //     for (let j = 0; j < 100; j++)
                        //     { // Draw 5 lines at different y heights
                        //         // Calculate y position based on feature values
                        //         let yAxis = map((feature.spectral_centroid + 500)*j*0.01, 0, maxSpectralCentroid, canvasHeight, 0);
                        //         // Calculate end points of the line
                        //         //! Changed Math.random() to rng()
                        //         let x1 = xAxis - lineLength / 2 * Math.cos(angle);
                        //         let y1 = yAxis - lineLength / 2 * Math.sin(angle*rng());
                        //         let x2 = xAxis + lineLength / 2 * Math.cos(angle);
                        //         let y2 = yAxis + lineLength / 2 * Math.sin(angle*rng());
    
                        //         // Calculate control points for the Bezier curve
                        //         let ctrlX1 = xAxis - lineLength * Math.cos(angle) / 4; // Control point 1 xAxis-coordinate
                        //         let ctrlY1 = yAxis - lineLength * Math.sin(angle) / 2; // Control point 1 y-coordinate
                        //         let ctrlX2 = xAxis + lineLength * Math.cos(angle) / 4; // Control point 2 xAxis-coordinate
                        //         let ctrlY2 = yAxis + lineLength * Math.sin(angle) / 2; // Control point 2 y-coordinate
    
    
                        //         // Create a path element for the Bezier curve
                        //         let path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        //         path.setAttribute("d", `M${x1},${y1} C${ctrlX1},${ctrlY1} ${ctrlX2},${ctrlY2} ${x2},${y2}`);
                        //         path.setAttribute("stroke", `rgb(${colorValue},${colorValue},${colorValue})`);
                        //         path.setAttribute("stroke-width", lineWidth);
                        //         path.setAttribute("fill", "none");
                        //         // Generate a random number between 1 and 3
                        //         // let randomClassNumber = Math.floor(Math.random() * 3) + 1;
    
                        //         // Add CSS class for animation with the random number appended
                        //         // path.classList.add(`path-animation-${randomClassNumber}`);
                        //         svgContainer.appendChild(path);
                        //     }
                        // }
    
                        path.setAttribute("d", `M${x1},${y1} L${x2},${y2}`);    
                        path.setAttribute("stroke-linecap", "round"); // Make the line ends rounded
                        path.setAttribute("stroke-linejoin", "round"); // Make the line joins rounded
    
                        // Clamp dashArray to avoid negative values
                        // dashArray = Math.max(dashArray, 0);
    
                        // Apply the stroke-dasharray
                        if(dashArray == 0)
                        {
                            path.removeAttribute("stroke-dasharray"); // Solid line
                        }
                        else
                        {
                            const linepx = Math.max(dashArray,0.5);
                            const spacepx = linepx;
                            // const spacepx = linepx; 
                            // path.setAttribute("stroke-dasharray", "4,"+dashArray.toString()); // Dashed line with spacing
                            path.setAttribute("stroke-dasharray", linepx.toString() + ","+ spacepx.toString()); // Dashed line with spacing
                        }
                        // path.setAttribute("stroke", `rgb(${colorValue},${colorValue},${colorValue})`);
                        // Set the stroke color using HSL
                        // console.log('colorLightness', colorLightness);
    
                        path.setAttribute("stroke", `hsl(${colorHue}, ${colorSaturation}%, ${colorLightness}%)`); // 100% saturation, 50% lightness
                        path.setAttribute("stroke-width", lineWidth);
                        path.setAttribute("fill", "none");
                        path.setAttribute("data-features",featureDescription);
    
                        // path.classList.add(`audio-path-${fileIndex}`); // Add unique class
                        // path.style.display = isHidden ? "none" : "inline";
                        // svgContainer.appendChild(path);
                        pathGroup.appendChild(path);
    
                        if (isThresholdCircleEnabled)
                        {
                            // Calculate properties for circle
                            // let radius = map(feature.loudness, 0, maxLoudness, 2, 10); //! Changed amplitude with loudness
                            // colorValue = map(feature.spectral_centroid, 0, maxSpectralCentroid, 100, 0);
                            //! Maybe do this instead?
                            let radius = lineWidth/2 + 0;
                            let hue = hue1
    
                            let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                            circle.setAttribute("cx", xAxis);
                            circle.setAttribute("cy", yAxis);
                            circle.setAttribute("r", radius);
                            circle.setAttribute("fill",`hsl(${hue}, ${colorSaturation}%,${colorLightness}%)`);
                            // circle.setAttribute("fill", `rgb(0,0,${colorValue})`);
    
                            //! Added this
                            circle.setAttribute("data-features",featureDescription);
                            // circle.classList.add(`audio-path-${fileIndex}`); // Add unique class
                            // circle.style.display = isHidden ? "none" : "inline";
                            // svgContainer.appendChild(circle);
                            pathGroup.appendChild(circle);
    
                            //! Replaced spiral with solid circle to reduce overloading
                            let width = 4
                            radius = lineLength/2 + lineWidth/2 - width/2;
                            circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                            circle.setAttribute("cx", xAxis);
                            circle.setAttribute("cy", yAxis);
                            circle.setAttribute("r", radius);
                            circle.setAttribute("stroke",`hsl(${hue}, ${colorSaturation}%,${colorLightness}%)`);
                            circle.setAttribute("stroke-width",width)
                            circle.setAttribute("fill","none");
                            circle.setAttribute("data-features",featureDescription);
                            pathGroup.appendChild(circle);
    
                            // const TWO_PI = 2 * Math.PI;
                            // // Spiral pattern
                            // for (let t = 0; t < TWO_PI; t += 0.1)
                            // {
                            //     //! Maybe do this instead?
                            //     let radius = lineLength/2;
                            //     let x_spiral = xAxis + radius*Math.cos(t);
                            //     let y_spiral = yAxis + radius*Math.sin(t);
                            //     // let radius = map(feature.spectral_centroid, 0, maxSpectralCentroid, 2, 10);
                            //     // let x_spiral = xAxis + radius * Math.cos(t) * (1 + feature.loudness); //! Changed amplitude with loudness
                            //     // let y_spiral = yAxis + radius * Math.sin(t) * (1 + feature.loudness); //! Changed amplitude with loudness
    
                            //     let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                            //     circle.setAttribute("cx",x_spiral);
                            //     circle.setAttribute("cy",y_spiral);
                            //     circle.setAttribute("r",2);
                            //     circle.setAttribute("fill",`hsl(${hue}, ${colorSaturation}%,${colorLightness}%)`);
                            //     // circle.setAttribute("fill",`rgb(${colorValue}, ${200 - colorValue}, 255)`);
    
                            //     circle.setAttribute("data-features",featureDescription);
                            //     // circle.classList.add(`audio-path-${fileIndex}`); // Add unique class
                            //     // circle.style.display = isHidden ? "none" : "inline";
                            //     // svgContainer.appendChild(circle);
                            //     pathGroup.appendChild(circle);
                            // }
                        }
    
                        if (isJoinPathsEnabled)
                        {
                            const numDots = 10;
    
                            // Adjust scatter range based on loudness
                            // let scatterRange = map(feature.loudness, minLoudness, maxLoudness, 0, 200); // Scale range based on loudness
                            let scatterRange = lineLength; //! Maybe like this this instead?
                            
                            let currentDots = []; // To store the scattered dots of the current data point
    
                            // Scatter multiple dots along the y-axis
                            //! Some changes here
                            for (let j = 0; j < numDots; j++)
                            {
                                let offset = j*scatterRange/(numDots - 1) - scatterRange/2;
                                let scatteredX = xAxis + offset*Math.cos(angle);
                                let scatteredY = yAxis + offset*Math.sin(angle);
    
                                // Scatter dots within the specified range around the main y-axis value
                                // let offsetY = (j - (numDots - 1) / 2) * (scatterRange / (numDots - 1)); // Even spacing
                                // let scatteredY = yAxis + offsetY;
    
                                // // Create a scattered dot
                                // let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                                // circle.setAttribute("cx", scatteredX);
                                // circle.setAttribute("cy", scatteredY);
                                // circle.setAttribute("r", 1); // Radius of each dot
                                // circle.setAttribute("fill", "none"); // Gradient color
                                // // svgContainer.appendChild(circle);
                                // pathGroup.appendChild(circle);
    
                                // Store the current dot's position
                                currentDots.push({xAxis : scatteredX,y : scatteredY});
                            }
    
                            // Connect the dots with lines to the previous data point
                            if (previousDots.length > 0)
                            {
                                //! Interpolating colors along the minor arc of the hue circle
                                let hue_diff = (hue2 - hue1 + 360) % 360;
                                if(hue_diff > 180)
                                {
                                    hue_diff -= 360
                                }
                                for (let j = 0; j < numDots; j++)
                                {
                                    let hue = (hue1 + j*hue_diff/(numDots - 1) + 360) % 360
                                    let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                                    line.setAttribute("x1", previousDots[j].xAxis);
                                    line.setAttribute("y1", previousDots[j].y);
                                    line.setAttribute("x2", currentDots[j].xAxis);
                                    line.setAttribute("y2", currentDots[j].y);
                                    line.setAttribute("stroke",`hsl(${hue}, 100%,50%)`); // Line color
                                    // line.setAttribute("stroke", `rgb(${j * (255 / numDots)}, 100, 200)`); // Line color
                                    line.setAttribute("stroke-width", 1);
                                    // line.classList.add(`audio-path-${fileIndex}`); // Add unique class
                                    // line.style.display = isHidden ? "none" : "inline";
                                    // svgContainer.appendChild(line);
                                    pathGroup.appendChild(line);
                                }
                            }
                            previousDots = currentDots;
                        }
                    }
                    //! Separated this from isLineSketchingEnabled
                    else if(isPolygonEnabled)
                    {
                        // ✅ Validate and clamp number of sides
                        const polygonCorners = Math.max(3, Math.floor(lineWidth)); // Minimum triangle
                        const polygonRadius = lineLength;
                        const polygonPath = generatePolygonPath(xAxis, yAxis, polygonCorners, polygonRadius);
                        const strokeWidth = 1;
                        const opacity = 1;
                        const skewX = -(angleDegrees - 90);
                        const texture = dashArray/100;
    
                        // 💡 Use HSL for dynamic coloring with added opacity
                        const fillColor = `hsl(${colorHue}, ${colorSaturation}%, ${colorLightness}%)`;
                        const strokeColor = `hsl(${colorHue}, ${colorSaturation}%, ${Math.max(colorLightness - 20,0)}%)`;
                        // const patternColor = `hsl(${colorHue}, ${Math.max(colorSaturation - 20,0)}%, ${Math.max(colorLightness - 20,0)}%)`;
                        const patternColor = `hsl(${colorHue}, ${Math.max(colorSaturation - 20,0)}%, ${Math.min(colorLightness + 20,100)}%)`;
    
                        const {pattern,patternId} = createPattern(texture,xAxis,yAxis,polygonRadius,opacity,fillColor,patternColor);
                        defs.appendChild(pattern);
    
                        const polygonContainer = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        polygonContainer.setAttribute("transform",`translate(${xAxis},${yAxis}) skewX(${skewX}) translate(${-xAxis},${-yAxis})`);
    
                        // 🎨 Create the polygon element
                        const polygonElement = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        polygonElement.setAttribute("d", polygonPath);
                        polygonElement.setAttribute("fill", `url(#${patternId})`);
                        // polygonElement.setAttribute("fill", fillColor);
                        // polygonElement.setAttribute("fill-opacity", opacity);  // ⬅️ subtle transparency
                        polygonElement.setAttribute("stroke", strokeColor);
                        polygonElement.setAttribute("stroke-width",strokeWidth);
                        polygonElement.setAttribute("stroke-opacity", opacity); // ⬅️ optional softer stroke
                        // polygonElement.setAttribute("stroke-linejoin", "round"); // ⬅️ rounded joins
    
                        //! Added this
                        polygonElement.setAttribute("data-features",featureDescription);
    
                        // 🖼️ Append to SVG
                        polygonContainer.appendChild(polygonElement);
                        pathGroup.appendChild(polygonContainer);
                        // svgContainer.appendChild(polygonContainer);
                    }
                } 
            }
        }
        if(!isObjectifyEnabled)
        {
            svgContainer.appendChild(pathGroup);
        }
        else
        {
            drawClusterOverlays(audioData.clusters,audioData.features,svgContainer,canvasWidth,canvasHeight,maxDuration,fileIndex);
        }
    });

    if(isPolygonEnabled)
    {
        createPattern.counter = 0;
    }

    //! Draw y axis on top of everything
    if(yAxisFormatter)
    {
        drawYAxisScale(canvasHeight,minValue,maxValue,scale,padding,yAxisFormatter,isInverted_y_axis);
    }

    prepareSynthData(pathData);
}

//! -------------------------------------------------------------------
//! Sonificators stuff (currently being moved to sonificators.js)
//! -------------------------------------------------------------------

// Step 2: Compute min-max ranges
function computeMinMaxRanges(data) {
    const minMaxValues = {
        yAxis: { min: Infinity, max: -Infinity },
        lineLength: { min: Infinity, max: -Infinity },
        lineWidth: { min: Infinity, max: -Infinity },
        colorSaturation: { min: Infinity, max: -Infinity },
        colorLightness: { min: Infinity, max: -Infinity },
        dashArray: { min: Infinity, max: -Infinity },
    };

    data.forEach((item) => {
        minMaxValues.yAxis.min = Math.min(minMaxValues.yAxis.min, item.yAxis);
        minMaxValues.yAxis.max = Math.max(minMaxValues.yAxis.max, item.yAxis);
        minMaxValues.lineLength.min = Math.min(minMaxValues.lineLength.min, item.lineLength);
        minMaxValues.lineLength.max = Math.max(minMaxValues.lineLength.max, item.lineLength);
        minMaxValues.lineWidth.min = Math.min(minMaxValues.lineWidth.min, item.lineWidth);
        minMaxValues.lineWidth.max = Math.max(minMaxValues.lineWidth.max, item.lineWidth);
        minMaxValues.colorSaturation.min = Math.min(minMaxValues.colorSaturation.min, item.colorSaturation);
        minMaxValues.colorSaturation.max = Math.max(minMaxValues.colorSaturation.max, item.colorSaturation);
        minMaxValues.colorLightness.min = Math.min(minMaxValues.colorLightness.min, item.colorLightness);
        minMaxValues.colorLightness.max = Math.max(minMaxValues.colorLightness.max, item.colorLightness);
        minMaxValues.dashArray.min = Math.min(minMaxValues.dashArray.min, item.dashArray);
        minMaxValues.dashArray.max = Math.max(minMaxValues.dashArray.max, item.dashArray);
    });

    return minMaxValues;
}

// Step 3: Normalize attributes
function normalizePathData(data, minMaxValues) {
    return data.map((item) => ({
        ...item,
        yAxis: normalize(item.yAxis, minMaxValues.yAxis.min, minMaxValues.yAxis.max),
        lineLength: normalize(item.lineLength, minMaxValues.lineLength.min, minMaxValues.lineLength.max),
        lineWidth: normalize(item.lineWidth, minMaxValues.lineWidth.min, minMaxValues.lineWidth.max),
        colorSaturation: normalize(item.colorSaturation, minMaxValues.colorSaturation.min, minMaxValues.colorSaturation.max),
        colorLightness: normalize(item.colorLightness, minMaxValues.colorLightness.min, minMaxValues.colorLightness.max),
        dashArray: normalize(item.dashArray, minMaxValues.dashArray.min, minMaxValues.dashArray.max),
    }));
}

// Utility: Normalize function
function normalize(value, min, max) {
    return (value - min) / (max - min);
}

// Usage
function preparePathData(pathData) {
    // computeRawPathData(features);
    const minMaxValues = computeMinMaxRanges(pathData);
    normalisedPathData = normalizePathData(pathData, minMaxValues);
}
// preparePathData(pathData);

let audioContexttmp;
let audioBuffer;
let animationFrameId;
let grainSourcePosition = 0; // Default grain source position
let activeGrains = []; // To track active grains and stop them
let masterGainNode; // Master gain node for volume control
let playbackMarkerX = 0; // Position of the playback marker
let isPlayingtmp = false; // Playback state

// const canvas = document.getElementById("waveformCanvas");
// const ctx = canvas.getContext("2d");

// Draw the waveform
// function drawWaveform(buffer) {
//     const data = buffer.getChannelData(0); // Use the first channel
//     const canvasWidth = canvas.width;
//     const canvasHeight = canvas.height;

//     ctx.clearRect(0, 0, canvasWidth, canvasHeight);

//     const step = Math.ceil(data.length / canvasWidth);
//     const amp = canvasHeight / 2;

//     ctx.beginPath();
//     ctx.moveTo(0, amp);
//     for (let i = 0; i < canvasWidth; i++) {
//         const min = Math.min(...data.slice(i * step, (i + 1) * step));
//         const max = Math.max(...data.slice(i * step, (i + 1) * step));
//         ctx.lineTo(i, (1 + max) * amp);
//         ctx.lineTo(i, (1 + min) * amp);
//     }
//     ctx.lineTo(canvasWidth, amp);
//     ctx.strokeStyle = "blue";
//     ctx.stroke();
// }

// Draw the red line (playback marker)
function drawRedLine(position) {
    const canvasWidth = canvas.width;
    const xAxis = (position / audioBuffer.duration) * canvasWidth;

    ctx.save();
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3; // Make it bolder
    ctx.beginPath();
    ctx.moveTo(xAxis, 0);
    ctx.lineTo(xAxis, canvas.height);
    ctx.stroke();
    ctx.restore();
}

// Draw markers for grain spread
function drawSpreadMarkers(grainSource, spread) {
    const canvasWidth = canvas.width;
    const grainWidth = canvasWidth / audioBuffer.duration; // Width of a grain in canvas pixels

    const startX = (grainSource - spread / 2) * grainWidth;
    const endX = (grainSource + spread / 2) * grainWidth;

    ctx.save();
    ctx.fillStyle = "rgba(255, 0, 0, 0.2)"; // Transparent red
    ctx.fillRect(Math.max(0, startX), 0, Math.min(endX - startX, canvasWidth), canvas.height);
    ctx.restore();
}

// Update the canvas
function updateCanvas(currentGrainSource, grainSpread) {
    drawWaveform(audioBuffer);

    if (playbackSpreads.length > 0) {
        playbackSpreads.forEach(({ grainSource, spread }) => {
            drawSpreadMarkers(grainSource, spread);
        });
    }

    // Draw the dynamic red line
    drawRedLine(currentGrainSource);
}

//! Temporarily commented out
// // Handle grain source position on click
// canvas.addEventListener("click", (event) => {
//     const rect = canvas.getBoundingClientRect();
//     const xAxis = event.clientX - rect.left; // X coordinate relative to canvas
//     grainSourcePosition = (xAxis / canvas.width) * audioBuffer.duration; // Map to audio duration

//     playbackSpreads = []; // Reset spreads
//     updateCanvas(grainSourcePosition, 0); // Redraw with updated grain position
// });

// Animate grain spread and marker during playback
function animateGrainSpread(totalDuration, grainSpread) {
    const startTime = performance.now();

    function update() {
        const elapsedTime = (performance.now() - startTime) / 1000; // Convert to seconds

        if (elapsedTime > totalDuration || !isPlayingtmp) {
            isPlayingtmp = false;
            return;
        }

        // Calculate dynamic grain source based on spread
        const currentGrainSource = grainSourcePosition + Math.sin(elapsedTime * 2 * Math.PI) * grainSpread / 2;

        // Update spreads and redraw canvas
        playbackSpreads = [
            {
                grainSource: currentGrainSource,
                spread: grainSpread,
            },
        ];
        updateCanvas(currentGrainSource, grainSpread);

        requestAnimationFrame(update);
    }

    update();
}

//! Temporarily commented out
// // Load the waveform after audio is decoded
// document.getElementById("audioFileInput").addEventListener("change", function (event) {
//     const file = event.target.files[0];
//     if (file) {
//         const reader = new FileReader();
//         reader.onload = function () {
//             const audioContexttmp = new (window.AudioContext || window.webkitAudioContext)();
//             audioContexttmp.decodeAudioData(reader.result, (buffer) => {
//                 audioBuffer = buffer;
//                 drawWaveform(audioBuffer); // Draw the waveform once the buffer is loaded
//             });
//         };
//         reader.readAsArrayBuffer(file);
//     }
// });

// // Start playback and animate grain spread
// document.getElementById("playGranulator").addEventListener("click", function () {
//     if (!audioBuffer) {
//         alert("Please load an audio file first!");
//         return;
//     }

//     const totalDuration = audioBuffer.duration; // Assume duration based on audio file
//     isPlayingtmp = true;

//     const grainSpread = 0.2; // Define spread in seconds
//     // animateGrainSpread(totalDuration, grainSpread);
// });

//! Temporarily commented out
// // Load and decode the audio file
// document.getElementById("audioFileInput").addEventListener("change", function (event) {
//     const file = event.target.files[0];
//     if (file) {
//         const reader = new FileReader();
//         reader.onload = function () {
//             audioContexttmp = new (window.AudioContext || window.webkitAudioContext)();
//             audioContexttmp.decodeAudioData(reader.result, (buffer) => {
//                 audioBuffer = buffer;
//                 drawWaveform(audioBuffer); // Draw the waveform once the buffer is loaded


//                 // Initialize masterGainNode
//                 masterGainNode = audioContexttmp.createGain();
//                 masterGainNode.gain.value = 0.8; // Default volume, adjust as needed
//                 masterGainNode.connect(audioContexttmp.destination);

//                 console.log("Audio loaded and ready for granulation.");
//             });
//         };
//         reader.readAsArrayBuffer(file);
//     }
// });

// Draw vertical progress line
function drawVerticalLine() {
    const svgCanvas = document.getElementById("svgCanvas");
    if (!svgCanvas) return;

    let line = document.getElementById("progressLine");
    if (!line) {
        line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("id", "progressLine");
        line.setAttribute("x1", 0);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", 0);
        line.setAttribute("y2", svgCanvas.getBoundingClientRect().height);
        line.setAttribute("stroke", "red");
        line.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(line);
    }
}

// Update the position of the progress line
function updateLinePosition(currentTimestamp, totalDuration, canvasWidth, isScrollableMode) {
    const line = document.getElementById("progressLine");
    if (!line) return;

    const xAxis = (currentTimestamp / totalDuration) * canvasWidth;
    line.setAttribute("x1", xAxis);
    line.setAttribute("x2", xAxis);
    // 👇 NEW: If scrollable mode is enabled, scroll horizontally
    console.log("isScrollableMode", isScrollableMode);
    if (isScrollableMode) {
        const scrollOffset = xAxis - window.innerWidth / 2;
        window.scrollTo({ left: scrollOffset, behavior: 'smooth' });
    }
}

// Animate the progress line based on path data
function smoothUpdateLinePositionForPathData(totalDuration) {
    const svgCanvas = document.getElementById("svgCanvas");
    const canvasWidth = svgCanvas.getBoundingClientRect().width;

    let startTime = performance.now(); // High-resolution timer

    function update() {
        const elapsedTime = (performance.now() - startTime) / 1000; // Seconds
        if (elapsedTime > totalDuration) {
            cancelAnimationFrame(animationFrameId);
            stopAllGrains(); // Stop any remaining grains
            return;
        }
        isScrollableMode = document.getElementById("scrollModeToggle").checked;
        updateLinePosition(elapsedTime, totalDuration, canvasWidth, isScrollableMode);

        function calculatePlaybackRate(yAxis, isInverted = false) {
            // const normalizedY = yAxis / 800; // Assuming the canvas height is 800px
            const playbackRate = isInverted 
                ? map(yAxis, 0, 1, 2, 0.5) // Inverted: High yAxis -> Low playback rate
                : map(yAxis, 0, 1, 0.5, 2); // Normal: High yAxis -> High playback rate
            return playbackRate;
        }

        // Find paths that should trigger grains
        normalisedPathData.forEach((path) => {
            if (elapsedTime >= path.timestamp && elapsedTime < path.timestamp + 0.1) {
                const playbackRate = calculatePlaybackRate(path.yAxis, true); // Pass `true` for inversion

                playGrain(grainSourcePosition, path.dashArray, path.lineWidth, path.lineLength   * 0.05, playbackRate);
            }
        });

        animationFrameId = requestAnimationFrame(update);
    }

    update();
}

// Play a single grain
function playGrain(baseSourcePosition, dashArraySpread, duration, volume, playbackRate) {
    const source = audioContexttmp.createBufferSource();
    source.buffer = audioBuffer;

    // Calculate a randomized source position based on the dashArray spread
    const randomSpread = (Math.random() - 0.5) * dashArraySpread; // Spread around the base position
    let grainSourcePosition = baseSourcePosition + randomSpread;

    // Ensure the position loops within the audio buffer's duration
    grainSourcePosition = Math.max(0, grainSourcePosition % audioBuffer.duration);

    source.playbackRate.value = playbackRate;

    const gainNode = audioContexttmp.createGain();
    const now = audioContexttmp.currentTime;

    // Apply fade-in and fade-out for smoother transitions
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(volume, now + duration * 0.1);
    gainNode.gain.setValueAtTime(volume, now + duration * 0.9);
    gainNode.gain.linearRampToValueAtTime(0, now + duration);

    // Connect the grain to the master gain node
    source.connect(gainNode).connect(masterGainNode);

    source.start(now, grainSourcePosition, duration);
    source.stop(now + duration);

    // Debugging output
    console.log(
        `Grain started at ${grainSourcePosition}s with duration ${duration}s, volume ${volume}, playbackRate ${playbackRate}`
    );

    activeGrains.push(source);

    setTimeout(() => {
        activeGrains = activeGrains.filter((grain) => grain !== source);
    }, duration * 1000);
}

// Stop all active grains
function stopAllGrains() {
    activeGrains.forEach((grain) => grain.stop());
    activeGrains = [];
}

// Start granulation process
function startGranulation() {
    if (!audioBuffer) {
        console.error("No audio loaded.");
        return;
    }

    if (normalisedPathData.length === 0) {
        console.error("Path data not computed. Please compute path attributes first.");
        return;
    }

    const totalDuration = normalisedPathData[normalisedPathData.length - 1].timestamp;
    drawVerticalLine();
    smoothUpdateLinePositionForPathData(totalDuration);
}

//! Temporarily commented out
// // Map features to path attributes and start granulation
// document.getElementById("playGranulator").addEventListener("click", function () {
//     if (!audioBuffer) {
//         alert("Please load an audio file first!");
//         return;
//     }

//     startGranulation();
// });

// // Adjust grain source position using slider
// document.getElementById("grainSourcePositionSlider").addEventListener("input", function (event) {
//     grainSourcePosition = parseFloat(event.target.value);
//     console.log(`Grain source position set to ${grainSourcePosition} seconds.`);
// });


// Define variables for the synth
// let synthAudioContext = new (window.AudioContext || window.webkitAudioContext)();
// let synthIsPlaying = false; // To manage the playback state
// let synthPathData = []; // Store normalized path data for the synth
// let selectedWaveform = "sine"; // Default waveform
// synthPathData = pathData;

// Play a wave with specified parameters
// function playSynthWave(frequency, volume, duration, distortion) {
//     const oscillator = synthAudioContext.createOscillator();
//     const gainNode = synthAudioContext.createGain();
//     const waveShaper = synthAudioContext.createWaveShaper();

//     // Configure oscillator
//     oscillator.type = selectedWaveform; // Use the selected waveform
//     oscillator.frequency.value = frequency;

//     // Configure gain with smooth fade-in and fade-out
//     const now = synthAudioContext.currentTime;
//     gainNode.gain.setValueAtTime(0, now); // Start at 0 volume
//     gainNode.gain.linearRampToValueAtTime(volume, now + 0.01); // Fade-in
//     gainNode.gain.setValueAtTime(volume, now + duration - 0.01); // Sustain
//     gainNode.gain.linearRampToValueAtTime(0, now + duration); // Fade-out

//     // Configure distortion
//     waveShaper.curve = createDistortionCurve(distortion);
//     waveShaper.oversample = "4x";

//     // Connect nodes
//     oscillator.connect(waveShaper).connect(gainNode).connect(synthAudioContext.destination);

//     // Start and stop oscillator
//     oscillator.start(now);
//     oscillator.stop(now + duration);

//     console.log(
//         `Synth Wave: Waveform ${selectedWaveform}, Frequency ${frequency}Hz, Volume ${volume}, Duration ${duration}s, Distortion ${distortion}`
//     );
// }

// Generate a distortion curve
// function createDistortionCurve(amount) {
//     const curve = new Float32Array(44100);
//     const k = typeof amount === "number" ? amount : 50;
//     const deg = Math.PI / 180;
//     for (let i = 0; i < 44100; i++) {
//         const xAxis = (i * 2) / 44100 - 1;
//         curve[i] = ((3 + k) * xAxis * 20 * deg) / (Math.PI + k * Math.abs(xAxis));
//     }
//     return curve;
// }

// Play the sketch using the synth
// function playSketchWithSynth() {
//     if (!synthPathData || synthPathData.length === 0) {
//         alert("No sketch data available to play!");
//         return;
//     }
//     playOscillatorButton.textContent = "Stop Synth"; // Update button text

//     synthIsPlaying = true;

//     synthPathData.forEach((path) => {
//         const frequency = mapSynth(path.yAxis, canvasHeight, 0, 100, 1000); // Map yAxis (reverse) to pitch
//         const volume = mapSynth(path.lineLength, 0, 100, 0.1, 1); // Map lineLength to volume
//         const duration = mapSynth(path.lineWidth, 0, 10, 0.1, 1); // Map lineWidth to duration
//         const distortion = mapSynth(path.dashArray, 0, 50, 0, 100); // Map dashArray to distortion

//         setTimeout(() => {
//             if (synthIsPlaying) {
//                 playSynthWave(frequency, volume, duration, distortion);
//             }
//         }, path.timestamp * 1000); // Trigger based on timestamp
//     });
// }

// // Map utility function for the synth
// function mapSynth(value, inMin, inMax, outMin, outMax) {
//     return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
// }

// // Stop all synth playback
// function stopSynthPlayback() {
//     synthIsPlaying = false;
//     console.log("Synth playback stopped.");
// }

// Handle waveform selection
// document.getElementById("waveformSelect").addEventListener("change", (event) => {
//     // selectedWaveform = event.target.value;
//     console.log(`Waveform changed to ${selectedWaveform}`);
// });

// const playOscillatorButton = document.getElementById("playOscillator");

// // Update waveform image and synth waveform
// function updateWaveform(waveform) {
//     // Update the displayed image
//     const waveformImage = document.getElementById("waveformImage");
//     waveformImage.src = waveformImages[waveform];
//     waveformImage.alt = `${waveform.charAt(0).toUpperCase() + waveform.slice(1)} Waveform`;

//     // Update the selected waveform for the synth
//     selectedWaveform = waveform;
//     console.log(`Waveform changed to ${selectedWaveform}`);
// }

// // Handle arrow button clicks
// document.getElementById("waveformPrev").addEventListener("click", () => {
//     const selectElement = document.getElementById("waveformSelect");
//     if (selectElement.selectedIndex > 0) {
//         selectElement.selectedIndex -= 1;
//         updateWaveform(selectElement.value);
//     }
// });

// document.getElementById("waveformNext").addEventListener("click", () => {
//     const selectElement = document.getElementById("waveformSelect");
//     if (selectElement.selectedIndex < selectElement.options.length - 1) {
//         selectElement.selectedIndex += 1;
//         updateWaveform(selectElement.value);
//     }
// });

// // Toggle play and stop functionality
// playOscillatorButton.addEventListener("click", () => {
//     if (synthIsPlaying) {
//         // selectedWaveform = event.target.value;
//         console.log(selectedWaveform)
//         stopSynthPlayback();
//     } else {
//         playSketchWithSynth();
//     }
// });
// Function to stop the synth
// function stopSynthPlayback() {
//     synthIsPlaying = false;
//     playOscillatorButton.textContent = "Play Synth"; // Update button text

//     // Your existing stop logic here
//     // clearInterval(synthPlaybackInterval); // Clear playback interval
//     console.log("Synth playback stopped.");
// }

// // Add event listener for the synth play button
// document.getElementById("playSynth").addEventListener("click", playSketchWithSynth);

// // Add event listener for stopping synth playback
// document.getElementById("stopSynth").addEventListener("click", stopSynthPlayback);


//! -------------------------------------------------------------------
//! Objectifier stuff (moved to objectifier.js)
//! -------------------------------------------------------------------

// // ===============================
// // 🎯 Assign features to regions before visualization
// // ===============================
// function assignFeaturesToRegions(clusters, features) {
//     clusters.forEach(cluster => {
//         cluster.regions.forEach(region => {region.features = features.filter(f =>
//                 f.timestamp >= region.start_time && f.timestamp <= region.end_time
//             );
//         });
//     });
// }

// // ===============================
// // 📐 Precompute layout and feature summaries per region
// // ===============================
// function computeVisualDataForRegions(clusters, maxDuration, canvasWidth) {
//     clusters.forEach(cluster => {
//         cluster.regions.forEach(region => {
//             const startX = map(region.start_time, 0, maxDuration, 0, canvasWidth);
//             const endX = map(region.end_time, 0, maxDuration, 0, canvasWidth);
//             const width = endX - startX;

//             const avg = computeAverageFeatures(region.features || []); region["visual"] = { startX, width, avg };
//         });
//     });
// }

// function drawUnifiedRegionPathFromVisual(svg, region, maxDuration, canvasWidth, baseColor = "hsl(0, 70%, 50%)", loudnessThreshold = 1, draw) {
//     const regionFrames = region.features;

//     const avgRoughness = regionFrames.reduce((sum, f) => sum + (f.visual?.roughness ?? 0.3), 0) / regionFrames.length;
//     const pattern = createAdaptiveTexturePattern(draw, avgRoughness, `${region.start_time}-${region.end_time}`);
//     const grainPattern = createGrainTexturePattern(draw, avgRoughness, `${region.start_time}-${region.end_time}`);


//     if (!regionFrames || regionFrames.length < 2) return;

//     const yMin = 100, yMax = draw.height() - 100;

//     const perceivedOpacity = loudness => {
//         return Math.max(0, Math.min(1, (loudness - 0.2) / (loudnessThreshold - 0.2)));
//     };

//     const topPoints = [];
//     const bottomPoints = [];

//     for (let i = 0; i < regionFrames.length; i++) {
//         const f = regionFrames[i];
//         if (!f.visual) continue;

//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual.yAxis;
//         const height = f.visual.lineLength ?? 60;
//         const mod = f.visual.mod ?? 0;

//         const yTop = y - height / 2 - mod;
//         const yBot = y + height / 2 + mod;

//         topPoints.push([xAxis, yTop]);
//         bottomPoints.unshift([xAxis, yBot]);
//     }

//     if (topPoints.length < 2 || bottomPoints.length < 2) return;

//     const fullPoints = [...topPoints, ...bottomPoints];

//     const pathData = catmullRomToPath(fullPoints);  // Smooth closed path

//     const avgLoudness = regionFrames.reduce((acc, f) => acc + (f.loudness ?? 0), 0) / regionFrames.length;
//     const alpha = perceivedOpacity(avgLoudness);

//     draw.path(pathData)
//         .fill({ color: "#000", opacity: 0 })
//         .opacity(alpha)
//         .stroke({ width: 0.5, color: "#000", opacity: 0.1 });
// }

// function drawUnifiedRegionPathFromVisual(
//     svg,
//     region,
//     maxDuration,
//     canvasWidth,
//     baseColor = "hsl(0, 70%, 50%)",
//     loudnessThreshold = 1,
//     draw
//   ) {
//     const regionFrames = region.features;
  
//     if (!regionFrames || regionFrames.length < 2) return;
  
//     const avgRoughness = regionFrames.reduce(
//       (sum, f) => sum + (f.visual?.roughness ?? 0.3),
//       0
//     ) / regionFrames.length;
//     const pattern = createAdaptiveTexturePattern(
//       draw,
//       avgRoughness,
//       `${region.start_time}-${region.end_time}`
//     );
//     const grainPattern = createGrainTexturePattern(
//       draw,
//       avgRoughness,
//       `${region.start_time}-${region.end_time}`
//     );
  
//     // === Robust stats ===
//     const ys = regionFrames.map(f => f.visual?.yAxis ?? 0);
//     const ysSorted = [...ys].sort((a, b) => a - b);
//     const medianY = ysSorted[Math.floor(ysSorted.length / 2)];
//     const p10 = ysSorted[Math.floor(ysSorted.length * 0.1)];
//     const p90 = ysSorted[Math.floor(ysSorted.length * 0.9)];
  
//     const lengths = regionFrames.map(f => f.visual?.lineLength ?? 60);
//     const lengthsSorted = [...lengths].sort((a, b) => a - b);
//     const medianLength = lengthsSorted[Math.floor(lengthsSorted.length / 2)];
//     const p10Length = lengthsSorted[Math.floor(lengthsSorted.length * 0.1)];
//     const p90Length = lengthsSorted[Math.floor(lengthsSorted.length * 0.9)];
  
//     const topPoints = [];
//     const bottomPoints = [];
  
//     for (let i = 0; i < regionFrames.length; i++) {
//       const f = regionFrames[i];
//       if (!f.visual) continue;
  
//       const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
  
//       // === Clamp yAxis and lineLength to robust range ===
//       let y = f.visual.yAxis;
//       y = clamp(y, p10, p90);
  
//       let height = f.visual.lineLength ?? medianLength;
//       height = clamp(height, p10Length, p90Length);
  
//       const mod = f.visual.mod ?? 0;
  
//       const yTop = y - height / 2 - mod;
//       const yBot = y + height / 2 + mod;
  
//       topPoints.push([xAxis, yTop]);
//       bottomPoints.unshift([xAxis, yBot]);
//     }
  
//     if (topPoints.length < 2 || bottomPoints.length < 2) return;
  
//     const fullPoints = [...topPoints, ...bottomPoints];
  
//     const pathData = catmullRomToPath(fullPoints); // Smooth closed path
  
//     const avgLoudness =
//       regionFrames.reduce((acc, f) => acc + (f.loudness ?? 0), 0) /
//       regionFrames.length;
//     const perceivedOpacity = loudness =>
//       Math.max(0, Math.min(1, (loudness - 0.2) / (loudnessThreshold - 0.2)));
//     const alpha = perceivedOpacity(avgLoudness);
  
//     draw.path(pathData)
//       .fill({ color: "#000", opacity: 0 })
//       .opacity(alpha)
//       .stroke({ width: 0.5, color: "#000", opacity: 0.1 });
//   }


// function createGrainTexturePattern(draw, roughness, regionId) {
//     const patternId = `grain-${regionId}`;

//     const patternSize = 14 + (1 - roughness) * 12;
//     const density = Math.floor(40 + roughness * 120);
//     const baseOpacity = 0.08 + roughness * 0.4;
//     const maxRadius = 1.8 - roughness * 1.2;

//     const pattern = draw.pattern(patternSize, patternSize, function (add) {
//         if (roughness < 0.2) {
//             // ✨ Smooth fill — fog-like micro blur
//             add.rect(patternSize, patternSize)
//                 .fill('#333')
//                 .opacity(0.02 + (1 - roughness) * 0.06);
//         }

//         for (let i = 0; i < density; i++) {
//             const xAxis = Math.random() * patternSize;
//             const y = Math.random() * patternSize;
//             const r = Math.random() * maxRadius + 0.4;

//             add.circle(r * 2)
//                 .center(xAxis, y)
//                 .fill('#222')
//                 .opacity(baseOpacity);
//         }
//     });

//     pattern.id(patternId);
//     return pattern;
// }



// ===============================
// 📌 Draw overlays with expressive shapes per region
// ===============================
// function drawClusterOverlays(clusters, features, svgContainer, canvasWidth, canvasHeight, maxDuration) {
//     assignFeaturesToRegions(clusters, features);
//     computeVisualDataForRegions(clusters, maxDuration, canvasWidth);

//     console.log(features)

//     assignFeaturesToRegions(clusters, features);
//     const draw = SVG().addTo('#svgCanvas').size('100%', canvasHeight);

//     //defineGlobalPatterns(draw);


//     clusters.forEach(cluster => {
//         cluster.regions.forEach(region => {
//             const startX = map(region.start_time, 0, maxDuration, 0, canvasWidth);
//             const endX = map(region.end_time, 0, maxDuration, 0, canvasWidth);
//             const width = endX - startX;

//             const overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect");
//             overlay.setAttribute("xAxis", startX);
//             overlay.setAttribute("y", 0);
//             overlay.setAttribute("width", width);
//             overlay.setAttribute("height", canvasHeight);
//             overlay.setAttribute("fill", cluster.color);
//             overlay.setAttribute("fill-opacity", 0.53);  // very subtle
//             overlay.setAttribute("opacity", 0.5);
//             overlay.setAttribute("class", "cluster-overlay");
//             svgContainer.appendChild(overlay);

//             const avg = computeAverageFeatures(region.features);
        

//             const draw = SVG().addTo('#svgCanvas').size(canvasWidth, canvasHeight);

//             drawUnifiedRegionPathFromVisual(svgContainer, region, maxDuration, canvasWidth, cluster.color, 1, draw);
//             // Create roughness-based sketch layer clipped to region shape
//             // 🔒 Safe ID generation
//             const regionKey = region.start_time.toFixed(2).replace('.', '-');
//             // In your main code:
//             const clipId = `region-clip-${region.start_time.toFixed(2)}`;
//             const clipPath = createClipPathFromRegion(draw, region, maxDuration, canvasWidth, canvasHeight, clipId);

//             if (clipPath) {
//                 drawRoughnessSketch(draw, region, region.features, canvasWidth, maxDuration, canvasHeight, clipPath);
//             }
//             // 1. Draw background soft contour region (based on cluster)
//             drawClusterBlob(draw, region, cluster.color, canvasWidth, canvasHeight, maxDuration);

//             // 2. Draw inner expressive shapes for each subregion (gesture) region.features && drawSubregionGestures(draw, region.features, region, canvasWidth, maxDuration, canvasHeight);

//         });
//     });
// }

// function drawClusterOverlays(clusters,features,svgContainer,canvasWidth,canvasHeight,maxDuration)
// {
//     assignFeaturesToRegions(clusters,features);
//     computeVisualDataForRegions(clusters,maxDuration,canvasWidth);

//     // console.log(features);

//     const draw = SVG().addTo('#svgCanvas').size('100%',canvasHeight);

//     clusters.forEach(cluster => {
//         // === Cluster-level background blob ===
//         const clusterStart = cluster.start_time;
//         const clusterEnd = cluster.end_time;
//         const clusterX = map(clusterStart,0,maxDuration,0,canvasWidth);
//         const clusterWidth = map(clusterEnd,0,maxDuration,0,canvasWidth) - clusterX;

//         const clusterFeatures = cluster.regions.flatMap(r => r.features || []);
//         const avgY = clusterFeatures.length
//             ? clusterFeatures.reduce((sum,f) => sum + (f.visual?.yAxis ?? canvasHeight / 2),0) / clusterFeatures.length
//             : canvasHeight / 2;
//         const avgBandwidth = clusterFeatures.length
//             ? clusterFeatures.reduce((sum,f) => sum + (f.visual?.lineLength ?? 50),0) / clusterFeatures.length
//             : 50;
//         // === Check average loudness ===
//         const avgLoudness = clusterFeatures.length
//         ? clusterFeatures.reduce((sum,f) => sum + (f.loudness ?? 0),0) / clusterFeatures.length
//         : 0;


//         if (avgLoudness > 0.51) { // adjust this threshold to taste
//             drawVagueClusterBlob({
//                 draw,
//                 features: clusterFeatures,
//                 color: cluster.color,
//                 opacity: 0.35,
//                 maxDuration,
//                 canvasWidth,
//                 canvasHeight
//             });
//         } else {
//             console.log("Skipping blob due to silence.");
//         }
            
//         // drawLigetiEnvelopeBlob({
//         //     draw,
//         //     features: clusterFeatures,
//         //     color: cluster.color,
//         //     opacity: 0.35,
//         //     maxDuration,
//         //     canvasWidth,
//         //     canvasHeight
//         // });
            
//         // drawLigetiClusterBlob({
//         //     draw,
//         //     features: clusterFeatures,
//         //     xAxis: clusterX,
//         //     width: clusterWidth,
//         //     color: cluster.color,
//         //     opacity: 0.35,
//         //     canvasHeight
//         // });
            
            
//         const LOUDNESS_THRESHOLD = 0; // same threshold logic as blobs

//         // === Region-level overlays and features ===
//         cluster.regions.forEach(region => {
//             const startX = map(region.start_time,0,maxDuration,0,canvasWidth);
//             const endX = map(region.end_time,0,maxDuration,0,canvasWidth);
//             const width = endX - startX;

//             const overlay = document.createElementNS("http://www.w3.org/2000/svg","rect");
//             overlay.setAttribute("xAxis",startX);
//             overlay.setAttribute("y",0);
//             overlay.setAttribute("width",width);
//             overlay.setAttribute("height",canvasHeight);
//             overlay.setAttribute("fill",cluster.color);
//             overlay.setAttribute("fill-opacity",0);
//             overlay.setAttribute("opacity",0.5);
//             overlay.setAttribute("class","cluster-overlay");
//             svgContainer.appendChild(overlay);

//             const avg = computeAverageFeatures(region.features);
//             drawUnifiedRegionPathFromVisual(svgContainer,region,maxDuration,canvasWidth,cluster.color,1,draw);

//             const clipId = `region-clip-${region.start_time.toFixed(2)}`;
//             const clipPath = createClipPathFromRegion(draw,region,maxDuration,canvasWidth,canvasHeight,clipId);

//             if (clipPath) {
//                 drawRoughnessSketch(draw,region,region.features,canvasWidth,maxDuration,canvasHeight,clipPath);
//             }

//             // ✅ Only draw gesture shapes if region is perceptually strong enough
//             if (region.features && region.features.length > 0) {
//                 const avgLoudness = region.features.reduce((sum,f) => sum + (f.loudness ?? 0),0) / region.features.length;

//                 if (avgLoudness > LOUDNESS_THRESHOLD) {
//                 drawSubregionGestures(draw,region.features,region,canvasWidth,maxDuration,canvasHeight);
//                 } else {
//                 console.log(`Skipping subregion gestures: too quiet, avg loudness = ${avgLoudness}`);
//                 }
//             }
//         });
//     });
// }

// function drawVagueClusterBlob({ draw, features, color, opacity, maxDuration, canvasWidth, canvasHeight }) {
//     if (!features || features.length < 2) return;

//     const padding = 40;
//     const stride = Math.max(1, Math.floor(features.length / 20)); // Fewer points

//     const topPoints = [];
//     const bottomPoints = [];

//     for (let i = 0; i < features.length; i += stride) {
//         const f = features[i];
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual?.yAxis ?? canvasHeight / 2;
//         const h = f.visual?.lineLength ?? 30;

//         topPoints.push([xAxis, y - h / 2 - padding]);
//         bottomPoints.unshift([xAxis, y + h / 2 + padding]);
//     }

//     const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]); // Closed shape

//     // Create vague reference box to morph from
//     const xValues = blobPoints.map(p => p[0]);
//     const yValues = blobPoints.map(p => p[1]);
//     const minX = Math.min(...xValues), maxX = Math.max(...xValues);
//     const minY = Math.min(...yValues), maxY = Math.max(...yValues);

//     const roundedBox = [
//         [minX, minY],
//         [maxX, minY],
//         [maxX, maxY],
//         [minX, maxY],
//         [minX, minY]
//     ];

//     const vaguePath = flubber.interpolate(
//         flubber.toPathString(roundedBox),
//         flubber.toPathString(blobPoints),
//         { maxSegmentLength: 10 }
//     )(0.8); // 80% interpolated toward actual blob

//     draw.path(vaguePath)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'filter': 'url(#blur)' // Optional
//         });
// }

// function drawVagueClusterBlob({ draw, features, color, opacity, maxDuration, canvasWidth, canvasHeight }) {
//     if (!features || features.length < 2) return;

//     const padding = 40;
//     const stride = Math.max(1, Math.floor(features.length / 20)); // Fewer points

//     const topPoints = [];
//     const bottomPoints = [];

//     for (let i = 0; i < features.length; i += stride) {
//         const f = features[i];
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual?.yAxis ?? canvasHeight / 2;
//         const h = f.visual?.lineLength ?? 30;

//         topPoints.push([xAxis, y - h / 2 - padding]);
//         bottomPoints.unshift([xAxis, y + h / 2 + padding]);
//     }

//     const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]); // Closed shape

//     // Create vague reference box to morph from
//     const xValues = blobPoints.map(p => p[0]);
//     const yValues = blobPoints.map(p => p[1]);
//     const minX = Math.min(...xValues), maxX = Math.max(...xValues);
//     const minY = Math.min(...yValues), maxY = Math.max(...yValues);

//     const roundedBox = [
//         [minX, minY],
//         [maxX, minY],
//         [maxX, maxY],
//         [minX, maxY],
//         [minX, minY]
//     ];

//     const vaguePath = flubber.interpolate(
//         flubber.toPathString(roundedBox),
//         flubber.toPathString(blobPoints),
//         { maxSegmentLength: 10 }
//     )(0.8); // 80% interpolated toward actual blob

//     draw.path(vaguePath)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'filter': 'url(#blur)' // Optional
//         });
// }

// function drawVagueClusterBlob({
//     draw,
//     features,
//     color,
//     opacity,
//     maxDuration,
//     canvasWidth,
//     canvasHeight
// }) {
//     if (!features || features.length < 2) return;

//     const padding = 40;
//     const stride = Math.max(1, Math.floor(features.length / 20));
//     const threshold = 0.01; // loudness threshold
//     const minLineLength = 20;

//     const topPoints = [];
//     const bottomPoints = [];

//     let lastY = canvasHeight / 2;
//     let lastH = 50;

//     for (let i = 0; i < features.length; i += stride) {
//         const f = features[i];
//         const loudness = f.loudness ?? 0;

//         let y, h;

//         if (loudness > threshold) {
//             y = f.visual?.yAxis ?? lastY;
//             h = f.visual?.lineLength ?? lastH;
//             lastY = y;
//             lastH = h;
//         } else {
//             // If silent, hold previous or fallback
//             y = lastY;
//             h = Math.max(lastH * 0.8, minLineLength); // shrink a bit if needed
//         }

//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);

//         topPoints.push([xAxis, y - h / 2 - padding]);
//         bottomPoints.unshift([xAxis, y + h / 2 + padding]);
//     }

//     const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]);

//     // OPTIONAL: use d3-shape for smooth closed path
//     const smoothPath = d3.line()
//         .xAxis(d => d[0])
//         .y(d => d[1])
//         .curve(d3.curveCatmullRomClosed.alpha(0.5)); // smooth closed curve

//     const pathData = smoothPath(blobPoints);

//     draw.path(pathData)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'filter': 'url(#blur)'
//         });
// }

// function drawVagueClusterBlob({
//     draw,
//     features,
//     color,
//     opacity,
//     maxDuration,
//     canvasWidth,
//     canvasHeight
// }) {
//     if (!features || features.length < 2) return;

//     const padding = 40;
//     const stride = Math.max(1, Math.floor(features.length / 20));
//     const threshold = 10; // silence threshold
//     const minLineLength = 20;

//     const topPoints = [];
//     const bottomPoints = [];

//     let lastY = canvasHeight / 2;
//     let lastH = 50;

//     for (let i = 0; i < features.length; i += stride) {
//         const f = features[i];
//         const loudness = f.loudness ?? 0;

//         let y, h;

//         if (loudness > threshold) {
//             y = f.visual?.yAxis ?? lastY;
//             h = f.visual?.lineLength ?? lastH;
//             lastY = y;
//             lastH = h;
//         } else {
//             // If silent, hold previous good value, but soften length a bit
//             y = lastY;
//             h = Math.max(lastH * 0.8, minLineLength);
//         }

//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);

//         topPoints.push([xAxis, y - h / 2 - padding]);
//         bottomPoints.unshift([xAxis, y + h / 2 + padding]);
//     }

//     // Close the blob
//     const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]);

//     // Reference box for morph
//     const xValues = blobPoints.map(p => p[0]);
//     const yValues = blobPoints.map(p => p[1]);
//     const minX = Math.min(...xValues), maxX = Math.max(...xValues);
//     const minY = Math.min(...yValues), maxY = Math.max(...yValues);

//     const roundedBox = [
//         [minX, minY],
//         [maxX, minY],
//         [maxX, maxY],
//         [minX, maxY],
//         [minX, minY]
//     ];

//     // Interpolate the vague blob
//     const vaguePath = flubber.interpolate(
//         flubber.toPathString(roundedBox),
//         flubber.toPathString(blobPoints),
//         { maxSegmentLength: 10 }
//     )(0.8);

//     draw.path(vaguePath)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'filter': 'url(#blur)'
//         });
// }


  
//   function drawVagueClusterBlob({
//     draw,
//     features,
//     color,
//     opacity,
//     maxDuration,
//     canvasWidth,
//     canvasHeight
//   }) {
//     if (!features || features.length < 2) return;
  
//     const padding = 40;
//     const stride = Math.max(1, Math.floor(features.length / 20));
//     const threshold = 10; // loudness silence threshold
//     const minLineLength = 20;
  
//     const topPoints = [];
//     const bottomPoints = [];
  
//     // === Compute cluster Y stats for clamping ===
//     const ys = features.map(f => f.visual?.yAxis ?? canvasHeight / 2);
//     const ysSorted = [...ys].sort((a, b) => a - b);
//     const lowerPercentile = ysSorted[Math.floor(ysSorted.length * 0.1)];
//     const upperPercentile = ysSorted[Math.floor(ysSorted.length * 0.9)];
//     const medianY = ysSorted[Math.floor(ysSorted.length / 2)];

//     const clampMin = lowerPercentile;
//     const clampMax = upperPercentile;

//     let lastY = medianY;
//     let lastH = 50;
  
//     for (let i = 0; i < features.length; i += stride) {
//       const f = features[i];
//       const loudness = f.loudness ?? 0;
  
//       let y, h;
  
//       if (loudness > threshold) {
//         y = f.visual?.yAxis ?? lastY;
//         h = f.visual?.lineLength ?? lastH;
//         lastY = y;
//         lastH = h;
//       } else {
//         y = lastY;
//         h = Math.max(lastH * 0.8, minLineLength);
//       }
  
//       // ✅ Clamp Y-axis to stay inside reasonable blob band
//       y = clamp(y, clampMin, clampMax);
  
//       const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
  
//       topPoints.push([xAxis, y - h / 2 - padding]);
//       bottomPoints.unshift([xAxis, y + h / 2 + padding]);
//     }
  
//     const blobPoints = topPoints.concat(bottomPoints, [topPoints[0]]);
  
//     const xValues = blobPoints.map(p => p[0]);
//     const yValues = blobPoints.map(p => p[1]);
//     const minX = Math.min(...xValues),
//       maxX = Math.max(...xValues);
//     const minYShape = Math.min(...yValues),
//       maxYShape = Math.max(...yValues);
  
//     const roundedBox = [
//       [minX, minYShape],
//       [maxX, minYShape],
//       [maxX, maxYShape],
//       [minX, maxYShape],
//       [minX, minYShape]
//     ];
  
//     const vaguePath = flubber.interpolate(
//       flubber.toPathString(roundedBox),
//       flubber.toPathString(blobPoints),
//       { maxSegmentLength: 10 }
//     )(0.8);
  
//     draw.path(vaguePath)
//       .fill(color)
//       .stroke({ width: 0 })
//       .opacity(opacity)
//       .attr({
//         'fill-opacity': opacity,
//         'vector-effect': 'non-scaling-stroke',
//         'stroke-linejoin': 'round',
//         'stroke-linecap': 'round',
//         'filter': 'url(#blur)'
//       });
//   }
  



// function drawLigetiEnvelopeBlob({ draw, features, color, opacity, maxDuration, canvasWidth, canvasHeight }) {
//     if (!features || features.length === 0) return;

//     const padding = 20;
//     const topPoints = [];
//     const bottomPoints = [];

//     features.forEach(f => {
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual?.yAxis ?? canvasHeight / 2;
//         const h = f.visual?.lineLength ?? 30;

//         topPoints.push([xAxis, y - h / 2 - padding]);
//         bottomPoints.unshift([xAxis, y + h / 2 + padding]); // reverse order
//     });

//     const pathPoints = topPoints.concat(bottomPoints, [topPoints[0]]); // close loop

//     const pathStr = flubber.toPathString(pathPoints);

//     draw.path(pathStr)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'filter': 'url(#blur)' // optional
//         });
// }


// function drawLigetiClusterBlob({ draw, features, xAxis, width, color, opacity, canvasHeight }) {
//     if (!features || features.length === 0) return;

//     const padding = 100;
//     const numPoints = 20;

//     // Compute min/max y based on feature positions
//     const yValues = features.map(f => f.visual?.yAxis ?? canvasHeight / 2);
//     const minY = Math.min(...yValues) - padding;
//     const maxY = Math.max(...yValues) + padding;

//     const centerX = xAxis + width / 2;
//     const points = [];

//     for (let i = 0; i < numPoints; i++) {
//         const angle = (Math.PI * 2 * i) / numPoints;
//         const radiusX = width / 2 + (Math.random() - 0.5) * width * 0.2;
//         const radiusY = (maxY - minY) / 2 + (Math.random() - 0.5) * (maxY - minY) * 0.2;
//         const px = centerX + Math.cos(angle) * radiusX;
//         const py = (minY + maxY) / 2 + Math.sin(angle) * radiusY;

//         points.push([px, py]);
//     }

//     // Close the shape
//     points.push(points[0]);

//     const pathStr = flubber.toPathString(points);

//     draw.path(pathStr)
//         .fill(color)
//         .stroke({ width: 0 })
//         .opacity(opacity)
//         .attr({
//             'fill-opacity': opacity,
//             'vector-effect': 'non-scaling-stroke',
//             'stroke-linejoin': 'round',
//             'stroke-linecap': 'round',
//             'stroke-width': 0,
//             'filter': 'url(#blur)'
//         });
// }






// function drawClusterBlob({ draw, xAxis, y, width, height, color, opacity = 0.35 }) {
//     draw.ellipse(width, height)
//         .center(xAxis + width / 2, y)
//         .fill(color)
//         .opacity(opacity);
// }


// Draw soft background blob for a region
// function drawClusterBlob(draw, region, color, canvasWidth, canvasHeight, maxDuration) {
//     const xAxis = map(region.start_time, 0, maxDuration, 0, canvasWidth);
//     const width = map(region.end_time, 0, maxDuration, 0, canvasWidth) - xAxis;

//     const spectralCentroid = region.avg_features?.spectral_centroid ?? 5000;
//     const bandwidth = region.avg_features?.spectral_bandwidth ?? 3000;
//     const maxFreq = 20000;

//     const y = map(spectralCentroid, 0, maxFreq, canvasHeight, 0);
//     const height = map(bandwidth, 0, maxFreq / 2, 0, canvasHeight * 0.8);

//     // Use soft rounded blob shape approximation
//     const blob = draw.ellipse(width, height)
//         .center(xAxis + width / 2, y)
//         .fill(color)
//         .opacity(0.12);  // soft background
// }

// Draw expressive gesture shapes inside each region
// function drawSubregionGestures(draw, features, region, canvasWidth, maxDuration, canvasHeight) {
//     if (!features || features.length === 0) return;

//     // Sort features by loudness (descending)
//     const sortedByLoudness = [...features].sort((a, b) => (b.loudness ?? 0) - (a.loudness ?? 0));
//     const top3 = sortedByLoudness.slice(0, 3);
//     const next2 = sortedByLoudness.slice(3, 5); // Optional

//     // === TRIANGLES for top 3 ===
//     top3.forEach(f => {
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual?.yAxis ?? canvasHeight / 2;
//         const rawLineWidth = f.visual?.lineWidth ?? 1;
//         const size = map(rawLineWidth, 1, 5, 5, 20);
//         const angle = f.visual?.angle ?? 0;
//         const color = f.visual?.colorHue ?? "#333";

//         // Define an upward-pointing triangle centered at (xAxis, y)
//         const halfSize = size / 2;
//         const points = [
//             [xAxis, y - halfSize],
//             [xAxis - halfSize * Math.sin(Math.PI / 3), y + halfSize / 2],
//             [xAxis + halfSize * Math.sin(Math.PI / 3), y + halfSize / 2]
//         ];

//         draw.polygon(points.map(p => p.join(',')).join(' '))
//             .fill(color)
//             .stroke({ color, width: 0.4, opacity: 0.6 })
//             .rotate((angle * 180) / Math.PI, xAxis, y);
//     });

//     // === CIRCLES for next 2 (optional) ===
//     next2.forEach(f => {
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual?.yAxis ?? canvasHeight / 2;
//         const rawLineWidthForRadius = f.visual?.lineWidth ?? 1;
//         const radius = map(rawLineWidthForRadius, 1, 5, 1, 5);

//         const rotation = (f.visual?.angle ?? 0) * (180 / Math.PI);
//         const color = f.visual?.colorHue ?? "#333";

//         draw.circle(radius * 2)
//             .center(xAxis, y)
//             .fill(color)
//             .stroke({ color, width: 0.4, opacity: 0.5 })
//             .rotate(rotation, xAxis, y);
//     });
// }




// ===============================
// ✨ Compute average features from a region
// ===============================
// function computeAverageFeatures(features) {
//     const sum = {
//         spectral_centroid: 0,
//         spectral_flatness: 0,
//         spectral_bandwidth: 0,         // ✅ added
//         zerocrossingrate: 0,
//         brightness: 0,
//         sharpness: 0,
//         loudness: 0,
//         yin_periodicity: 0,
//         mir_mps_roughness: 0,
//         crepe_confidence: 0,
//         crepe_f0: 0,
//     };

//     features.forEach(f => {
//         sum.spectral_centroid += f.spectral_centroid || 0;
//         sum.spectral_flatness += f.spectral_flatness || 0;
//         sum.spectral_bandwidth += f.spectral_bandwidth || 0;  // ✅ added
//         sum.zerocrossingrate += f.zerocrossingrate || 0;
//         sum.brightness += f.brightness || 0;
//         sum.sharpness += f.sharpness || 0;
//         sum.loudness += f.loudness || 0;
//         sum.yin_periodicity += f.yin_periodicity || 0;
//         sum.mir_mps_roughness += f.mir_mps_roughness || 0;
//         sum.crepe_confidence += f.crepe_confidence || 0;
//         sum.crepe_f0 += f.crepe_f0 || 0;
//     });

//     const count = features.length || 1;
//     return {
//         spectral_centroid: sum.spectral_centroid / count,
//         spectral_flatness: sum.spectral_flatness / count,
//         spectral_bandwidth: sum.spectral_bandwidth / count,   // ✅ added
//         zerocrossingrate: sum.zerocrossingrate / count,
//         brightness: sum.brightness / count,
//         sharpness: sum.sharpness / count,
//         loudness: sum.loudness / count,
//         yin_periodicity: sum.yin_periodicity / count,
//         mir_mps_roughness: sum.mir_mps_roughness / count,
//         crepe_confidence: sum.crepe_confidence / count,
//         crepe_f0: sum.crepe_f0 / count,
//     };
// }

// function drawRoughnessSketch(draw, region, features, canvasWidth, maxDuration, canvasHeight, clipElement) {
//     const group = draw.group().id(`sketch-${region.start_time.toFixed(2)}`).clipWith(clipElement);

//     const maxSampleFrames = 150;
//     const step = Math.ceil(features.length / maxSampleFrames);
//     const MAX_TOTAL_LINES = 55000;

//     const roughnessSamples = [];

//     // Step 1: Collect graininess per frame using dashArray
//     for (let i = 0; i < features.length; i += step) {
//         const f = features[i];
//         const dashArray = f.visual?.dashArray ?? 0;
//         const lineWidth = f.visual?.lineWidth ?? 0.3;


//         // Map: low dashArray → dense texture, high dashArray → sparse
//         const graininess = map(dashArray, 0, 10, 4000, 10); // up to 120 lines at smoothness

//         roughnessSamples.push({ f, lineWidth, graininess });
//     }

//     // Step 2: Scaling factor if we exceed total line limit
//     const totalLines = roughnessSamples.reduce((sum, d) => sum + Math.floor(d.graininess), 0);
//     const scaleFactor = totalLines > MAX_TOTAL_LINES ? MAX_TOTAL_LINES / totalLines : 1;

//     let globalPathData = '';
//     let opacitySum = 0;
//     let lineCount = 0;

//     // Step 3: Generate path lines for sketch texture
//     roughnessSamples.forEach(({ f, lineWidth, graininess }) => {
//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const numLines = Math.floor(graininess * scaleFactor);
//         const lineOpacity = map(lineWidth, 0, 5, 0.05, 1);


//         for (let j = 0; j < numLines; j++) {
//             const offset = (Math.random() - 0.5) * 16;
//             const y1 = 100 + Math.random() * (canvasHeight - 200);
//             const y2 = y1 + Math.random() * 40 - 20;
//             globalPathData += `M ${xAxis + offset} ${y1} L ${xAxis + offset} ${y2} `;
//             opacitySum += lineOpacity;
//             lineCount++;
//         }
//     });

//     if (lineCount > 0) {
//         const avgOpacity = Math.min(1, opacitySum / lineCount);
//         group.path(globalPathData.trim())
//              .stroke({ color: '#333', width: avgOpacity, opacity: avgOpacity });
//     }
// }



// function createClipPathFromRegion(draw, region, maxDuration, canvasWidth, canvasHeight, clipId) {
//     const regionFrames = region.features;
//     if (!regionFrames || regionFrames.length < 2) return null;

//     const topPoints = [], bottomPoints = [];

//     regionFrames.forEach(f => {
//         if (!f.visual) return;

//         const xAxis = map(f.timestamp, 0, maxDuration, 0, canvasWidth);
//         const y = f.visual.yAxis;
//         const h = f.visual.lineLength ?? 60;
//         const mod = f.visual.mod ?? 0;

//         topPoints.push([xAxis, y - h / 2 - mod]);
//         bottomPoints.unshift([xAxis, y + h / 2 + mod]);
//     });

//     const allPoints = topPoints.concat(bottomPoints);
//     const pathData = catmullRomToPath(allPoints);

//     const clipPath = draw.clip().id(clipId);
//     clipPath.path(pathData).fill('#000'); // fill color is irrelevant for clip

//     return clipPath; // ✅ RETURN this
// }

//! -------------------------------------------------------------------
//! Other stuff
//! -------------------------------------------------------------------

// let clickedPosition = 0; // Global variable to store the clicked position
// const svgElement = document.getElementById("svgCanvas");
// let selectedElements = [];

function initializeSvgDragSelect() {
return svgDragSelect({
  svg: svgElement,
  referenceElement: null,
  selector: "enclosure",

  onSelectionStart({ svg, pointerEvent, cancel }) {
    if (pointerEvent.button !== 0) {
      cancel();
      return;
    }

    selectedElements = svg.querySelectorAll('[data-selected]');
    for (let i = 0; i < selectedElements.length; i++) {
      selectedElements[i].removeAttribute('data-selected');
    }
  },

  onSelectionChange({ svg, pointerEvent, selectedElements: newSelectedElements, previousSelectedElements, newlySelectedElements, newlyDeselectedElements }) {
    newlyDeselectedElements.forEach(element => element.removeAttribute('data-selected'));
    newlySelectedElements.forEach(element => element.setAttribute('data-selected', ''));
  },

  onSelectionEnd({ svg, pointerEvent, selectedElements: newSelectedElements }) {
    selectedElements = newSelectedElements;
    if (selectedElements.length > 0) {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute('id', 'selected-group');
      
      selectedElements.forEach(el => {
        group.appendChild(el.cloneNode(true));
        el.remove();
      });

      svg.appendChild(group);

      gsap.registerPlugin(Draggable);
      Draggable.create(group, {
        type: "xAxis,y",
        onPress: function() {
          dragSelectInstance.cancel();
        },
        onRelease: function() {
          dragSelectInstance = initializeSvgDragSelect();
        },
      });
    }
  },
});
}
// let dragSelectInstance = initializeSvgDragSelect();

function createAdaptiveTexturePattern(draw, roughness, regionId) {
    const patternId = `texture-${regionId}`;
    const patternSize = 4 + (1 - roughness) * 12;
    const stripeWidth = 0.4 + roughness * 1.2;
    const dotRadius = 0.5 + roughness * 1.3;

    const pattern = draw.pattern(patternSize, patternSize, function (add) {
        if (roughness < 0.3) {
            // 🎯 Smooth → dense dotted pattern
            for (let y = 0; y < patternSize; y += dotRadius * 2 + 1) {
                for (let xAxis = 0; xAxis < patternSize; xAxis += dotRadius * 2 + 1) {
                    add.circle(dotRadius * 2)
                        .center(xAxis, y)
                        .fill('#333');
                }
            }
        } else if (roughness < 0.65) {
            // 🎯 Medium roughness → diagonal stripe pattern
            add.line(0, 0, patternSize, patternSize)
                .stroke({ color: '#444', width: stripeWidth });

            add.line(0, patternSize, patternSize, 0)
                .stroke({ color: '#444', width: stripeWidth });
        } else {
            // 🎯 High roughness → hybrid (dots + stripes)
            for (let y = 0; y < patternSize; y += dotRadius * 2 + 2) {
                for (let xAxis = 0; xAxis < patternSize; xAxis += dotRadius * 2 + 2) {
                    add.circle(dotRadius * 2)
                        .center(xAxis, y)
                        .fill('#222');
                }
            }
            add.line(0, 0, patternSize, patternSize)
                .stroke({ color: '#555', width: stripeWidth * 0.8, opacity: 0.6 });

            add.line(0, patternSize, patternSize, 0)
                .stroke({ color: '#555', width: stripeWidth * 0.8, opacity: 0.6 });
        }
    });

    pattern.id(patternId);
    return pattern;
}



function createHatchPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));  // clamp between 0 and 1
    const spacing = Math.max(3, 5 + (1 - roughness) * 20);      // enforce min spacing

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "0");
    line.setAttribute("y1", "0");
    line.setAttribute("x2", spacing.toString());
    line.setAttribute("y2", spacing.toString());
    line.setAttribute("stroke", "white");
    line.setAttribute("stroke-width", "0.5");

    pattern.appendChild(line);
    defs.appendChild(pattern);
}

function createDotPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));
    const spacing = Math.max(4, 10 - roughness * 8);

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", spacing / 2);
    circle.setAttribute("cy", spacing / 2);
    circle.setAttribute("r", spacing * 0.1);
    circle.setAttribute("fill", "white");

    pattern.appendChild(circle);
    defs.appendChild(pattern);
}

function createCrossHatchPattern(defs, patternId, roughnessRaw) {
    const roughness = Math.max(0, Math.min(1, roughnessRaw));
    const spacing = Math.max(4, 8 - roughness * 6); // tighter spacing for more roughness

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);

    const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line1.setAttribute("x1", "0");
    line1.setAttribute("y1", "0");
    line1.setAttribute("x2", spacing.toString());
    line1.setAttribute("y2", spacing.toString());
    line1.setAttribute("stroke", "white");
    line1.setAttribute("stroke-width", "1");

    const line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line2.setAttribute("x1", spacing.toString());
    line2.setAttribute("y1", "0");
    line2.setAttribute("x2", "0");
    line2.setAttribute("y2", spacing.toString());
    line2.setAttribute("stroke", "white");
    line2.setAttribute("stroke-width", "1");

    pattern.appendChild(line1);
    pattern.appendChild(line2);
    defs.appendChild(pattern);
}

function createCirclePattern(draw, regionId, density = 5, radius = 2, color = '#555') {
    const patternSize = 10; // smaller = denser pattern
    const pattern = draw.pattern(patternSize, patternSize, add => {
        for (let xAxis = 0; xAxis < patternSize; xAxis += density) {
            for (let y = 0; y < patternSize; y += density) {
                add.circle(radius).move(xAxis, y).fill(color);
            }
        }
    }).id(`circle-pattern-${regionId}`);
    return pattern;
}

function average(values) {
    if (!values || values.length === 0) return 0;
    const valid = values.filter(v => typeof v === 'number' && isFinite(v));
    return valid.length > 0 ? valid.reduce((sum, v) => sum + v, 0) / valid.length : 0;
}

function catmullRomToPath(points) {
    if (points.length < 2) return '';

    let d = `M ${points[0][0]},${points[0][1]}`;

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i - 1] || points[i];
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[i + 2] || p2;

        const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
        const cp1y = p1[1] + (p2[1] - p0[1]) / 6;

        const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
        const cp2y = p2[1] - (p3[1] - p1[1]) / 6;

        d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
    }

    return d + ' Z'; // Close the shape
}
