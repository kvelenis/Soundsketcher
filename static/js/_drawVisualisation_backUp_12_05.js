// Initialize an array to store dot positions
let pointsArray = [];
function drawVisualization(audioData, svgContainer, canvasWidth, maxDuration, fileIndex) {
    console.log(fileIndex);
    // Remove existing SVG if it exists
    console.log('audioData', audioData);
    // AUX FUNCTIONS
    function polarToCartesian(centerX, centerY, radius, angleInDegrees) {
        var angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
        return {
            x: centerX + (radius * Math.cos(angleInRadians)),
            y: centerY + (radius * Math.sin(angleInRadians))
        };
    }

    function polygon(centerX, centerY, points, radius) {
        var degreeIncrement = 360 / points;
        var d = "M"; // Start the SVG path string
        for (var i = 0; i < points; i++) {
            var angle = degreeIncrement * i;
            var point = polarToCartesian(centerX, centerY, radius, angle);
            d += point.x + "," + point.y + " ";
        }
        d += "Z"; // Close the path
        return d;
    }

    // Function to convert HEX to HSL
    function hexToHSL(hex) {
        // Convert hex to RGB
        let r = parseInt(hex.slice(1, 3), 16) / 255;
        let g = parseInt(hex.slice(3, 5), 16) / 255;
        let b = parseInt(hex.slice(5, 7), 16) / 255;

        // Find min and max RGB values
        let max = Math.max(r, g, b);
        let min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;

        if (max === min) {
            h = s = 0; // Achromatic
        } else {
            let d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }

        return Math.round(h * 360);
    }

    let existingSvg = document.querySelector('svg');
    // if (existingSvg) {
    //     existingSvg.remove();
    // }

    console.log('audioData', audioData)

    // Get the selected feature and characteristic
    const selectedFeature1 = featureSelect1.value;
    const selectedFeature2 = featureSelect2.value;
    const selectedFeature3 = featureSelect3.value;
    const selectedFeature4 = featureSelect4.value;
    const selectedFeature5 = featureSelect5.value;
    const selectedFeature6 = featureSelect6.value;
    const selectedFeature7 = featureSelect7.value;
    

    // Get the current state of the checkbox and the slider value
    const isThresholdEnabled = thresholdToggle.checked;
    const zeroCrossingThreshold = parseFloat(thresholdSlider.value);

    const isThresholdCircleEnabled = thresholdCircle.checked;
    const isPolygonEnabled = polygonToggle.checked;
    const isJoinPathsEnabled = joinDataPoints.checked;


  

    

    // Initialize maxAmplitude to the smallest possible value
    let maxSpectralCentroid = Number.MIN_SAFE_INTEGER;
    let maxSpectralFlux = Number.MIN_SAFE_INTEGER;
    let maxSpectralFlatness = Number.MIN_SAFE_INTEGER;
    let maxSpectralBandwidth = Number.MIN_SAFE_INTEGER;
    let maxAmplitude = Number.MIN_SAFE_INTEGER;
    let maxZerocrossingrate = Number.MIN_SAFE_INTEGER;
    let maxYinF0Librosa = Number.MIN_SAFE_INTEGER;
    let maxYinF0Aubio = Number.MIN_SAFE_INTEGER;
    let maxF0Crepe = Number.MIN_SAFE_INTEGER;
    let maxF0CrepeConfidence = Number.MIN_SAFE_INTEGER;
    let maxStandardDeviation = Number.MIN_SAFE_INTEGER;
    let maxBrightness = Number.MIN_SAFE_INTEGER;
    let maxSharpness = Number.MIN_SAFE_INTEGER;
    let maxLoudness = Number.MIN_SAFE_INTEGER;
    let maxPeriodicity = Number.MIN_SAFE_INTEGER;
    let maxF0Candidates = Number.MIN_SAFE_INTEGER;

    // MIR features
    let maxMirMpsRoughness = Number.MIN_SAFE_INTEGER;
    let maxMirRoughnessZwicker = Number.MIN_SAFE_INTEGER;
    let maxMirSharpnessZwicker = Number.MIN_SAFE_INTEGER;
    let maxMirRoughnessVassilakis = Number.MIN_SAFE_INTEGER;
    let maxMirRoughnessSethares = Number.MIN_SAFE_INTEGER;


    let maxSpectralCentroidMinusStandardDeviation = Number.MIN_SAFE_INTEGER;
    // Threshold for zero crossing rate
    //const zeroCrossingThreshold = 30;
    // Calculate maxAmplitude and maxZerocrossingrate
    console.log("audioData.features",audioData.features)
    
    for (let i = 0; i < audioData.features.length; i++) {
        const f = audioData.features[i];
    
        // Core features
        if (f.amplitude > maxAmplitude) maxAmplitude = f.amplitude;
        if (f.zerocrossingrate > maxZerocrossingrate) maxZerocrossingrate = f.zerocrossingrate;
        if (f.spectral_centroid > maxSpectralCentroid) maxSpectralCentroid = f.spectral_centroid;
        if (f.spectral_bandwidth > maxSpectralBandwidth) maxSpectralBandwidth = f.spectral_bandwidth;
        if (f.spectral_flatness > maxSpectralFlatness) maxSpectralFlatness = f.spectral_flatness;
        if (f.spectral_flux > maxSpectralFlux) maxSpectralFlux = f.spectral_flux;
    
        // Pitch-related
        if (f.yin_f0_librosa > maxYinF0Librosa) maxYinF0Librosa = f.yin_f0_librosa;
        if (f.yin_f0_aubio > maxYinF0Aubio) maxYinF0Aubio = f.yin_f0_aubio;
        if (f.crepe_f0 > maxF0Crepe) maxF0Crepe = f.crepe_f0;
        if (f.crepe_confidence > maxF0CrepeConfidence) maxF0CrepeConfidence = f.crepe_confidence;
        if (f.yin_periodicity > maxPeriodicity) maxPeriodicity = f.yin_periodicity;
    
        // Timbre-related
        if (f.standard_deviation > maxStandardDeviation) maxStandardDeviation = f.standard_deviation;
        const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
        if (centroidMinusSD > maxSpectralCentroidMinusStandardDeviation) {
            maxSpectralCentroidMinusStandardDeviation = centroidMinusSD;
        }
        if (f.brightness > maxBrightness) maxBrightness = f.brightness;
        if (f.sharpness > maxSharpness) maxSharpness = f.sharpness;
        if (f.loudness > maxLoudness) maxLoudness = f.loudness;
    
        // MIR features
        if (f.mir_mps_roughness > maxMirMpsRoughness) maxMirMpsRoughness = f.mir_mps_roughness;
        if (f.mir_roughness_zwicker > maxMirRoughnessZwicker) maxMirRoughnessZwicker = f.mir_roughness_zwicker;
        if (f.mir_sharpness_zwicker > maxMirSharpnessZwicker) maxMirSharpnessZwicker = f.mir_sharpness_zwicker;
        if (f.mir_roughness_vassilakis > maxMirRoughnessVassilakis) maxMirRoughnessVassilakis = f.mir_roughness_vassilakis;
        if (f.mir_roughness_sethares > maxMirRoughnessSethares) maxMirRoughnessSethares = f.mir_roughness_sethares;
    }

    // Initialize min values to the largest possible value
    let minSpectralCentroid = Number.MAX_SAFE_INTEGER;
    let minSpectralFlux = Number.MAX_SAFE_INTEGER;
    let minSpectralFlatness = Number.MAX_SAFE_INTEGER;
    let minSpectralBandwidth = Number.MAX_SAFE_INTEGER;
    let minAmplitude = Number.MAX_SAFE_INTEGER;
    let minZerocrossingrate = Number.MAX_SAFE_INTEGER;
    let minYinF0Librosa = Number.MAX_SAFE_INTEGER;
    let minYinF0Aubio = Number.MAX_SAFE_INTEGER;
    let minF0Crepe = Number.MAX_SAFE_INTEGER;
    let minF0CrepeConfidence = Number.MAX_SAFE_INTEGER;
    let minStandardDeviation = Number.MAX_SAFE_INTEGER;
    let minBrightness = Number.MAX_SAFE_INTEGER;
    let minSharpness = Number.MAX_SAFE_INTEGER;
    let minLoudness = Number.MAX_SAFE_INTEGER;
    let minPeriodicity = Number.MAX_SAFE_INTEGER;

    // MIR features
    let minMirMpsRoughness = Number.MAX_SAFE_INTEGER;
    let minMirRoughnessZwicker = Number.MAX_SAFE_INTEGER;
    let minMirSharpnessZwicker = Number.MAX_SAFE_INTEGER;
    let minMirRoughnessVassilakis = Number.MAX_SAFE_INTEGER;
    let minMirRoughnessSethares = Number.MAX_SAFE_INTEGER;

    let minSpectralCentroidMinusStandardDeviation = Number.MAX_SAFE_INTEGER;

    for (let i = 0; i < audioData.features.length; i++) {
        const f = audioData.features[i];
    
        // Skip silent frames
        if (f.loudness === 0) continue;
    
        // Core features
        if (f.amplitude < minAmplitude) minAmplitude = f.amplitude;
        if (f.zerocrossingrate < minZerocrossingrate) minZerocrossingrate = f.zerocrossingrate;
        if (f.spectral_centroid < minSpectralCentroid) minSpectralCentroid = f.spectral_centroid;
        if (f.spectral_bandwidth < minSpectralBandwidth) minSpectralBandwidth = f.spectral_bandwidth;
        if (f.spectral_flatness < minSpectralFlatness) minSpectralFlatness = f.spectral_flatness;
        if (f.spectral_flux < minSpectralFlux) minSpectralFlux = f.spectral_flux;
    
        // Pitch-related
        if (f.yin_f0_librosa < minYinF0Librosa) minYinF0Librosa = f.yin_f0_librosa;
        if (f.yin_f0_aubio < minYinF0Aubio) minYinF0Aubio = f.yin_f0_aubio;
        if (f.crepe_f0 < minF0Crepe) minF0Crepe = f.crepe_f0;
        if (f.crepe_confidence < minF0CrepeConfidence) minF0CrepeConfidence = f.crepe_confidence;
        if (f.yin_periodicity < minPeriodicity) minPeriodicity = f.yin_periodicity;
    
        // Timbre-related
        if (f.standard_deviation < minStandardDeviation) minStandardDeviation = f.standard_deviation;
        const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
        if (centroidMinusSD < minSpectralCentroidMinusStandardDeviation) {
            minSpectralCentroidMinusStandardDeviation = centroidMinusSD;
        }
        if (f.brightness < minBrightness) minBrightness = f.brightness;
        if (f.sharpness < minSharpness) minSharpness = f.sharpness;
        if (f.loudness < minLoudness) minLoudness = f.loudness;
    
        // MIR features
        if (f.mir_mps_roughness < minMirMpsRoughness) minMirMpsRoughness = f.mir_mps_roughness;
        if (f.mir_roughness_zwicker < minMirRoughnessZwicker) minMirRoughnessZwicker = f.mir_roughness_zwicker;
        if (f.mir_sharpness_zwicker < minMirSharpnessZwicker) minMirSharpnessZwicker = f.mir_sharpness_zwicker;
        if (f.mir_roughness_vassilakis < minMirRoughnessVassilakis) minMirRoughnessVassilakis = f.mir_roughness_vassilakis;
        if (f.mir_roughness_sethares < minMirRoughnessSethares) minMirRoughnessSethares = f.mir_roughness_sethares;
    }


    // Log the minimum values to verify correctness
    // console.log("Minimum Values:");
    // console.log("Min Amplitude:", minAmplitude);
    // console.log("Min Zerocrossingrate:", minZerocrossingrate);
    // console.log("Min Spectral Centroid:", minSpectralCentroid);
    // console.log("Min Spectral Bandwidth:", minSpectralBandwidth);
    // console.log("Min Spectral Flatness:", minSpectralFlatness);
    // console.log("Min Spectral Flux:", minSpectralFlux);
    // console.log("Min Yin F0 Librosa:", minYinF0Librosa);
    // console.log("Min Yin F0 Aubio:", minYinF0Aubio);
    // console.log("Min Crepe F0:", minF0Crepe);
    // console.log("Min Periodicity:", minPeriodicity);
    // console.log("Min Crepe Confidence:", minF0CrepeConfidence);
    // console.log("Min Standard Deviation:", minStandardDeviation);
    // console.log("Min Spectral Centroid - Std Dev:", minSpectralCentroidMinusStandardDeviation);
    // console.log("Min Brightness:", minBrightness);
    // console.log("Min Sharpness:", minSharpness);
    // console.log("Min Loudness:", minLoudness);



    for (let i = 0; i < audioData.features.length; i++) {
        const f = audioData.features[i];
    
        f.normalizedAmplitude = f.amplitude / maxAmplitude || 0;
        f.normalizedZerocrossingrate = f.zerocrossingrate / maxZerocrossingrate || 0;
        f.normalizedSpectralCentroid = f.spectral_centroid / maxSpectralCentroid || 0;
        f.normalizedSpectralFlux = f.spectral_flux / maxSpectralFlux || 0;
        f.normalizedYinF0Librosa = f.yin_f0_librosa / maxYinF0Librosa || 0;
        f.normalizedYinF0Aubio = f.yin_f0_aubio / maxYinF0Aubio || 0;
        f.normalizedF0Crepe = f.crepe_f0 / maxF0Crepe || 0;
        f.normalizedF0CrepeConfidence = f.crepe_confidence / maxF0CrepeConfidence || 0;
        f.normalizedStandardDeviation = f.standard_deviation / maxStandardDeviation || 0;
    
        const centroidMinusSD = f.spectral_centroid - f.standard_deviation / 2;
        f.normalizedSpectralCentroidMinusStandardDeviation =
            centroidMinusSD / maxSpectralCentroidMinusStandardDeviation || 0;
    
        f.normalizedBrightness = f.brightness / maxBrightness || 0;
        f.normalizedSharpness = f.sharpness / maxSharpness || 0;
        f.normalizedLoudness = f.loudness / maxLoudness || 0;
        f.normalizedPeriodicity = f.yin_periodicity / maxPeriodicity || 0;
    
        // MIR features
        f.normalizedMirMpsRoughness = f.mir_mps_roughness / maxMirMpsRoughness || 0;
        f.normalizedMirRoughnessZwicker = f.mir_roughness_zwicker / maxMirRoughnessZwicker || 0;
        f.normalizedMirSharpnessZwicker = f.mir_sharpness_zwicker / maxMirSharpnessZwicker || 0;
        f.normalizedMirRoughnessVassilakis = f.mir_roughness_vassilakis / maxMirRoughnessVassilakis || 0;
        f.normalizedMirRoughnessSethares = f.mir_roughness_sethares / maxMirRoughnessSethares || 0;
    }
    
    let maxLogSpectralCentroid = Math.log10(maxSpectralCentroid + 1);
    let minLogSpectralCentroid = Math.log10(1);

    // Add padding (in pixels) to the top and adjust the bottom range

    let previousDots = []; // To store the scattered dots of the previous data point

    // Loop through each time frame
    let pathData = []; // Array to store attributes for each path

    for (let i = 0; i < audioData.features.length; i++) {
        let feature = audioData.features[i];

        // Calculate x position based on timestamp
        // Calculate the x position based on the normalized timestamp
        let x = map(
            feature.timestamp, 
            0, 
            maxDuration, // Use the maximum duration for normalization
            0, 
            canvasWidth
        );

        function drawYAxisScale(canvasHeight, minValue, maxValue, isLogarithmic = false, topPadding = 50) {
            const svgCanvas = document.getElementById("svgCanvas");
            const scaleGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
            scaleGroup.setAttribute("class", "y-axis-scale");

            // Number of ticks
            const numTicks = 10;

            for (let i = 0; i <= numTicks; i++) {
                // Calculate linear value for this tick
                const linearValue = minValue + (i / numTicks) * (maxValue - minValue);

                // Calculate position based on logarithmic scale with padding
                const y = isLogarithmic
                    ? map(Math.log10(linearValue), Math.log10(minValue), Math.log10(maxValue), canvasHeight - topPadding, topPadding)
                    : map(linearValue, minValue, maxValue, canvasHeight - topPadding, topPadding);

                // Create tick line
                const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
                tick.setAttribute("x1", 0); // Adjust for position
                tick.setAttribute("x2", 10); // Length of the tick
                tick.setAttribute("y1", y);
                tick.setAttribute("y2", y);
                tick.setAttribute("stroke", "black");
                scaleGroup.appendChild(tick);

                // Create label showing linear value
                const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
                label.setAttribute("x", 15); // Position the label
                label.setAttribute("y", y + 3); // Center the label
                label.setAttribute("font-size", "10");
                label.textContent = linearValue.toFixed(2); // Show linear value
                scaleGroup.appendChild(label);
            }

            // Add the group to the canvas
            svgCanvas.appendChild(scaleGroup);
        }
        const isLogSelected = document.getElementById("log-linear").checked;

        // Example usage
        const canvasHeight = 800;
        const minValue = 100;
        const maxValue = 3000;
        let topPadding = 200; // Top padding in pixels
        drawYAxisScale(canvasHeight, minValue, maxValue, isLogSelected, topPadding);

        // Utility function for mapping
        function map(value, start1, stop1, start2, stop2) {
            return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
        }


        // Assuming slider values are already initialized and stored in variables
        function getDynamicRange(slider) {
            const values = slider.noUiSlider.get(); // Get the current slider range as [min, max]
            const minValue = parseFloat(values[0]);
            const maxValue = parseFloat(values[1]);
            return { minValue, maxValue };
        }

        function calculateDynamicRange(slider, isInverted, startKey = "startRange", endKey = "endRange") {
            const [minValue, maxValue] = slider.noUiSlider.get().map(parseFloat);

            const startRange = isInverted ? maxValue : minValue;
            const endRange = isInverted ? minValue : maxValue;

            return {
                [startKey]: startRange,
                [endKey]: endRange
            };
        }

        function mapToLinearScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {
            if (invert) {
                // Inverted mapping: High values at the bottom, low values at the top
                return map(
                    value,                              // Raw linear value
                    minValue,                           // Minimum linear value
                    maxValue,                           // Maximum linear value
                    topPadding,                         // Top of the canvas becomes bottom
                    canvasHeight - topPadding           // Bottom of the canvas becomes top
                );
            } else {
                // Standard mapping: Low values at the bottom, high values at the top
                return map(
                    value,                              // Raw linear value
                    minValue,                           // Minimum linear value
                    maxValue,                           // Maximum linear value
                    canvasHeight - topPadding,          // Bottom of the canvas
                    topPadding                          // Top of the canvas
                );
            }
        }

        function mapToLogScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {
            if (invert) {
                // Inverted mapping: High values at the bottom, low values at the top
                return map(
                    Math.log10(value),                   // Logarithmic transformation of value
                    Math.log10(minValue),               // Logarithmic min value
                    Math.log10(maxValue),               // Logarithmic max value
                    topPadding,                         // Top of the canvas becomes bottom
                    canvasHeight - topPadding           // Bottom of the canvas becomes top
                );
            } else {
                // Standard mapping: Low values at the bottom, high values at the top
                return map(
                    Math.log10(value),                   // Logarithmic transformation of value
                    Math.log10(minValue),               // Logarithmic min value
                    Math.log10(maxValue),               // Logarithmic max value
                    canvasHeight - topPadding,          // Bottom of the canvas
                    topPadding                          // Top of the canvas
                );
            }
        }


        // Example usage:
        const slider5 = document.getElementById("slider-5"); // Get the slider element
        const invertMappingCheckbox_y_axis = document.getElementById("invertMapping-5");
        const isInverted_y_axis = invertMappingCheckbox_y_axis && invertMappingCheckbox_y_axis.checked;

        // Get the dynamic range
        const { startRange_y_axis, endRange_y_axis } = calculateDynamicRange(slider5, isInverted_y_axis, "startRange_y_axis", "endRange_y_axis");
        
        function adjustPeriodicity(yin_periodicity) {
            threshold = 0.85;
            return yin_periodicity > threshold ? 1 : 0;

        }

        let y_axis = 0;
        switch (selectedFeature5) {
            case "amplitude":
                y_axis = map(feature.amplitude, minAmplitude, maxAmplitude, endRange_y_axis - topPadding, topPadding);
                break;
            case "zerocrossingrate":
                y_axis = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "spectral_centroid":
                if (isLogSelected) {
                    y_axis = mapToLogScale(feature.spectral_centroid+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(feature.spectral_centroid+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                // console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.crepe_f0 * feature.crepe_confidence) + ((feature.spectral_centroid * (1-feature.crepe_confidence))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.crepe_f0 * feature.crepe_confidence) + ((feature.spectral_centroid * (1-feature.crepe_confidence))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch-librosa":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                //console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.yin_f0_librosa * feature.crepe_confidence) + ((feature.spectral_centroid * (1-feature.crepe_confidence))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.yin_f0_librosa * feature.crepe_confidence) + ((feature.spectral_centroid * (1-feature.crepe_confidence))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch-librosa-periodicity":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                //console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.yin_f0_librosa * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.yin_f0_librosa * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch-crepe-periodicity":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                //console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.crepe_f0 * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.crepe_f0 * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch-f0-or-SC":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                //console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.crepe_f0 * adjustPeriodicity(feature.yin_periodicity)) + ((feature.spectral_centroid * (1-adjustPeriodicity(feature.yin_periodicity)))*0.30)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.crepe_f0 * adjustPeriodicity(feature.yin_periodicity)) + ((feature.spectral_centroid * (1-adjustPeriodicity(feature.yin_periodicity)))*0.30)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "perceived_pitch-f0_candidates-periodicity":
                // y_axis = map(((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/1.5) , 0, maxSpectralCentroid, startRange_y_axis, endRange_y_axis);
                //console.log((feature.crepe_f0 * feature.crepe_confidence) + (feature.spectral_centroid * (1-feature.crepe_confidence))/4);
                if (isLogSelected) {
                    y_axis = mapToLogScale(((feature.f0_candidates * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(((feature.f0_candidates * feature.yin_periodicity) + ((feature.spectral_centroid * (1-feature.yin_periodicity))*0.2)), minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                break;
            case "spectral_flux":
                y_axis = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "spectral_bandwidth":
                y_axis = map(feature.spectral_bandwidth, 0, maxSpectralBandwidth, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "spectral_flatness":
                y_axis = map(feature.spectral_flatness, 0, maxSpectralFlatness, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "yin_f0_librosa":
                if (isLogSelected) {
                    y_axis = mapToLogScale(feature.yin_f0_librosa+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(feature.yin_f0_librosa+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                // y_axis = map(feature.yin_f0_librosa, 0, maxYinF0Librosa, startRange_y_axis, endRange_y_axis);
                break;
            case "yin_f0_aubio":
                if (isLogSelected) {
                    y_axis = mapToLogScale(feature.yin_f0_aubio+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(feature.yin_f0_aubio+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                // y_axis = map(feature.yin_f0_aubio, 0, maxYinF0Aubio, startRange_y_axis, endRange_y_axis);
                break;
            case "crepe_f0":
                if (isLogSelected) {
                    y_axis = mapToLogScale(feature.crepe_f0+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                } else {
                    y_axis = mapToLinearScale(feature.crepe_f0+1, minValue, maxValue, canvasHeight, topPadding, isInverted_y_axis);
                }
                // y_axis = map(feature.crepe_f0, 0, maxF0Crepe, startRange_y_axis, endRange_y_axis);
                break;
            case "crepe_confidence":
                y_axis = map(feature.crepe_confidence, 0, maxF0CrepeConfidence, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "yin_periodicity":
                y_axis = map(feature.yin_periodicity, maxPeriodicity, 0, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "f0_candidates":
                y_axis = map(feature.f0_candidates, maxF0Candidates, 0, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "spectral_deviation":
                y_axis = map(feature.standard_deviation, 0, maxStandardDeviation, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "normalized_height":
                y_axis = map((feature.spectral_centroid - feature.standard_deviation/2)+600, 0, maxSpectralCentroidMinusStandardDeviation, startRange_y_axis, endRange_y_axis);
                break;
            case "brightness":
                y_axis = map(feature.brightness, 0, maxBrightness, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "sharpness":
                y_axis = map(feature.sharpness, 0, maxSharpness, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "loudness":
                y_axis = map(feature.loudness, 0, maxLoudness, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "none":
                y_axis = 400
                break;
            case "mir_mps_roughness":
                y_axis = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "mir_roughness_zwicker":
                y_axis = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "mir_sharpness_zwicker":
                y_axis = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "mir_roughness_vassilakis":
                y_axis = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            case "mir_roughness_sethares":
                y_axis = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
                break;
            default:
                y_axis = map(feature.spectral_centroid, 0, maxSpectralCentroid, startRange_y_axis + topPadding, endRange_y_axis - topPadding);
        }
        // // Calculate line length based on amplitude
        // let lineLength = map(feature.amplitude, 0, maxAmplitude, 0, 80);

        // Example usage:
        const slider1 = document.getElementById("slider-1"); // Get the slider element
        const invertMappingCheckbox_lineLength = document.getElementById("invertMapping-1");
        const isInverted_lineLength = invertMappingCheckbox_lineLength && invertMappingCheckbox_lineLength.checked;

        // Get the dynamic range
        const { startRange_lineLength, endRange_lineLength } = calculateDynamicRange(slider1, isInverted_lineLength, "startRange_lineLength", "endRange_lineLength");

        // console.log(startRange_lineLength, endRange_lineLength)



        // const scaleKnob = document.getElementById("scaleKnob");
        // const scaleValueDisplay = document.getElementById("scaleValue");
        // let scaleCompression = parseFloat(scaleKnob.value); // Initial scale value

        // // Update the scale value when the knob changes
        // scaleKnob.addEventListener("input", (event) => {
        //     scaleCompression = parseFloat(event.target.value);
        //     scaleValueDisplay.textContent = scaleCompression.toFixed(1); // Display the current scale value
        //     console.log(`Compression scale set to: ${scaleCompression}`);
        // });

        // Use the updated scaleCompression in your mapping
        function mapWithSoftClipping(value, minInput, maxInput, minOutput, maxOutput, scale = scaleCompression) {
            const normalized = (value - minInput) / (maxInput - minInput); // Normalize to [0, 1]
            const clipped = Math.atan(scale * (normalized - 0.5)) / Math.atan(scale) + 0.5; // Adjust scale for soft clipping
            return clipped * (maxOutput - minOutput) + minOutput;
        }

        // // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_lineLength = document.getElementById("invertMapping-1"); // Update ID as needed
        // const isInverted_lineLength = invertMappingCheckbox_lineLength && invertMappingCheckbox_lineLength.checked;

        // // Determine the range based on inversion state
        // let startRange_lineLength = isInverted_lineLength ? 0 : 80; // Swap the start range if inverted
        // let endRange_lineLength = isInverted_lineLength ? 80 : 0; // Swap the end range if inverted
        // Calculate line length based on selected feature1
        let lineLength = 0;
        // Define parameters
        const BaseLength = 2; // Minimum line length
        const MaxEffect = 80;  // Maximum effect of loudness on line length
        switch (selectedFeature1) {
            case "amplitude":
                lineLength = mapWithSoftClipping(feature.amplitude, minAmplitude, maxAmplitude, startRange_lineLength, endRange_lineLength, scaleCompression);
                break;
            case "zerocrossingrate":
                lineLength = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineLength, endRange_lineLength);
                break;
            case "spectral_centroid":
                lineLength = map(feature.spectral_centroid, 0, maxSpectralCentroid, startRange_lineLength, endRange_lineLength);
                break;
            case "spectral_flux":
                lineLength = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_lineLength, endRange_lineLength);
                break;
            case "spectral_bandwidth":
                lineLength = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_lineLength, endRange_lineLength);
                break;
            case "spectral_flatness":
                lineLength = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_lineLength, endRange_lineLength);
                break;
            case "yin_f0_librosa":
                lineLength = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_lineLength, endRange_lineLength);
                break;
            case "yin_f0_aubio":
                lineLength = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_lineLength, endRange_lineLength);
                break;
            case "crepe_f0":
                lineLength = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_lineLength, endRange_lineLength);
                break;
            case "crepe_confidence":
                lineLength = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_lineLength, endRange_lineLength);
                break;
            case "yin_periodicity":
                lineLength = map(feature.yin_periodicity, minPeriodicity, maxPeriodicity, startRange_lineLength, endRange_lineLength);
                break;
            case "spectral_deviation":
                lineLength = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_lineLength, endRange_lineLength);
                break;
            case "brightness":
                lineLength = map(feature.brightness, minBrightness, maxBrightness, startRange_lineLength, endRange_lineLength);
                break;
            case "sharpness":
                lineLength = map(feature.sharpness, minSharpness, maxSharpness, startRange_lineLength, endRange_lineLength);
                break;
            case "loudness":
                lineLength = map(feature.loudness, minLoudness, maxLoudness, startRange_lineLength, endRange_lineLength);
                break;
            case "loudness-zcr":
                
                const MaxZCR = maxZerocrossingrate; // Replace with the actual maximum ZCR

                // Calculate the contribution of loudness and ZCR
                let ZCRLoudnessEffect = map(feature.loudness, 0, maxLoudness, 0, endRange_lineLength);
                let ZCRInfluence = feature.zerocrossingrate / MaxZCR;

                // Combine loudness and ZCR to calculate the line length
                lineLength = startRange_lineLength + (ZCRLoudnessEffect * ZCRInfluence);
                break;
            case "loudness-periodicity":
                const MaxPeriodicity = maxPeriodicity; // Replace with the actual maximum ZCR

                // Calculate the contribution of loudness and ZCR
                let PeriodicityLoudnessEffect = map(feature.loudness, minLoudness, maxLoudness, 0, endRange_lineLength);
                let InvertedPeriodicity = MaxPeriodicity - feature.yin_periodicity;

                let PeriodicityInfluence = InvertedPeriodicity / MaxPeriodicity;

                // Combine loudness and ZCR to calculate the line length
                lineLength = startRange_lineLength + (PeriodicityLoudnessEffect * PeriodicityInfluence);
                break;
            case "loudness-pitchConf":
                // Define parameters
                
                const MaxPitchConfidence = maxF0CrepeConfidence; // Replace with the actual maximum ZCR
                
                // Calculate the contribution of loudness and ZCR
                let crepeLoudnessEffect = map(feature.loudness, minLoudness, maxLoudness, 0, endRange_lineLength);
                let InvertedPitchConfInfluence = 1 - (feature.crepe_confidence / MaxPitchConfidence); // Invert pitch confidence

                // Combine loudness and ZCR to calculate the line length
                lineLength = startRange_lineLength + (crepeLoudnessEffect * InvertedPitchConfInfluence);
                break;
            case "none":
                lineLength = 1
                break;

            case "mir_mps_roughness":
                lineLength = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_lineLength, endRange_lineLength);
                break;
            case "mir_roughness_zwicker":
                lineLength = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_lineLength, endRange_lineLength);
                break;
            case "mir_sharpness_zwicker":
                lineLength = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_lineLength, endRange_lineLength);
                break;
            case "mir_roughness_vassilakis":
                lineLength = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_lineLength, endRange_lineLength);
                break;
            case "mir_roughness_sethares":
                lineLength = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_lineLength, endRange_lineLength);
                break;
            default:
                lineLength = map(feature.amplitude, 0, maxAmplitude, startRange_lineLength, endRange_lineLength);
        }
        const slider2 = document.getElementById("slider-2"); // Get the slider element
        const invertMappingCheckbox_lineWidth = document.getElementById("invertMapping-2");
        const isInverted_lineWidth = invertMappingCheckbox_lineWidth && invertMappingCheckbox_lineWidth.checked;

        // Get the dynamic range
        const { startRange_lineWidth, endRange_lineWidth } = calculateDynamicRange(slider2, isInverted_lineWidth, "startRange_lineWidth", "endRange_lineWidth");
        
        // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_lineWidth = document.getElementById("invertMapping-2"); // Update ID as needed
        // const isInverted_lineWidth = invertMappingCheckbox_lineWidth && invertMappingCheckbox_lineWidth.checked;

        // Determine the range based on inversion state
        // let startRange_lineWidth = isInverted_lineWidth ? 1 : 5; // Swap the start range if inverted
        // let endRange_lineWidth = isInverted_lineWidth ? 5 : 1; // Swap the end range if inverted
        // Calculate line width based on selected feature2
        let lineWidth = 1;
        switch (selectedFeature2) {
            case "amplitude":
                lineWidth = map(feature.amplitude, minAmplitude, maxAmplitude, startRange_lineWidth, endRange_lineWidth);
                break;
            case "zerocrossingrate":
                lineWidth = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineWidth, endRange_lineWidth);
                break;
            case "spectral_centroid":
                lineWidth = map(feature.spectral_centroid, minSpectralCentroid, maxSpectralCentroid, startRange_lineWidth, endRange_lineWidth);
                break;
            case "spectral_flux":
                lineWidth = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_lineWidth, endRange_lineWidth);
                break;
            case "spectral_bandwidth":
                lineWidth = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_lineWidth, endRange_lineWidth);
                break;
            case "spectral_flatness":
                lineWidth = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_lineWidth, endRange_lineWidth);
                break;
            case "yin_f0_librosa":
                lineWidth = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_lineWidth, endRange_lineWidth);
                break;
            case "yin_f0_aubio":
                lineWidth = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_lineWidth, endRange_lineWidth);
                break;
            case "crepe_f0":
                lineWidth = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_lineWidth, endRange_lineWidth);
                break;
            case "crepe_confidence":
                lineWidth = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_lineWidth, endRange_lineWidth);
                break;
            case "yin_periodicity":
                lineWidth = map(feature.yin_periodicity, minPeriodicity, maxPeriodicity, startRange_lineWidth, endRange_lineWidth);
                break;
            case "spectral_deviation":
                lineWidth = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_lineWidth, endRange_lineWidth);
                break;
            case "brightness":
                lineWidth = map(feature.brightness, minBrightness, maxBrightness, startRange_lineWidth, endRange_lineWidth);
                break;
            case "sharpness":
                lineWidth = map(feature.sharpness, minSharpness, maxSharpness, startRange_lineWidth, endRange_lineWidth);
                break;
            case "loudness":
                lineWidth = map(feature.loudness, minLoudness, maxLoudness, startRange_lineWidth, endRange_lineWidth);
                break;
            case "none":
                lineWidth = 2
                break;
            case "mir_mps_roughness":
                lineWidth = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_lineWidth, endRange_lineWidth);
                break;
            case "mir_roughness_zwicker":
                lineWidth = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_lineWidth, endRange_lineWidth);
                break;
            case "mir_sharpness_zwicker":
                lineWidth = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_lineWidth, endRange_lineWidth);
                break;
            case "mir_roughness_vassilakis":
                lineWidth = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_lineWidth, endRange_lineWidth);
                break;
            case "mir_roughness_sethares":
                lineWidth = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_lineWidth, endRange_lineWidth);
                break;
            default:
                lineWidth = map(feature.amplitude, minAmplitude, maxAmplitude, startRange_lineWidth, endRange_lineWidth);
        }

        const slider3 = document.getElementById("slider-3"); // Get the slider element
        const invertMappingCheckbox_lineColorSaturation = document.getElementById("invertMapping-3");
        const isInverted_lineColorSaturation = invertMappingCheckbox_lineColorSaturation && invertMappingCheckbox_lineColorSaturation.checked;

        // Get the dynamic range
        const { startRange_lineColorSaturation, endRange_lineColorSaturation } = calculateDynamicRange(slider3, isInverted_lineColorSaturation, "startRange_lineColorSaturation", "endRange_lineColorSaturation");
        // console.log("isInverted_lineColorSaturation", isInverted_lineColorSaturation);
        // console.log("startRange_lineColorSaturation", startRange_lineColorSaturation);
        // console.log("endRange_lineColorSaturation", endRange_lineColorSaturation);
        
        // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_lineColorSaturation = document.getElementById("invertMapping-3"); // Update ID as needed
        // const isInverted_lineColorSaturation = invertMappingCheckbox_lineColorSaturation && invertMappingCheckbox_lineColorSaturation.checked;

        // Determine the range based on inversion state
        // let startRange_lineColorSaturation = isInverted_lineColorSaturation ? 0 : 100; // Swap the start range if inverted
        // let endRange_lineColorSaturation = isInverted_lineColorSaturation ? 100 : 0; // Swap the end range if inverted
        // Calculate color based on selected feature3
        function adjustForSaturationPeriodicity(yin_periodicity) {
            threshold = 0.3;
            return yin_periodicity < threshold ? 0 : yin_periodicity;
        }

        function calibrateZCR(zcr, spectralCentroid, sampleRate, k = 1.0) {
            let nyquist = sampleRate / 2; // Maximum possible spectral centroid
            let normalizedSC = spectralCentroid / nyquist; // Normalize SC (0 to 1)
            
            let adjustedZCR = zcr / (1 + k * normalizedSC);
            
            return adjustedZCR;
        }


        let colorValue = 0;
        let hueColorSaturation = 0;
        let hueLightness = 0;
        switch (selectedFeature3) {
            case "amplitude":
                hueColorSaturation = map(feature.amplitude, minAmplitude, maxAmplitude, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "zerocrossingrate":
                hueColorSaturation = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "calibrated-zerocrossingrate":
                hueColorSaturation = map(calibrateZCR(feature.zerocrossingrate, feature.spectral_centroid, 44100), minZerocrossingrate, maxZerocrossingrate, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "spectral_centroid":
                hueColorSaturation = map(feature.spectral_centroid, minSpectralCentroid, maxSpectralCentroid, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "spectral_flux":
                hueColorSaturation = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "spectral_bandwidth":
                hueColorSaturation = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "spectral_flatness":
                hueColorSaturation = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "yin_f0_librosa":
                hueColorSaturation = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "yin_f0_aubio":
                hueColorSaturation = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "crepe_f0":
                hueColorSaturation = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "crepe_confidence":
                hueColorSaturation = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "yin_periodicity":
                hueColorSaturation = map(adjustForSaturationPeriodicity(feature.yin_periodicity), minPeriodicity, maxPeriodicity, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "spectral_deviation":
                hueColorSaturation = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "brightness":
                hueColorSaturation = map(feature.brightness, minBrightness, maxBrightness, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "sharpness":
                hueColorSaturation = map(feature.sharpness, minSharpness, maxSharpness, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "loudness":
                hueColorSaturation = map(feature.loudness, minLoudness, maxLoudness, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "none":
                hueColorSaturation = 0;
                break;
            case "mir_mps_roughness":
                hueColorSaturation = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "mir_roughness_zwicker":
                hueColorSaturation = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "mir_sharpness_zwicker":
                hueColorSaturation = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "mir_roughness_vassilakis":
                hueColorSaturation = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            case "mir_roughness_sethares":
                hueColorSaturation = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_lineColorSaturation, endRange_lineColorSaturation);
                break;
            default:
                hueColorSaturation = map(feature.loudness, minLoudness, maxLoudness, startRange_lineColorSaturation, endRange_lineColorSaturation);
        }
        // console.log("hueColorSaturation BEFORE", hueColorSaturation);
        hueColorSaturation = Math.floor(hueColorSaturation);
        // console.log("hueColorSaturation AFTER", hueColorSaturation);
        const slider6 = document.getElementById("slider-6"); // Get the slider element
        const invertMappingCheckbox_lineColorLightness = document.getElementById("invertMapping-6");
        const isInverted_lineColorLightness = invertMappingCheckbox_lineColorLightness && invertMappingCheckbox_lineColorLightness.checked;

        // Get the dynamic range
        const { startRange_lineColorLightness, endRange_lineColorLightness } = calculateDynamicRange(slider6, isInverted_lineColorLightness, "startRange_lineColorLightness", "endRange_lineColorLightness");

        // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_lineColorLightness = document.getElementById("invertMapping-6"); // Update ID as needed
        // const isInverted_lineColorLightness = invertMappingCheckbox_lineColorLightness && invertMappingCheckbox_lineColorLightness.checked;

        // Determine the range based on inversion state
        // let startRange_lineColorLightness = isInverted_lineColorLightness ? 0 : 100; // Swap the start range if inverted
        // let endRange_lineColorLightness = isInverted_lineColorLightness ? 100 : 0; // Swap the end range if inverted
        // Calculate color based on selected feature3
       
        switch (selectedFeature6) {
            case "amplitude":
                hueLightness = map(feature.amplitude, minAmplitude, maxAmplitude, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "zerocrossingrate":
                hueLightness = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "spectral_centroid":
                hueLightness = map(feature.spectral_centroid, minSpectralCentroid, maxSpectralCentroid, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "spectral_flux":
                hueLightness = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "spectral_bandwidth":
                hueLightness = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "spectral_flatness":
                hueLightness = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "yin_f0_librosa":
                hueLightness = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "yin_f0_aubio":
                hueLightness = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "crepe_f0":
                hueLightness = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "crepe_confidence":
                hueLightness = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "yin_periodicity":
                hueLightness = map(feature.yin_periodicity, minPeriodicity, maxPeriodicity, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "spectral_deviation":
                hueLightness = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "brightness":
                hueLightness = map(feature.brightness, minBrightness, maxBrightness, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "sharpness":
                hueLightness = map(feature.sharpness, minSharpness, maxSharpness, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "loudness":
                hueLightness = map(feature.loudness, minLoudness, maxLoudness, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "none":
                hueLightness = 50;
                break;
            case "mir_mps_roughness":
                hueLightness = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "mir_roughness_zwicker":
                hueLightness = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "mir_sharpness_zwicker":
                hueLightness = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "mir_roughness_vassilakis":
                hueLightness = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            case "mir_roughness_sethares":
                hueLightness = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_lineColorLightness, endRange_lineColorLightness);
                break;
            default:
                hueLightness = map(feature.loudness, minLoudness, maxLoudness, startRange_lineColorLightness, endRange_lineColorLightness);
        }
        // console.log("y_axis:", y_axis)
        hueLightness = Math.floor(hueLightness);
        
        let hslColor = hexToHSL(huePicker.value); // Convert hex to HSL

        const slider4 = document.getElementById("slider-4"); // Get the slider element
        const invertMappingCheckbox_lineAngle = document.getElementById("invertMapping-4");
        const isInverted_lineAngle = invertMappingCheckbox_lineAngle && invertMappingCheckbox_lineAngle.checked;
        // console.log("isInverted_lineAngle:", isInverted_lineAngle);
        
        // Get the dynamic range
        const { startRange_lineAngle, endRange_lineAngle } = calculateDynamicRange(slider4, isInverted_lineAngle, "startRange_lineAngle", "endRange_lineAngle");
        // console.log("startRange_lineAngle", startRange_lineAngle, "endRange_lineAngle", endRange_lineAngle);
        // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_lineAngle = document.getElementById("invertMapping-4"); // Update ID as needed
        // const isInverted_lineAngle = invertMappingCheckbox_lineAngle && invertMappingCheckbox_lineAngle.checked;

        // Determine the range based on inversion state
        // let startRange_lineAngle = isInverted_lineAngle ? 0 : 100; // Swap the start range if inverted
        // let endRange_lineAngle = isInverted_lineAngle ? 100 : 0; // Swap the end range if inverted
        
        // Calculate angle based on selected feature4
        let angleRange = 0;
        let maxAngleRange = 90;
        switch (selectedFeature4) {
            case "amplitude":
                angleRange = map(feature.amplitude, minAmplitude, maxAmplitude,  startRange_lineAngle, endRange_lineAngle);
                break;
            case "zerocrossingrate":
                angleRange = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineAngle, endRange_lineAngle);
                break;
            case "spectral_centroid":
                angleRange = map(feature.spectral_centroid, minSpectralCentroid, maxSpectralCentroid, startRange_lineAngle, endRange_lineAngle);
                break;
            case "spectral_flux":
                angleRange = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_lineAngle, endRange_lineAngle);
                break;
            case "spectral_bandwidth":
                angleRange = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_lineAngle, endRange_lineAngle);
                break;
            case "spectral_flatness":
                angleRange = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_lineAngle, endRange_lineAngle);
                break;
            case "yin_f0_librosa":
                angleRange = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_lineAngle, endRange_lineAngle);
                break;
            case "yin_f0_aubio":
                angleRange = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_lineAngle, endRange_lineAngle);
                break;
            case "crepe_f0":
                angleRange = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_lineAngle, endRange_lineAngle);
                break;
            case "crepe_confidence":
                angleRange = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_lineAngle, endRange_lineAngle);
                break;
            case "yin_periodicity":
                angleRange = map(feature.yin_periodicity, minPeriodicity, maxPeriodicity, startRange_lineAngle, endRange_lineAngle);
                break;
            case "spectral_deviation":
                angleRange = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_lineAngle, endRange_lineAngle);
                break;
            case "brightness":
                angleRange = map(feature.brightness, minBrightness, maxBrightness, startRange_lineAngle, endRange_lineAngle);
                break;
            case "sharpness":
                angleRange = map(feature.sharpness, minSharpness, maxSharpness, startRange_lineAngle, endRange_lineAngle);
                break;
            case "loudness":
                angleRange = map(feature.loudness, minLoudness, maxLoudness, startRange_lineAngle, endRange_lineAngle);
                break;
            case "none":
                angleRange = 0
                break;
            case "mir_mps_roughness":
                angleRange = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_lineAngle, endRange_lineAngle);
                break;
            case "mir_roughness_zwicker":
                angleRange = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_lineAngle, endRange_lineAngle);
                break;
            case "mir_sharpness_zwicker":
                angleRange = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_lineAngle, endRange_lineAngle);
                break;
            case "mir_roughness_vassilakis":
                angleRange = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_lineAngle, endRange_lineAngle);
                break;
            case "mir_roughness_sethares":
                angleRange = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_lineAngle, endRange_lineAngle);
                break;
            default:
                // let angleDifference = (feature.zerocrossingrate - audioData.json_data.Song.general_info.median_zcr)/audioData.json_data.Song.general_info.iqr_zcr;

                angle = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_lineAngle, maxAngleRange);
        }

        const slider7 = document.getElementById("slider-7"); // Get the slider element
        const invertMappingCheckbox_dashArray = document.getElementById("invertMapping-7");
        const isInverted_dashArray = invertMappingCheckbox_dashArray && invertMappingCheckbox_dashArray.checked;

        // Get the dynamic range
        const { startRange_dashArray, endRange_dashArray } = calculateDynamicRange(slider7, isInverted_dashArray, "startRange_dashArray", "endRange_dashArray");

        // Check if inversion is required for the selected feature
        // const invertMappingCheckbox_dashArray = document.getElementById("invertMapping-7"); // Update ID as needed
        // const isInverted_dashArray = invertMappingCheckbox_dashArray && invertMappingCheckbox_dashArray.checked;

        // Determine the range based on inversion state
        // let startRange_dashArray = isInverted_dashArray ? 0 : 10; // Swap the start range if inverted
        // let endRange_dashArray = isInverted_dashArray ? 10 : 0; // Swap the end range if inverted

        // Calculate angle based on selected feature4
        let dashArray = 0;
        switch (selectedFeature7) {
            case "amplitude":
                dashArray = map(feature.amplitude, minAmplitude, maxAmplitude, startRange_dashArray, endRange_dashArray);
                break;
            case "zerocrossingrate":
                dashArray = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_dashArray,  endRange_dashArray);
                break;
            case "spectral_centroid":
                dashArray = map(feature.spectral_centroid, minSpectralCentroid, maxSpectralCentroid, startRange_dashArray,  endRange_dashArray);
                break;
            case "spectral_flux":
                dashArray = map(feature.spectral_flux, minSpectralFlux, maxSpectralFlux, startRange_dashArray,  endRange_dashArray);
                break;
            case "spectral_bandwidth":
                dashArray = map(feature.spectral_bandwidth, minSpectralBandwidth, maxSpectralBandwidth, startRange_dashArray, endRange_dashArray);
                break;
            case "spectral_flatness":
                dashArray = map(feature.spectral_flatness, minSpectralFlatness, maxSpectralFlatness, startRange_dashArray, endRange_dashArray);
                break;
            case "yin_f0_librosa":
                dashArray = map(feature.yin_f0_librosa, minYinF0Librosa, maxYinF0Librosa, startRange_dashArray,  endRange_dashArray);
                break;
            case "yin_f0_aubio":
                dashArray = map(feature.yin_f0_aubio, minYinF0Aubio, maxYinF0Aubio, startRange_dashArray,  endRange_dashArray);
                break;
            case "crepe_f0":
                dashArray = map(feature.crepe_f0, minF0Crepe, maxF0Crepe, startRange_dashArray,  endRange_dashArray);
                break;
            case "crepe_confidence":
                dashArray = map(feature.crepe_confidence, minF0CrepeConfidence, maxF0CrepeConfidence, startRange_dashArray,  endRange_dashArray);
                break;
            case "yin_periodicity":
                dashArray = map(feature.yin_periodicity, minPeriodicity, maxPeriodicity, startRange_dashArray, endRange_dashArray);
                break;
            case "spectral_deviation":
                dashArray = map(feature.standard_deviation, minStandardDeviation, maxStandardDeviation, startRange_dashArray,  endRange_dashArray);
                break;
            case "brightness":
                dashArray = map(feature.brightness, minBrightness, maxBrightness, startRange_dashArray,  endRange_dashArray);
                break;
            case "sharpness":
                dashArray = map(feature.sharpness, minSharpness, maxSharpness, startRange_dashArray,  endRange_dashArray);
                break;
            case "loudness":
                dashArray = map(feature.loudness, minLoudness, maxLoudness, startRange_dashArray,  endRange_dashArray);
                break;
            case "mir_mps_roughness":
                dashArray = map(feature.mir_mps_roughness, minMirMpsRoughness, maxMirMpsRoughness, startRange_dashArray, endRange_dashArray);
                break;
            case "mir_roughness_zwicker":
                dashArray = map(feature.mir_roughness_zwicker, minMirRoughnessZwicker, maxMirRoughnessZwicker, startRange_dashArray, endRange_dashArray);
                break;
            case "mir_sharpness_zwicker":
                dashArray = map(feature.mir_sharpness_zwicker, minMirSharpnessZwicker, maxMirSharpnessZwicker, startRange_dashArray, endRange_dashArray);
                break;
            case "mir_roughness_vassilakis":
                dashArray = map(feature.mir_roughness_vassilakis, minMirRoughnessVassilakis, maxMirRoughnessVassilakis, startRange_dashArray, endRange_dashArray);
                break;
            case "mir_roughness_sethares":
                dashArray = map(feature.mir_roughness_sethares, minMirRoughnessSethares, maxMirRoughnessSethares, startRange_dashArray, endRange_dashArray);
                break;
            case "loudness-zcr":
                //const BaseLength_dashArray = 0;
                const MaxZCR_dashArray = maxZerocrossingrate; // Replace with the actual maximum ZCR
                //const MaxEffect_dashAray = 10;
                // Calculate the contribution of loudness and ZCR
                let ZCRLoudnessEffect_dashArray = map(feature.loudness, minLoudness, maxLoudness, startRange_dashArray, endRange_dashArray);
                let ZCRInfluence_dashArray = feature.zerocrossingrate / MaxZCR_dashArray;

                // Combine loudness and ZCR to calculate the line length
                dashArray = startRange_dashArray + (ZCRLoudnessEffect_dashArray * ZCRInfluence_dashArray);
                break;
            case "none":
                dashArray = 0
                break;
            default:
                // let angleDifference = (feature.zerocrossingrate - audioData.json_data.Song.general_info.median_zcr)/audioData.json_data.Song.general_info.iqr_zcr;

                dashArray = map(feature.zerocrossingrate, minZerocrossingrate, maxZerocrossingrate, startRange_dashArray,  endRange_dashArray);
        }
        // angle = angle * getRandomSign() + 80;
        // Generate a random angle within the calculated range, centered around 90 degrees (vertical)
        let angle = 90 + getRandomSign() * map(Math.random(), 0, 1, 0, angleRange);
        // Convert the angle to radians for trigonometric calculations
        let angleInRadians = (angle * Math.PI) / 180;
        // angle = angle * getRandomSign()

        // Skip line drawing if loudness is below the threshold
        loudnessThreshold = 1
        if (Math.abs(feature.loudness) > loudnessThreshold) {
            

            // Calculate end points of the line
            // let x1 = x - lineLength / 2 * Math.cos(angle);
            // let y1 = y_axis - lineLength / 2 * Math.sin(angle);
            // let x2 = x + lineLength / 2 * Math.cos(angle);
            // let y2 = y_axis + lineLength / 2 * Math.sin(angle);


            // Calculate end points of the line
            let x1 = x - (lineLength / 2) * Math.cos(angleInRadians);
            let y1 = y_axis - (lineLength / 2) * Math.sin(angleInRadians);
            let x2 = x + (lineLength / 2) * Math.cos(angleInRadians);
            let y2 = y_axis + (lineLength / 2) * Math.sin(angleInRadians);      

            
            // Calculate control points for the Bezier curve
            // let ctrlX1 = x - lineLength * Math.cos(angle) / 4; // Control point 1 x-coordinate
            // let ctrlY1 = y_axis - lineLength * Math.sin(angle) / 2; // Control point 1 y-coordinate
            // let ctrlX2 = x + lineLength * Math.cos(angle) / 4; // Control point 2 x-coordinate
            // let ctrlY2 = y_axis + lineLength * Math.sin(angle) / 2; // Control point 2 y-coordinate
            

            // if (isJoinPathsEnabled) {
            
            // Number of dots to scatter for each data point
            // Number of dots to scatter for each data point
            const numDots = 10;
            // Adjust scatter range based on loudness
            let scatterRange = map(feature.loudness, minLoudness, maxLoudness, 0, 200); // Scale range based on loudness
            let currentDots = []; // To store the scattered dots of the current data point

            // Scatter multiple dots along the y-axis
            // Scatter multiple dots along the y-axis
            for (let j = 0; j < numDots; j++) {
                // Scatter dots within the specified range around the main y-axis value
                let offsetY = (j - (numDots - 1) / 2) * (scatterRange / (numDots - 1)); // Even spacing
                let scatteredY = y_axis + offsetY;

                // Create a scattered dot
                let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", x);
                circle.setAttribute("cy", scatteredY);
                circle.setAttribute("r", 3); // Radius of each dot
                circle.setAttribute("fill", "none"); // Gradient color
                svgContainer.appendChild(circle);

                // Store the current dot's position
                currentDots.push({ x, y: scatteredY });
            }

            // Store the dot position for polyline
            pointsArray.push(`${x},${y_axis}`);
            // }
            







            if (isThresholdEnabled && feature.zerocrossingrate > zeroCrossingThreshold) {
                // Draw multiple lines at different y heights (spectral centroids)
                for (let j = 0; j < 100; j++) { // Draw 5 lines at different y heights
                    // Calculate y position based on feature values
                    let y_axis = map((feature.spectral_centroid + 500)*j*0.01, 0, 5000, 800, 0);
                    // Calculate end points of the line
                    let x1 = x - lineLength / 2 * Math.cos(angle);
                    let y1 = y_axis - lineLength / 2 * Math.sin(angle*Math.random());
                    let x2 = x + lineLength / 2 * Math.cos(angle);
                    let y2 = y_axis + lineLength / 2 * Math.sin(angle*Math.random());

                    // Calculate control points for the Bezier curve
                    let ctrlX1 = x - lineLength * Math.cos(angle) / 4; // Control point 1 x-coordinate
                    let ctrlY1 = y_axis - lineLength * Math.sin(angle) / 2; // Control point 1 y-coordinate
                    let ctrlX2 = x + lineLength * Math.cos(angle) / 4; // Control point 2 x-coordinate
                    let ctrlY2 = y_axis + lineLength * Math.sin(angle) / 2; // Control point 2 y-coordinate


                    // Create a path element for the Bezier curve
                    let path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    path.setAttribute("d", `M${x1},${y1} C${ctrlX1},${ctrlY1} ${ctrlX2},${ctrlY2} ${x2},${y2}`);
                    path.setAttribute("stroke", `rgb(${colorValue},${colorValue},${colorValue})`);
                    path.setAttribute("stroke-width", lineWidth);
                    path.setAttribute("fill", "none");
                    // Generate a random number between 1 and 3
                    // let randomClassNumber = Math.floor(Math.random() * 3) + 1;

                    // Add CSS class for animation with the random number appended
                    // path.classList.add(`path-animation-${randomClassNumber}`);
                    svgContainer.appendChild(path);
                }
            } else {
                
                

                if (isThresholdCircleEnabled) {

                    // Calculate properties for circle
                    let radius = map(feature.amplitude, 0, maxAmplitude, 2, 10);
                    colorValue = map(feature.spectral_centroid, 0, maxSpectralCentroid, 100, 0);

                    let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("cx", x);
                    circle.setAttribute("cy", y_axis);
                    circle.setAttribute("r", radius);
                    circle.setAttribute("fill", `rgb(0,0,${colorValue})`);

                    svgContainer.appendChild(circle);

                    const TWO_PI = 2 * Math.PI;
                    // Spiral pattern
                    for (let t = 0; t < TWO_PI; t += 0.1) {
                        let radius = map(feature.spectral_centroid, 0, maxSpectralCentroid, 2, 10);
                        let x_spiral = x + radius * Math.cos(t) * (1 + feature.amplitude);
                        let y_spiral = y_axis + radius * Math.sin(t) * (1 + feature.amplitude);

                        let circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                        circle.setAttribute("cx", x_spiral);
                        circle.setAttribute("cy", y_spiral);
                        circle.setAttribute("r", 2);
                        circle.setAttribute("fill", `rgb(${colorValue}, ${200 - colorValue}, 255)`);

                        svgContainer.appendChild(circle);
                    }
                }
                    
            let path = document.createElementNS("http://www.w3.org/2000/svg", "path");

            // console.log("x1:",x1, "y1:",y1,"ctrlX1:", ctrlX1, "ctrlX2:", ctrlX2, "ctrlY2:", ctrlY2, "x2:", x2, "y2:", y2)
            // path.setAttribute("d", `M${x1},${y1} C${ctrlX1},${ctrlY1} ${ctrlX2},${ctrlY2} ${x2},${y2}`);
            // Set the path data for a straight line
            path.setAttribute("d", `M${x1},${y1} L${x2},${y2}`);    
            //path.setAttribute("stroke", "black"); // Set the brush color
            //path.setAttribute("stroke-width", "5"); // Set the brush thickness
            path.setAttribute("stroke-linecap", "round"); // Make the line ends rounded
            path.setAttribute("stroke-linejoin", "round"); // Make the line joins rounded
            // Example: Add features as a string (you can format it nicely)
            // Generate the data-features attribute dynamically based on the selected feature
            // Generate the data-features attribute dynamically based on the selected feature
            let featureDescription = `Timestamp: ${feature.timestamp.toFixed(2)}s<br>`;
            featureDescription += `Amplitude: ${feature.amplitude.toFixed(2)}<br>`;
            featureDescription += `Centroid: ${feature.spectral_centroid.toFixed(2)}<br>`;
            featureDescription += `Flux: ${feature.spectral_flux.toFixed(2)}<br>`;
            featureDescription += `Flatness: ${feature.spectral_flatness.toFixed(2)}<br>`;
            featureDescription += `Bandwidth: ${feature.spectral_bandwidth.toFixed(2)}<br>`;
            featureDescription += `ZCR: ${feature.zerocrossingrate.toFixed(2)}<br>`;
            featureDescription += `Confidence: ${feature.crepe_confidence.toFixed(2)}<br>`;
            featureDescription += `Combo Crepe | Crepe Conf: ${(
                feature.crepe_f0 * feature.crepe_confidence +
                (feature.spectral_centroid * (1 - feature.crepe_confidence)) * 0.2
            ).toFixed(2)}<br>`;
            featureDescription += `Crepe F0: ${feature.crepe_f0.toFixed(2)}<br>`;
            featureDescription += `Periodicity: ${feature.yin_periodicity.toFixed(2)}<br>`;
            featureDescription += `Brightness: ${feature.brightness.toFixed(2)}<br>`;
            featureDescription += `Sharpness: ${feature.sharpness.toFixed(2)}<br>`;
            featureDescription += `Loudness: ${feature.loudness.toFixed(2)}<br>`;

            // New MIR-based features
            featureDescription += `MPS Roughness: ${feature.mir_mps_roughness.toFixed(2)}<br>`;
            featureDescription += `Zwicker Roughness: ${feature.mir_roughness_zwicker.toFixed(2)}<br>`;
            featureDescription += `Zwicker Sharpness: ${feature.mir_sharpness_zwicker.toFixed(2)}<br>`;
            featureDescription += `Vassilakis Roughness: ${feature.mir_roughness_vassilakis.toFixed(2)}<br>`;
            featureDescription += `Sethares Roughness: ${feature.mir_roughness_sethares.toFixed(2)}`;
            

            // Attach the data-features attribute to the path
            path.setAttribute("data-features", featureDescription);
            // Clamp dashArray to avoid negative values
            // dashArray = Math.max(dashArray, 0);

            // Apply the stroke-dasharray
            if (dashArray == 0) {
                path.removeAttribute("stroke-dasharray"); // Solid line
            } else {
                path.setAttribute("stroke-dasharray", "4,"+dashArray.toString()); // Dashed line with spacing
            }
            // path.setAttribute("stroke", `rgb(${colorValue},${colorValue},${colorValue})`);
            // Set the stroke color using HSL
            path.setAttribute("stroke", `hsl(${hslColor}, ${hueColorSaturation}%, ${hueLightness}%)`); // 100% saturation, 50% lightness
            path.setAttribute("stroke-width", lineWidth);
            path.setAttribute("fill", "none");
            path.classList.add(`audio-path-${fileIndex}`); // Add unique class


            if (isJoinPathsEnabled) {
                // Connect the dots with lines to the previous data point
                if (previousDots.length > 0) {
                    for (let j = 0; j < numDots; j++) {
                        let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        line.setAttribute("x1", previousDots[j].x);
                        line.setAttribute("y1", previousDots[j].y);
                        line.setAttribute("x2", currentDots[j].x);
                        line.setAttribute("y2", currentDots[j].y);
                        line.setAttribute("stroke", `rgb(${j * (255 / numDots)}, 100, 200)`); // Line color
                        line.setAttribute("stroke-width", 1);
                        svgContainer.appendChild(line);
                    }
                }

                // Update previousDots to currentDots for the next iteration
                previousDots = currentDots;
            }
            
            // Generate a random number between 1 and 3
            // let randomClassNumber = Math.floor(Math.random() * 3) + 1;

            // Add CSS class for animation with the random number appended
            // path.classList.add(`path-animation-${randomClassNumber}`);
            // Store the attributes in an object
            timestamp = feature.timestamp
            pathData.push({
                timestamp,
                y_axis,
                lineLength,
                lineWidth,
                hueColorSaturation,
                hueLightness,
                dashArray,
            });
            if (isPolygonEnabled) {
                    // Map features to polygon properties
                    const numSides = Math.floor(lineLength); // Map to number of sides
                    // console.log(numSides)
                    const radius = lineWidth;
                    const fillColor = "red"

                    // Generate the SVG path data for the polygon
                    const polygonPath = polygon(x, y_axis, numSides, radius);

                    // Create the SVG <path> element for the polygon
                    const polygonElement = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    polygonElement.setAttribute("d", polygonPath);
                    polygonElement.setAttribute("fill", `hsl(${hslColor}, ${hueColorSaturation}%, ${hueLightness}%)`);
                    polygonElement.setAttribute("stroke", `hsl(${hslColor}, ${hueColorSaturation}%, ${hueLightness}%)`);
                    polygonElement.setAttribute("stroke-width", 1);

                    // Append the polygon to the SVG container
                    svgContainer.appendChild(polygonElement);

                } else {
                    svgContainer.appendChild(path);
                }
            
            // }       
        } 
    } else {
        console.log("Loudness too low, skipping line drawing.", Math.abs(feature.loudness), "at timestamp:", feature.timestamp);
    }      
}

// Step 2: Compute min-max ranges
function computeMinMaxRanges(data) {
    const minMaxValues = {
        y_axis: { min: Infinity, max: -Infinity },
        lineLength: { min: Infinity, max: -Infinity },
        lineWidth: { min: Infinity, max: -Infinity },
        hueColorSaturation: { min: Infinity, max: -Infinity },
        hueLightness: { min: Infinity, max: -Infinity },
        dashArray: { min: Infinity, max: -Infinity },
    };

    data.forEach((item) => {
        minMaxValues.y_axis.min = Math.min(minMaxValues.y_axis.min, item.y_axis);
        minMaxValues.y_axis.max = Math.max(minMaxValues.y_axis.max, item.y_axis);
        minMaxValues.lineLength.min = Math.min(minMaxValues.lineLength.min, item.lineLength);
        minMaxValues.lineLength.max = Math.max(minMaxValues.lineLength.max, item.lineLength);
        minMaxValues.lineWidth.min = Math.min(minMaxValues.lineWidth.min, item.lineWidth);
        minMaxValues.lineWidth.max = Math.max(minMaxValues.lineWidth.max, item.lineWidth);
        minMaxValues.hueColorSaturation.min = Math.min(minMaxValues.hueColorSaturation.min, item.hueColorSaturation);
        minMaxValues.hueColorSaturation.max = Math.max(minMaxValues.hueColorSaturation.max, item.hueColorSaturation);
        minMaxValues.hueLightness.min = Math.min(minMaxValues.hueLightness.min, item.hueLightness);
        minMaxValues.hueLightness.max = Math.max(minMaxValues.hueLightness.max, item.hueLightness);
        minMaxValues.dashArray.min = Math.min(minMaxValues.dashArray.min, item.dashArray);
        minMaxValues.dashArray.max = Math.max(minMaxValues.dashArray.max, item.dashArray);
    });

    // console.log("Computed min-max ranges:", minMaxValues);
    return minMaxValues;
}

// Step 3: Normalize attributes
function normalizePathData(data, minMaxValues) {
    return data.map((item) => ({
        ...item,
        y_axis: normalize(item.y_axis, minMaxValues.y_axis.min, minMaxValues.y_axis.max),
        lineLength: normalize(item.lineLength, minMaxValues.lineLength.min, minMaxValues.lineLength.max),
        lineWidth: normalize(item.lineWidth, minMaxValues.lineWidth.min, minMaxValues.lineWidth.max),
        hueColorSaturation: normalize(item.hueColorSaturation, minMaxValues.hueColorSaturation.min, minMaxValues.hueColorSaturation.max),
        hueLightness: normalize(item.hueLightness, minMaxValues.hueLightness.min, minMaxValues.hueLightness.max),
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
    // console.log("Normalized path data:", normalisedPathData);
}
preparePathData(pathData);

let audioContext;
let audioBuffer;
let animationFrameId;
let grainSourcePosition = 0; // Default grain source position
let activeGrains = []; // To track active grains and stop them
let masterGainNode; // Master gain node for volume control
let playbackMarkerX = 0; // Position of the playback marker
let isPlaying = false; // Playback state

const canvas = document.getElementById("waveformCanvas");
const ctx = canvas.getContext("2d");

// Draw the waveform
function drawWaveform(buffer) {
    const data = buffer.getChannelData(0); // Use the first channel
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const step = Math.ceil(data.length / canvasWidth);
    const amp = canvasHeight / 2;

    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < canvasWidth; i++) {
        const min = Math.min(...data.slice(i * step, (i + 1) * step));
        const max = Math.max(...data.slice(i * step, (i + 1) * step));
        ctx.lineTo(i, (1 + max) * amp);
        ctx.lineTo(i, (1 + min) * amp);
    }
    ctx.lineTo(canvasWidth, amp);
    ctx.strokeStyle = "blue";
    ctx.stroke();
}

// Draw the red line (playback marker)
function drawRedLine(position) {
    const canvasWidth = canvas.width;
    const x = (position / audioBuffer.duration) * canvasWidth;

    ctx.save();
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3; // Make it bolder
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
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

// Handle grain source position on click
canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left; // X coordinate relative to canvas
    grainSourcePosition = (x / canvas.width) * audioBuffer.duration; // Map to audio duration

    playbackSpreads = []; // Reset spreads
    updateCanvas(grainSourcePosition, 0); // Redraw with updated grain position
    // console.log(`Grain source position set to ${grainSourcePosition} seconds.`);
});

// Animate grain spread and marker during playback
function animateGrainSpread(totalDuration, grainSpread) {
    const startTime = performance.now();

    function update() {
        const elapsedTime = (performance.now() - startTime) / 1000; // Convert to seconds

        if (elapsedTime > totalDuration || !isPlaying) {
            isPlaying = false;
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

// Load the waveform after audio is decoded
document.getElementById("audioFileInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function () {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            audioContext.decodeAudioData(reader.result, (buffer) => {
                audioBuffer = buffer;
                drawWaveform(audioBuffer); // Draw the waveform once the buffer is loaded
            });
        };
        reader.readAsArrayBuffer(file);
    }
});

// Start playback and animate grain spread
document.getElementById("playGranulator").addEventListener("click", function () {
    if (!audioBuffer) {
        alert("Please load an audio file first!");
        return;
    }

    const totalDuration = audioBuffer.duration; // Assume duration based on audio file
    isPlaying = true;

    const grainSpread = 0.2; // Define spread in seconds
    // animateGrainSpread(totalDuration, grainSpread);
});

// Load and decode the audio file
document.getElementById("audioFileInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function () {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            audioContext.decodeAudioData(reader.result, (buffer) => {
                audioBuffer = buffer;
                drawWaveform(audioBuffer); // Draw the waveform once the buffer is loaded


                // Initialize masterGainNode
                masterGainNode = audioContext.createGain();
                masterGainNode.gain.value = 0.8; // Default volume, adjust as needed
                masterGainNode.connect(audioContext.destination);

                console.log("Audio loaded and ready for granulation.");
            });
        };
        reader.readAsArrayBuffer(file);
    }
});

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
function updateLinePosition(currentTimestamp, totalDuration, canvasWidth) {
    const line = document.getElementById("progressLine");
    if (!line) return;

    const x = (currentTimestamp / totalDuration) * canvasWidth;
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
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

        updateLinePosition(elapsedTime, totalDuration, canvasWidth);

        function calculatePlaybackRate(y_axis, isInverted = false) {
            // const normalizedY = y_axis / 800; // Assuming the canvas height is 800px
            const playbackRate = isInverted 
                ? map(y_axis, 0, 1, 2, 0.5) // Inverted: High y_axis -> Low playback rate
                : map(y_axis, 0, 1, 0.5, 2); // Normal: High y_axis -> High playback rate
            return playbackRate;
        }

        // Find paths that should trigger grains
        normalisedPathData.forEach((path) => {
            if (elapsedTime >= path.timestamp && elapsedTime < path.timestamp + 0.1) {
                const playbackRate = calculatePlaybackRate(path.y_axis, true); // Pass `true` for inversion

                playGrain(grainSourcePosition, path.dashArray, path.lineWidth, path.lineLength   * 0.05, playbackRate);
            }
        });

        animationFrameId = requestAnimationFrame(update);
    }

    update();
}

// Play a single grain
function playGrain(baseSourcePosition, dashArraySpread, duration, volume, playbackRate) {
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;

    // Calculate a randomized source position based on the dashArray spread
    const randomSpread = (Math.random() - 0.5) * dashArraySpread; // Spread around the base position
    let grainSourcePosition = baseSourcePosition + randomSpread;

    // Ensure the position loops within the audio buffer's duration
    grainSourcePosition = Math.max(0, grainSourcePosition % audioBuffer.duration);

    source.playbackRate.value = playbackRate;

    const gainNode = audioContext.createGain();
    const now = audioContext.currentTime;

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

// Map features to path attributes and start granulation
document.getElementById("playGranulator").addEventListener("click", function () {
    if (!audioBuffer) {
        alert("Please load an audio file first!");
        return;
    }

    startGranulation();
});

// // Adjust grain source position using slider
// document.getElementById("grainSourcePositionSlider").addEventListener("input", function (event) {
//     grainSourcePosition = parseFloat(event.target.value);
//     console.log(`Grain source position set to ${grainSourcePosition} seconds.`);
// });




// Define variables for the synth
let synthAudioContext = new (window.AudioContext || window.webkitAudioContext)();
let synthIsPlaying = false; // To manage the playback state
let synthPathData = []; // Store normalized path data for the synth
// let selectedWaveform = "sine"; // Default waveform
synthPathData = pathData;

// Play a wave with specified parameters
function playSynthWave(frequency, volume, duration, distortion) {
    const oscillator = synthAudioContext.createOscillator();
    const gainNode = synthAudioContext.createGain();
    const waveShaper = synthAudioContext.createWaveShaper();

    // Configure oscillator
    oscillator.type = selectedWaveform; // Use the selected waveform
    oscillator.frequency.value = frequency;

    // Configure gain with smooth fade-in and fade-out
    const now = synthAudioContext.currentTime;
    gainNode.gain.setValueAtTime(0, now); // Start at 0 volume
    gainNode.gain.linearRampToValueAtTime(volume, now + 0.01); // Fade-in
    gainNode.gain.setValueAtTime(volume, now + duration - 0.01); // Sustain
    gainNode.gain.linearRampToValueAtTime(0, now + duration); // Fade-out

    // Configure distortion
    waveShaper.curve = createDistortionCurve(distortion);
    waveShaper.oversample = "4x";

    // Connect nodes
    oscillator.connect(waveShaper).connect(gainNode).connect(synthAudioContext.destination);

    // Start and stop oscillator
    oscillator.start(now);
    oscillator.stop(now + duration);

    console.log(
        `Synth Wave: Waveform ${selectedWaveform}, Frequency ${frequency}Hz, Volume ${volume}, Duration ${duration}s, Distortion ${distortion}`
    );
}

// Generate a distortion curve
function createDistortionCurve(amount) {
    const curve = new Float32Array(44100);
    const k = typeof amount === "number" ? amount : 50;
    const deg = Math.PI / 180;
    for (let i = 0; i < 44100; i++) {
        const x = (i * 2) / 44100 - 1;
        curve[i] = ((3 + k) * x * 20 * deg) / (Math.PI + k * Math.abs(x));
    }
    return curve;
}

// Play the sketch using the synth
function playSketchWithSynth() {
    if (!synthPathData || synthPathData.length === 0) {
        alert("No sketch data available to play!");
        return;
    }
    toggleSynthButton.textContent = "Stop Synth"; // Update button text

    synthIsPlaying = true;

    synthPathData.forEach((path) => {
        const frequency = mapSynth(path.y_axis, 800, 0, 100, 1000); // Map y_axis (reverse) to pitch
        const volume = mapSynth(path.lineLength, 0, 100, 0.1, 1); // Map lineLength to volume
        const duration = mapSynth(path.lineWidth, 0, 10, 0.1, 1); // Map lineWidth to duration
        const distortion = mapSynth(path.dashArray, 0, 50, 0, 100); // Map dashArray to distortion

        setTimeout(() => {
            if (synthIsPlaying) {
                playSynthWave(frequency, volume, duration, distortion);
            }
        }, path.timestamp * 1000); // Trigger based on timestamp
    });
}

// Map utility function for the synth
function mapSynth(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

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

const toggleSynthButton = document.getElementById("toggleSynth");

// Update waveform image and synth waveform
function updateWaveform(waveform) {
    // Update the displayed image
    const waveformImage = document.getElementById("waveformImage");
    waveformImage.src = waveformImages[waveform];
    waveformImage.alt = `${waveform.charAt(0).toUpperCase() + waveform.slice(1)} Waveform`;

    // Update the selected waveform for the synth
    selectedWaveform = waveform;
    console.log(`Waveform changed to ${selectedWaveform}`);
}

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

// Toggle play and stop functionality
toggleSynthButton.addEventListener("click", () => {
    if (synthIsPlaying) {
        // selectedWaveform = event.target.value;
        console.log(selectedWaveform)
        stopSynthPlayback();
    } else {
        playSketchWithSynth();
    }
});
// Function to stop the synth
function stopSynthPlayback() {
    synthIsPlaying = false;
    toggleSynthButton.textContent = "Play Synth"; // Update button text

    // Your existing stop logic here
    // clearInterval(synthPlaybackInterval); // Clear playback interval
    console.log("Synth playback stopped.");
}

// // Add event listener for the synth play button
// document.getElementById("playSynth").addEventListener("click", playSketchWithSynth);

// // Add event listener for stopping synth playback
// document.getElementById("stopSynth").addEventListener("click", stopSynthPlayback);











// let clickedPosition = 0; // Global variable to store the clicked position
const svgElement = document.getElementById("svgCanvas");
let selectedElements = [];

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
        type: "x,y",
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

let dragSelectInstance = initializeSvgDragSelect();



    // cleanup for when the select-on-drag behavior is no longer needed
    // (including unbinding of the event listeners)
    // cancel()


}



function drawClusterOverlays(audioData, svgContainer, canvasWidth, canvasHeight, maxDuration) {
    audioData.forEach(cluster => {
        cluster.regions.forEach(region => {
            const startX = map(region.start_time, 0, maxDuration, 0, canvasWidth);
            const endX = map(region.end_time, 0, maxDuration, 0, canvasWidth);
            const width = endX - startX;

            const overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            overlay.setAttribute("x", startX);
            overlay.setAttribute("y", 0);
            overlay.setAttribute("width", width);
            overlay.setAttribute("height", canvasHeight);
            overlay.setAttribute("fill", cluster.color);
            overlay.setAttribute("opacity", 0.15);  // Adjust for desired transparency
            overlay.setAttribute("class", "cluster-overlay");

            svgContainer.appendChild(overlay);
        });
    });
}