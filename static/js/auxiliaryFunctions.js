// GENERAL UTIL FUNCTIONS:

function floatsAreEqual(a,b,epsilon = 1e-9)
{
    return Math.abs(a - b) < epsilon;
}

//! Added this for "deterministic randomness"
function random_engine(seed) {
    return function() {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
}
let seed = 0;
let rng = random_engine(seed);

// Function to generate a random value of -1 or 1
function getRandomSign() {
    return rng() < 0.5 ? -1 : 1;
}

function clampAndMap(value,inMin,inMax,outMin,outMax)
{
    const clamped = Math.max(inMin,Math.min(value,inMax));
    return map(clamped,inMin,inMax,outMin,outMax);
}

function clamp(value,min,max)
{
    return Math.max(min,Math.min(value,max));
}

// SMOOTHING FUNCTION

// Moving Median
function movingMedian(data, windowSize = 5) {
    const half = Math.floor(windowSize / 2);
    const result = [];
  
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - half);
      const end = Math.min(data.length, i + half + 1);
      const window = data.slice(start, end).filter(v => v != null && !isNaN(v));
  
      if (window.length === 0) {
        result.push(null); // fallback if all values are invalid
      } else {
        window.sort((a, b) => a - b);
        const mid = Math.floor(window.length / 2);
        const median = window.length % 2 !== 0
          ? window[mid]
          : (window[mid - 1] + window[mid]) / 2;
        result.push(median);
      }
    }
  
    return result;
}

function medfilt(signal,kernelSize = 3)
{
    // Ensure kernel size is odd
    if (kernelSize % 2 === 0) kernelSize += 1;
    const half = Math.floor(kernelSize / 2);
    const n = signal.length;
    const result = new Array(n);

    // Zero-padding at the boundaries
    const padded = new Array(n + 2 * half).fill(0);
    for (let i = 0; i < n; i++)
    {
        padded[i + half] = signal[i];
    }

    // Sliding window median computation
    for (let i = 0; i < n; i++)
    {
        const win = padded.slice(i, i + kernelSize);
        win.sort((a, b) => a - b);
        result[i] = win[half];
    }

    return result;
}

function calculate_optimal_length(overlap,min_length = 3,max_length = 11)
{
    let length = map(overlap,0,1,min_length,max_length);
    length = Math.ceil(length);
    length = length % 2 ? length : length + 1;
    return length;
}

// MAPPING FUNCTIONS:
function hzToMel(f)
{
    return 2595*Math.log10(1 + f/700);
}

function melToHz(m)
{
    return 700*(Math.pow(10,m/2595) - 1);
}

// Utility function for mapping
function map(value, start1, stop1, start2, stop2)
{
    let mapped_value = (start2 + stop2) / 2;
    //! Added this check
    if(stop1 !== start1)
    {
        mapped_value = start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
    }
    return mapped_value;
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

    //console.log("INSIDE mapToLogScale:", "value:",  value, "minValue:", minValue, "maxValue:", maxValue, "canvasHeight:", canvasHeight)

    //! Added this
    value = value >= 1 ? value : 1;
    minValue = minValue >= 1 ? minValue : 1;
    maxValue = maxValue >= 1 ? maxValue : 1;

    //! Removed "+ 1" from log10
    if (invert)
    {
        // Inverted mapping: High values at the bottom, low values at the top
        return map(
            Math.log10(value),                   // Logarithmic transformation of value
            Math.log10(minValue),               // Logarithmic min value
            Math.log10(maxValue),               // Logarithmic max value
            topPadding,                         // Top of the canvas becomes bottom
            canvasHeight - topPadding           // Bottom of the canvas becomes top
        );
    }
    else
    {
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

function mapToMelScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {

    if (invert)
    {
        // Inverted mapping: High values at the bottom, low values at the top
        return map(
            hzToMel(value),                   // Logarithmic transformation of value
            hzToMel(minValue),               // Logarithmic min value
            hzToMel(maxValue),               // Logarithmic max value
            topPadding,                         // Top of the canvas becomes bottom
            canvasHeight - topPadding           // Bottom of the canvas becomes top
        );
    }
    else
    {
        // Standard mapping: Low values at the bottom, high values at the top
        return map(
            hzToMel(value),                   // Logarithmic transformation of value
            hzToMel(minValue),               // Logarithmic min value
            hzToMel(maxValue),               // Logarithmic max value
            canvasHeight - topPadding,          // Bottom of the canvas
            topPadding                          // Top of the canvas
        );
    }
}

// Use the updated scale in your mapping

/**
 * scale controls how strong the soft clipping is in the mapping.
 *
 * - 1: Almost linear, very gentle curve
 * - 3–5: Moderate soft clipping (good default)
 * - 7–10: Strong compression at edges, focus near center
 * - >15: Extreme sigmoid shape, mostly flat at edges
 *
 * Higher values = more emphasis near the middle, less sensitivity at extremes.
 */

//! New version
function mapWithSoftClipping(value,minInput,maxInput,minOutput,maxOutput,shift = 0,scale = 10)
{
    let clipped;
    if(maxInput !== minInput)
    {
        const normalized = (value - minInput)/(maxInput - minInput);
        let compressed;
        if(scale != 0)
        {
            const compressor = (x) => Math.tanh(scale*(x - shift));
            const compressor_min = compressor(0);
            const compressor_max = compressor(1);
            compressed = compressor(normalized);
            compressed = (compressed - compressor_min)/(compressor_max - compressor_min);
        }
        else
        {
            compressed = normalized;
        }
        clipped = minOutput + compressed*(maxOutput - minOutput);
    }
    else
    {
        clipped = (minOutput + maxOutput)/2;
    }
    return clipped
}

//! Deprecated
// function mapWithSoftClipping(value, minInput, maxInput, minOutput, maxOutput, scale = 10)
// {
//     let mapped_value = (minOutput + maxOutput)/2;
//     if(maxInput !== minInput)
//     {
//         const normalized = (value - minInput)/(maxInput - minInput); // Normalize to [0, 1]
//         let clipped = normalized;
//         if(scale != 0)
//         {
//             clipped = (Math.atan(scale*(normalized - 0.5))/Math.atan(0.5*scale))*0.5 + 0.5; //! Formula below is wrong
//         }
//         // const clipped = Math.atan(scale * (normalized - 0.5)) / Math.atan(scale) + 0.5; // Adjust scale for soft clipping
//         mapped_value = minOutput + clipped*(maxOutput - minOutput);
//     }
//     return mapped_value
// }

// SKETCHING FUNCTIONS:

function generatePolygonPath(cx,cy,corners,radius)
{
    let path = "";
    for (let i = 0; i < corners; i++)
    {
        const angle = (2 * Math.PI * i) / corners - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        path += (i === 0 ? "M " : "L ") + x + " " + y + " ";
    }
    path += "Z";
    return path;
}

function createPattern(density, cx, cy, radius, opacity = 1,fillColor = "white", strokeColor = "gray") {

    if (typeof createPattern.counter === "undefined") {
        createPattern.counter = 0;
    }
    const patternId = `pattern${createPattern.counter++}`;

    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    pattern.setAttribute("id", patternId);

    const angle = 45;

    const baseSpacing = radius * 1.4;
    const spacing = Math.max(3, baseSpacing * (1 - density * 0.8));

    pattern.setAttribute("width", spacing);
    pattern.setAttribute("height", spacing);
    pattern.setAttribute("patternUnits", "userSpaceOnUse");

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", 0);
    rect.setAttribute("y", 0);
    rect.setAttribute("width", spacing);
    rect.setAttribute("height", spacing);
    rect.setAttribute("fill", fillColor);
    rect.setAttribute("fill-opacity", opacity);
    pattern.appendChild(rect);

    const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line1.setAttribute("x1", 0);
    line1.setAttribute("y1", 0);
    line1.setAttribute("x2", spacing);
    line1.setAttribute("y2", spacing);
    line1.setAttribute("stroke", strokeColor);
    line1.setAttribute("stroke-width", 2);
    line1.setAttribute("stroke-opacity", density);

    const line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line2.setAttribute("x1", 0);
    line2.setAttribute("y1", spacing);
    line2.setAttribute("x2", spacing);
    line2.setAttribute("y2", 0);
    line2.setAttribute("stroke", strokeColor);
    line2.setAttribute("stroke-width", 2);
    line2.setAttribute("stroke-opacity", density);

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.appendChild(line1);
    g.appendChild(line2);
    pattern.appendChild(g);

    pattern.setAttribute(
        "patternTransform",
        `translate(${cx - spacing / 2}, ${cy - spacing / 2}) rotate(${angle}, ${spacing / 2}, ${spacing / 2})`
    );

    return {pattern,patternId};
}

// function polarToCartesian(centerX, centerY, radius, angleInDegrees) {
//     var angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
//     return {
//         x: centerX + (radius * Math.cos(angleInRadians)),
//         y: centerY + (radius * Math.sin(angleInRadians))
//     };
// }

// // FUNCTION TO CREATE POLYGON
// function polygon(centerX, centerY, points, radius) {
//     var degreeIncrement = 360 / points;
//     var d = "M"; // Start the SVG path string
//     for (var i = 0; i < points; i++) {
//         var angle = degreeIncrement * i;
//         var point = polarToCartesian(centerX, centerY, radius, angle);
//         d += point.x + "," + point.y + " ";
//     }
//     d += "Z"; // Close the path
//     return d;
// }

// FUNCTION TO CONVERT HEX TO HSL
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

function hslToHex(h, s, l) {
    h = h % 360;
    s /= 100;
    l /= 100;

    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;

    if (0 <= h && h < 60) {
        r = c; g = x; b = 0;
    } else if (60 <= h && h < 120) {
        r = x; g = c; b = 0;
    } else if (120 <= h && h < 180) {
        r = 0; g = c; b = x;
    } else if (180 <= h && h < 240) {
        r = 0; g = x; b = c;
    } else if (240 <= h && h < 300) {
        r = x; g = 0; b = c;
    } else if (300 <= h && h < 360) {
        r = c; g = 0; b = x;
    }

    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);

    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function perceivedBrightness(r, g, b) {
    // return 0.299*r + 0.587*g + 0.114*b;
    return 0.2126*r + 0.7152*g + 0.0722*b;
}

// Convert HSL (Hue, Saturation, Lightness) to RGB
function hslToRgb(h, s, l){
    s /= 100;
    l /= 100;
    const k = n => (n + h/30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => l - a * Math.max(Math.min(k(n)-3, 9-k(n), 1), -1);
    return [Math.round(f(0)*255), Math.round(f(8)*255), Math.round(f(4)*255)];
}

//! Modified to invert y axis
function drawYAxisScale(canvasHeight, minValue, maxValue, scale = "linear", topPadding = 50, labelFormatter = (v) => v.toFixed(0),invert = false)
{
    const svgCanvas = document.getElementById("svgCanvas");
  
    // Clear old ticks
    const oldScale = svgCanvas.querySelector(".y-axis-scale");
    if (oldScale) svgCanvas.removeChild(oldScale);
  
    const scaleGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    scaleGroup.setAttribute("class", "y-axis-scale");
    scaleGroup.setAttribute("style", "user-select: none;");
  
    //! Added this check
    let numTicks;
    if(minValue !== maxValue)
    {
        numTicks = 10;
    }
    else
    {
        numTicks = 1;
    }

    //! Added this
    const mapY = (value,min,max) =>
    {
        if(invert)
        {
            // inverted: min at top, max at bottom
            return map(value,min,max,topPadding,canvasHeight - topPadding);
        }
        else
        {
            // normal: min at bottom, max at top
            return map(value,min,max,canvasHeight - topPadding,topPadding);
        }
    };
  
    if(scale == "mel")
    {
        const melMin = hzToMel(minValue);
        const melMax = hzToMel(maxValue);
        for (let i = 0; i <= numTicks; i++)
        {
            const melValue = melMin + (i / numTicks) * (melMax - melMin);
            const value  = melToHz(melValue);
            const y = mapY(melValue,melMin,melMax);
            drawTick(y,labelFormatter(value),scaleGroup);
        }

    }
    else if(scale == "log")
    {
        const logMin = Math.log10(minValue + 1);
        const logMax = Math.log10(maxValue + 1);

        for (let i = 0; i <= numTicks; i++)
        {
            const logValue = logMin + (i / numTicks) * (logMax - logMin);
            const value = Math.pow(10, logValue) - 1; //! Added - 1 here
            // const y = map(logValue, logMin, logMax, canvasHeight - topPadding, topPadding);
            let y = mapY(logValue, logMin, logMax); //! Changed to this
            drawTick(y, labelFormatter(value), scaleGroup);
        }
    }
    else if(scale == "linear")
    {
        for (let i = 0; i <= numTicks; i++)
        {
            const value = minValue + (i / numTicks) * (maxValue - minValue);
            // const y = map(value, minValue, maxValue, canvasHeight - topPadding, topPadding);
            const y = mapY(value, minValue, maxValue); //! Changed to this
            drawTick(y, labelFormatter(value), scaleGroup);
        }
    }
    svgCanvas.appendChild(scaleGroup);
  }

function drawTick(y, labelValue, group) {
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", 0);
    tick.setAttribute("x2", 10);
    tick.setAttribute("y1", y);
    tick.setAttribute("y2", y);
    tick.setAttribute("stroke", "black");
    group.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", 15);
    label.setAttribute("y", y + 3);
    label.setAttribute("font-size", "10");
    label.textContent = labelValue;
    group.appendChild(label);
}




// UI FUNCTIONS:

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

// FEATURE DETICATED FUNCTIONS:
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

function adjustPeriodicity(yin_periodicity) {
    threshold = 0.85;
    return yin_periodicity > threshold ? 1 : 0;
}

function safeLogInput(value, shift = 1) {
    return Math.max(value + shift, 1);
}

