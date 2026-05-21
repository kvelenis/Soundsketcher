// General math, color, SVG, and mapping helpers shared by the sandbox scripts.

function floatsAreEqual(a, b, epsilon = 1e-9) {
    return Math.abs(a - b) < epsilon;
}

function dBToGain(dB, threshold = -100) {
    return dB <= threshold ? 0 : Math.pow(10, dB / 20);
}

function random_engine(seed) {
    return function () {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

var seed = 0;
var rng = random_engine(seed);

function getRandomSign() {
    return rng() < 0.5 ? -1 : 1;
}

function clampAndMap(value, inMin, inMax, outMin, outMax) {
    const clamped = Math.max(inMin, Math.min(value, inMax));
    return map(clamped, inMin, inMax, outMin, outMax);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
}

function movingMedian(data, windowSize = 5) {
    const half = Math.floor(windowSize / 2);
    const result = [];

    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(data.length, i + half + 1);
        const windowValues = data.slice(start, end).filter((value) => value != null && !isNaN(value));

        if (windowValues.length === 0) {
            result.push(null);
        } else {
            windowValues.sort((a, b) => a - b);
            const mid = Math.floor(windowValues.length / 2);
            result.push(windowValues.length % 2 !== 0
                ? windowValues[mid]
                : (windowValues[mid - 1] + windowValues[mid]) / 2);
        }
    }

    return result;
}

function medfilt(signal, kernelSize = 3) {
    if (kernelSize % 2 === 0) kernelSize += 1;
    const half = Math.floor(kernelSize / 2);
    const result = new Array(signal.length);
    const padded = new Array(signal.length + 2 * half).fill(0);

    for (let i = 0; i < signal.length; i++) {
        padded[i + half] = signal[i];
    }

    for (let i = 0; i < signal.length; i++) {
        const win = padded.slice(i, i + kernelSize);
        win.sort((a, b) => a - b);
        result[i] = win[half];
    }

    return result;
}

function calculate_optimal_length(overlap, min_length = 3, max_length = 11) {
    let length = map(overlap, 0, 1, min_length, max_length);
    length = Math.ceil(length);
    return length % 2 ? length : length + 1;
}

function hzToMel(f) {
    return 2595 * Math.log10(1 + f / 700);
}

function melToHz(m) {
    return 700 * (Math.pow(10, m / 2595) - 1);
}

function map(value, start1, stop1, start2, stop2) {
    if (stop1 === start1) {
        return (start2 + stop2) / 2;
    }

    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}

function mapToLinearScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {
    return invert
        ? map(value, minValue, maxValue, topPadding, canvasHeight - topPadding)
        : map(value, minValue, maxValue, canvasHeight - topPadding, topPadding);
}

function mapToLogScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {
    value = value >= 1 ? value : 1;
    minValue = minValue >= 1 ? minValue : 1;
    maxValue = maxValue >= 1 ? maxValue : 1;

    return invert
        ? map(Math.log10(value), Math.log10(minValue), Math.log10(maxValue), topPadding, canvasHeight - topPadding)
        : map(Math.log10(value), Math.log10(minValue), Math.log10(maxValue), canvasHeight - topPadding, topPadding);
}

function mapToMelScale(value, minValue, maxValue, canvasHeight, topPadding = 50, invert = false) {
    return invert
        ? map(hzToMel(value), hzToMel(minValue), hzToMel(maxValue), topPadding, canvasHeight - topPadding)
        : map(hzToMel(value), hzToMel(minValue), hzToMel(maxValue), canvasHeight - topPadding, topPadding);
}

function mapWithSoftClipping(value, minInput, maxInput, minOutput, maxOutput, shift = 0, scale = 10) {
    if (maxInput === minInput) {
        return (minOutput + maxOutput) / 2;
    }

    const normalized = (value - minInput) / (maxInput - minInput);
    let compressed = normalized;

    if (scale !== 0) {
        const compressor = (x) => Math.tanh(scale * (x - shift));
        const compressorMin = compressor(0);
        const compressorMax = compressor(1);
        compressed = (compressor(normalized) - compressorMin) / (compressorMax - compressorMin);
    }

    return minOutput + compressed * (maxOutput - minOutput);
}

function generatePolygonPath(cx, cy, corners, radius) {
    let path = "";

    for (let i = 0; i < corners; i++) {
        const angle = (2 * Math.PI * i) / corners - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        path += (i === 0 ? "M " : "L ") + x + " " + y + " ";
    }

    return `${path}Z`;
}

function createPattern(density, cx, cy, radius, opacity = 1, fillColor = "white", strokeColor = "gray") {
    if (typeof createPattern.counter === "undefined") {
        createPattern.counter = 0;
    }

    const patternId = `pattern${createPattern.counter++}`;
    const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
    const spacing = Math.max(3, radius * 1.4 * (1 - density * 0.8));

    pattern.setAttribute("id", patternId);
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

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.appendChild(line1);
    group.appendChild(line2);
    pattern.appendChild(group);

    pattern.setAttribute(
        "patternTransform",
        `translate(${cx - spacing / 2}, ${cy - spacing / 2}) rotate(45, ${spacing / 2}, ${spacing / 2})`
    );

    return { pattern, patternId };
}

function hexToHSL(hex) {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h;
    const l = (max + min) / 2;

    if (max === min) {
        h = 0;
    } else {
        const d = max - min;
        switch (max) {
            case r:
                h = (g - b) / d + (g < b ? 6 : 0);
                break;
            case g:
                h = (b - r) / d + 2;
                break;
            default:
                h = (r - g) / d + 4;
                break;
        }
        h /= 6;
    }

    return Math.round(h * 360);
}

function hslToHex(h, s, l) {
    h %= 360;
    s /= 100;
    l /= 100;

    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r = 0;
    let g = 0;
    let b = 0;

    if (0 <= h && h < 60) {
        r = c; g = x;
    } else if (60 <= h && h < 120) {
        r = x; g = c;
    } else if (120 <= h && h < 180) {
        g = c; b = x;
    } else if (180 <= h && h < 240) {
        g = x; b = c;
    } else if (240 <= h && h < 300) {
        r = x; b = c;
    } else if (300 <= h && h < 360) {
        r = c; b = x;
    }

    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);

    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function perceivedBrightness(r, g, b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function hslToRgb(h, s, l) {
    s /= 100;
    l /= 100;
    const k = (n) => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => l - a * Math.max(Math.min(k(n) - 3, 9 - k(n), 1), -1);
    return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

function getDynamicRange(slider) {
    const values = slider.noUiSlider.get();
    return {
        minValue: parseFloat(values[0]),
        maxValue: parseFloat(values[1]),
    };
}

function calculateDynamicRange(slider, isInverted, startKey = "startRange", endKey = "endRange") {
    const [minValue, maxValue] = slider.noUiSlider.get().map(parseFloat);

    return {
        [startKey]: isInverted ? maxValue : minValue,
        [endKey]: isInverted ? minValue : maxValue,
    };
}

function safeLogInput(value, shift = 1) {
    return Math.max(value + shift, 1);
}
