export function buildSynthPathEntry(feature, frame, colorHue) {
    return {
        timestamp: feature["timestamp"],
        yAxis: frame.yAxis,
        lineLength: frame.lineLength,
        lineWidth: frame.lineWidth,
        colorHue,
        colorSaturation: frame.colorSaturation,
        colorLightness: frame.colorLightness,
        angle: frame.angle,
        dashArray: frame.dashArray,
    };
}

export function appendSynthPathFrame(pathData, fileIndex, feature, frame, colorHue) {
    pathData[fileIndex].push(buildSynthPathEntry(feature, frame, colorHue));
}

export function computeMinMaxRanges(data) {
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

export function normalizePathData(data, minMaxValues) {
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

export function normalize(value, min, max) {
    return (value - min) / (max - min);
}

export function preparePathData(pathData) {
    const minMaxValues = computeMinMaxRanges(pathData);
    normalisedPathData = normalizePathData(pathData, minMaxValues);
}
