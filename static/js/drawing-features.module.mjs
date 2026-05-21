function getMappedFeatureValue(feature, featureName, config, startRange, endRange, applySoftclip = false, softclipScale = 10, defaultValue = 0) {
    if (featureName === "none") {
        return defaultValue;
    }

    const entry = config[featureName];
    if (!entry) {
        return 0;
    }

    const min = entry.min;
    const max = entry.max;
    const clamped = clamp(entry.val(feature), min, max);

    if (applySoftclip && clampConfig[featureName]) {
        const shift = (entry.median - min) / (max - min);
        return mapWithSoftClipping(clamped, min, max, startRange, endRange, shift, softclipScale);
    }

    return map(clamped, min, max, startRange, endRange);
}

function getYAxisScaleFromConfig(featureConfig, featureName, scale = "linear") {
    const entry = featureConfig[featureName];
    if (!entry) {
        return null;
    }

    const min = entry.min;
    const max = entry.max;

    const labelFormatter = scale !== "linear"
        ? (value) => (value < 10 ? value.toFixed(2) : value.toFixed(0))
        : (value) => value.toFixed(value < 10 ? 2 : 0);

    return { min, max, labelFormatter };
}

function computeYAxisValue(feature, featureName, config, height, padding, inverted, scale, min, max, applySoftclip = false, softclipScale = 10) {
    let value = null;
    const entry = config[featureName];
    const mapFn = scale === "linear" ? mapToLinearScale :
                  scale === "log" ? mapToLogScale :
                  scale === "mel" ? mapToMelScale :
                  undefined;

    if (!entry) {
        value = (min + max) / 2;
        return mapFn(value, min, max, height, padding, inverted);
    }

    const minValue = entry.min;
    const maxValue = entry.max;
    value = clamp(entry.val(feature), minValue, maxValue);

    if (applySoftclip && clampConfig[featureName]) {
        const shift = (entry.median - minValue) / (maxValue - minValue);
        value = mapWithSoftClipping(value, minValue, maxValue, minValue, maxValue, shift, softclipScale);
    }

    return mapFn(value, min, max, height, padding, inverted);
}

function computeRobustStats(data, lowerPercentile = 5, upperPercentile = 95) {
    const length = data.length;
    if (!length) {
        return { min: 0, max: 1 };
    }

    const sorted = [...data].sort((a, b) => a - b);
    let lowerIndex;
    let upperIndex;

    if (lowerPercentile == upperPercentile) {
        const index = Math.round((lowerPercentile / 100) * (length - 1));
        lowerIndex = index;
        upperIndex = index;
    } else {
        lowerIndex = Math.floor((lowerPercentile / 100) * (length - 1));
        upperIndex = Math.ceil((upperPercentile / 100) * (length - 1));
    }

    let mid = Math.floor(length / 2);
    const median = length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

    const clamped = sorted.slice(lowerIndex, upperIndex + 1);
    const clampedLength = clamped.length;
    mid = Math.floor(clampedLength / 2);
    const clampedMedian = clampedLength % 2 !== 0 ? clamped[mid] : (clamped[mid - 1] + clamped[mid]) / 2;

    return {
        min: sorted[lowerIndex],
        max: sorted[upperIndex],
        median,
        clamped_median: clampedMedian,
    };
}

function prepareRobustStats(features, rawFeatureNames, derivedFeatureFuncs, lower = 5, upper = 95) {
    const robustStats = {};

    rawFeatureNames.forEach((name) => {
        const values = features
            .map((feature) => feature[name])
            .filter((value, index) => isFinite(value) && features[index]["loudness"] !== 0);
        robustStats[name] = computeRobustStats(values, lower, upper);
    });

    for (const [name, func] of Object.entries(derivedFeatureFuncs)) {
        const values = features
            .map(func)
            .filter((value, index) => isFinite(value) && features[index]["loudness"] !== 0);
        robustStats[name] = computeRobustStats(values, lower, upper);
    }

    return robustStats;
}

function computeFeatureBounds(features, rawFeatureNames, derivedFeatureFuncs) {
    const maxFeatureValues = Object.fromEntries(rawFeatureNames.map((key) => [key, Number.MIN_SAFE_INTEGER]));
    const minFeatureValues = Object.fromEntries(rawFeatureNames.map((key) => [key, Number.MAX_SAFE_INTEGER]));

    for (const feature of features) {
        for (const [name, func] of Object.entries(derivedFeatureFuncs)) {
            feature[name] = func(feature);
        }

        if (feature["loudness"] === 0) {
            continue;
        }

        rawFeatureNames.forEach((name) => {
            if (name === "yin_periodicity" || name === "loudness_periodicity") {
                return;
            }

            if (feature[name] < minFeatureValues[name]) {
                minFeatureValues[name] = feature[name];
            }
            if (feature[name] > maxFeatureValues[name]) {
                maxFeatureValues[name] = feature[name];
            }
        });
    }

    minFeatureValues["yin_periodicity"] = 0;
    maxFeatureValues["yin_periodicity"] = 1;
    minFeatureValues["loudness_periodicity"] = 0;
    maxFeatureValues["loudness_periodicity"] = maxFeatureValues["loudness"];

    return { minFeatureValues, maxFeatureValues };
}

function buildFeatureConfig(rawFeatureNames, minFeatureValues, maxFeatureValues, robustStats, isGlobalClampEnabled, clampConfig) {
    const featureConfig = {};

    rawFeatureNames.forEach((name) => {
        featureConfig[name] = {
            val: (feature) => feature[name],
            min: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].min : minFeatureValues[name],
            max: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].max : maxFeatureValues[name],
            median: (isGlobalClampEnabled && clampConfig[name]) ? robustStats[name].clamped_median : robustStats[name].median,
        };
    });

    return featureConfig;
}

export function buildFeatureConfigurations(audioFiles, options) {
    const configurations = [];
    let minValue = Number.MAX_SAFE_INTEGER;
    let maxValue = Number.MIN_SAFE_INTEGER;
    let yAxisFormatter;

    audioFiles.forEach((audioData) => {
        const robustStats = prepareRobustStats(
            audioData.features,
            options.rawFeatureNames,
            options.derivedFeatureFuncs,
            options.lowerClampBound,
            options.upperClampBound,
        );
        const { minFeatureValues, maxFeatureValues } = computeFeatureBounds(
            audioData.features,
            options.rawFeatureNames,
            options.derivedFeatureFuncs,
        );

        console.log("maxFeatureValues:", maxFeatureValues);
        console.log("minFeatureValues:", minFeatureValues);

        const featureConfig = buildFeatureConfig(
            options.rawFeatureNames,
            minFeatureValues,
            maxFeatureValues,
            robustStats,
            options.isGlobalClampEnabled,
            options.clampConfig,
        );

        const scaleInfo = getYAxisScaleFromConfig(featureConfig, options.selectedFeature5, options.scale);
        if (scaleInfo) {
            const { min, max, labelFormatter } = scaleInfo;
            if (min < minValue) {
                minValue = min;
            }
            if (max > maxValue) {
                maxValue = max;
            }
            yAxisFormatter = labelFormatter;
        } else {
            console.warn(`Feature ${options.selectedFeature5} not found in config`);
        }

        configurations.push(featureConfig);
    });

    return { configurations, minValue, maxValue, yAxisFormatter };
}

export function buildVisualFrame(feature, featureConfig, options) {
    const xAxis = map(feature["timestamp"], 0, options.maxDuration, 0, options.canvasWidth);
    const yAxis = computeYAxisValue(
        feature,
        options.selectedFeature5,
        featureConfig,
        options.canvasHeight,
        options.padding,
        options.isInverted_y_axis,
        options.scale,
        options.minValue,
        options.maxValue,
        options.isSoftclipEnabled,
        options.softclipScale,
    );
    const lineLength = getMappedFeatureValue(
        feature,
        options.selectedFeature1,
        featureConfig,
        options.startRange_lineLength,
        options.endRange_lineLength,
        options.isSoftclipEnabled,
        options.softclipScale,
        1,
    );
    const lineWidth = getMappedFeatureValue(
        feature,
        options.selectedFeature2,
        featureConfig,
        options.startRange_lineWidth,
        options.endRange_lineWidth,
        options.isSoftclipEnabled,
        options.softclipScale,
        1,
    );
    const colorSaturation = Math.floor(getMappedFeatureValue(
        feature,
        options.selectedFeature3,
        featureConfig,
        options.startRange_colorSaturation,
        options.endRange_colorSaturation,
        options.isSoftclipEnabled,
        options.softclipScale,
        0,
    ));
    const colorLightness = Math.floor(getMappedFeatureValue(
        feature,
        options.selectedFeature6,
        featureConfig,
        options.startRange_colorLightness,
        options.endRange_colorLightness,
        options.isSoftclipEnabled,
        options.softclipScale,
        50,
    ));

    let angleRange = getMappedFeatureValue(
        feature,
        options.selectedFeature4,
        featureConfig,
        options.startRange_angle,
        options.endRange_angle,
        options.isSoftclipEnabled,
        options.softclipScale,
        0,
    );
    const angleLowerBound = options.isInverted_angle ? options.endRange_angle : options.startRange_angle;
    if (angleRange >= angleLowerBound) {
        angleRange = map(rng(), 0, 1, angleLowerBound, angleRange);
    }
    const angleDegrees = 90 + getRandomSign() * angleRange;
    const angle = (angleDegrees * Math.PI) / 180;

    const dashArray = getMappedFeatureValue(
        feature,
        options.selectedFeature7,
        featureConfig,
        options.startRange_dashArray,
        options.endRange_dashArray,
        options.isSoftclipEnabled,
        options.softclipScale,
        0,
    );

    const maxLoudness = featureConfig["loudness"].max;
    const normalized_loudness = clamp(feature["loudness"], 0, maxLoudness) / maxLoudness;

    return {
        xAxis,
        yAxis,
        lineLength,
        lineWidth,
        colorSaturation,
        colorLightness,
        angle,
        angleDegrees,
        dashArray,
        normalized_loudness,
    };
}

export function buildFeatureDescription(feature, featureConfig, options) {
    let description = `Timestamp: ${feature["timestamp"].toFixed(2)}s<br>`;

    options.rawFeatureNames.forEach((name, index) => {
        const entry = featureConfig[name];
        const minValue = entry.min;
        const maxValue = entry.max;
        let value = clamp(entry.val(feature), minValue, maxValue);

        if (options.isSoftclipEnabled && options.clampConfig[name]) {
            const shift = (entry.median - minValue) / (maxValue - minValue);
            value = mapWithSoftClipping(value, minValue, maxValue, minValue, maxValue, shift, options.softclipScale);
        }

        description += `${options.visibleFeatureNames[index]}: ${value.toFixed(2)}<br>`;
    });

    return description;
}
