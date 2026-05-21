import {
    buildVisualFrame,
} from "./drawing-features.module.mjs?v=frontend-migration-117";
import {
    renderEligibleVisualFrame,
} from "./drawing-frame-renderer.module.mjs?v=frontend-migration-125";

export function applyVisualFrameToFeature(feature, visualFrame) {
    const {
        xAxis,
        yAxis,
        lineLength,
        lineWidth,
        colorSaturation,
        colorLightness,
        angle,
        dashArray,
        normalized_loudness,
    } = visualFrame;

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

    return feature["visual"];
}

export function processDrawingFeatureFrame({
    feature,
    profiler,
    featureConfig,
    visualFrameOptions,
    isObjectifyEnabled,
    pathData,
    fileIndex,
    pathGroup,
    defs,
    colorHue,
    hue1,
    hue2,
    loudness_threshold,
    previousDots,
    isThresholdCircleEnabled,
    isJoinPathsEnabled,
    isLineSketchingEnabled,
    isPolygonEnabled,
    rawFeatureNames,
    visibleFeatureNames,
    isSoftclipEnabled,
    clampConfig,
    softclipScale,
}) {
    profiler.count("feature frames");
    const visualFrame = profiler.measure("visual frame mapping", () => buildVisualFrame(feature,featureConfig,visualFrameOptions));
    applyVisualFrameToFeature(feature, visualFrame);

    if(isObjectifyEnabled)
    {
        return previousDots;
    }

    return renderEligibleVisualFrame({
        feature,
        visualFrame,
        profiler,
        pathData,
        fileIndex,
        featureConfig,
        pathGroup,
        defs,
        colorHue,
        hue1,
        hue2,
        loudness_threshold,
        previousDots,
        isThresholdCircleEnabled,
        isJoinPathsEnabled,
        isLineSketchingEnabled,
        isPolygonEnabled,
        rawFeatureNames,
        visibleFeatureNames,
        isSoftclipEnabled,
        clampConfig,
        softclipScale,
    });
}
