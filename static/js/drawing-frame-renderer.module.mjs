import {
    buildFeatureDescription,
} from "./drawing-features.module.mjs?v=frontend-migration-117";
import {
    appendLineSketchFrame,
    appendPolygonFrame,
} from "./drawing-renderers.module.mjs?v=frontend-migration-118";
import {
    appendSynthPathFrame,
} from "./sonification-bridge.module.mjs?v=frontend-migration-119";

export function renderEligibleVisualFrame({
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
}) {
    if (visualFrame.normalized_loudness <= loudness_threshold) {
        return previousDots;
    }

    profiler.measure("synth path frame", () => {
        appendSynthPathFrame(pathData,fileIndex,feature,visualFrame,colorHue);
    });

    const featureDescription = profiler.measure("feature description", () => buildFeatureDescription(feature,featureConfig,
    {
        rawFeatureNames,
        visibleFeatureNames,
        isSoftclipEnabled,
        clampConfig,
        softclipScale,
    }));

    if(isLineSketchingEnabled)
    {
        return profiler.measure("line render", () => appendLineSketchFrame(pathGroup,visualFrame,
        {
            colorHue,
            hue1,
            hue2,
            featureDescription,
            isThresholdCircleEnabled,
            isJoinPathsEnabled,
            previousDots,
        }));
    }
    //! Separated this from isLineSketchingEnabled
    else if(isPolygonEnabled)
    {
        profiler.measure("polygon render", () => {
            appendPolygonFrame(pathGroup,defs,visualFrame,
        {
            colorHue,
            featureDescription,
        });
        });
    }

    return previousDots;
}
