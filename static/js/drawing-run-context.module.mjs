import {
    drawYAxisScale,
} from "./drawing-axis.module.mjs?v=frontend-migration-120";
import {
    createPattern,
} from "./drawing-patterns.module.mjs?v=frontend-migration-121";
import {
    buildFeatureConfigurations,
} from "./drawing-features.module.mjs?v=frontend-migration-117";

export function buildDerivedFeatureFuncsForDrawing({
    periodicity_selector,
    threshold_value,
    gamma_value,
    division_value,
}) {
    return {
        perceived_pitch_f0_or_SC_weighted: (feature) => perceivedPitchF0OrSC(
            feature[periodicity_selector],
            feature.crepe_f0,
            feature.weighted_spectral_centroid,
            threshold_value,
            gamma_value,
            division_value,
        ),
    };
}

export function readDrawingCanvasContext(svgContainer, audioFiles) {
    return {
        canvasWidth: svgContainer.getAttribute("width"),
        canvasHeight: svgContainer.getAttribute("height"),
        padding: parseInt(svgContainer.getAttribute("padding")),
        maxDuration: Math.max(...audioFiles.map((file) => file.features[file.features.length - 1]?.timestamp || 0)),
    };
}

export function buildDrawingRunContext({
    profiler,
    controlState,
    audioFiles,
}) {
    const {
        threshold_value,
        gamma_value,
        division_value,
        periodicity_selector,
        lowerClampBound,
        upperClampBound,
        isGlobalClampEnabled,
        selectedFeature5,
        scale,
    } = controlState;

    //! Moved derived feature calculations to backend #tmpf0
    const derivedFeatureFuncs = buildDerivedFeatureFuncsForDrawing({
        periodicity_selector,
        threshold_value,
        gamma_value,
        division_value,
    });
    const svgContainer = document.getElementById("svgCanvas");
    const {
        canvasWidth,
        canvasHeight,
        padding,
        maxDuration,
    } = readDrawingCanvasContext(svgContainer, audioFiles);

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    svgContainer.appendChild(defs);

    const isObjectifyEnabled = document.getElementById("objectifierMode").checked;

    const {
        configurations,
        minValue,
        maxValue,
        yAxisFormatter,
    } = profiler.measure("feature configurations", () => buildFeatureConfigurations(audioFiles, {
        rawFeatureNames,
        derivedFeatureFuncs,
        lowerClampBound,
        upperClampBound,
        isGlobalClampEnabled,
        clampConfig,
        selectedFeature5,
        scale,
    }));

    pathData = {};

    return {
        ...controlState,
        audioFiles,
        svgContainer,
        canvasWidth,
        canvasHeight,
        padding,
        maxDuration,
        defs,
        isObjectifyEnabled,
        configurations,
        minValue,
        maxValue,
        yAxisFormatter,
        pathData,
        rawFeatureNames,
        visibleFeatureNames,
        clampConfig,
    };
}

export function completeDrawingRun({
    profiler,
    pathData,
    audioFiles,
    canvasHeight,
    minValue,
    maxValue,
    scale,
    padding,
    yAxisFormatter,
    isInverted_y_axis,
    isObjectifyEnabled,
    isPolygonEnabled,
    maxDuration,
}) {
    if(isPolygonEnabled)
    {
        createPattern.counter = 0;
    }

    const questionnaireMode = document.body?.dataset?.questionnaireMode === "true";

    //! Draw y axis on top of everything
    if(yAxisFormatter && !questionnaireMode)
    {
        profiler.measure("y-axis render", () => {
            drawYAxisScale(canvasHeight,minValue,maxValue,scale,padding,yAxisFormatter,isInverted_y_axis);
        });
    }

    profiler.measure("synth data prep", () => {
        prepareSynthData(pathData);
    });
    profiler.finish({
        files: audioFiles.length,
        mode: isObjectifyEnabled ? "objectifier" : (isPolygonEnabled ? "polygon" : "line"),
        maxDuration,
    });
}

export function completeDrawingRunFromContext(profiler, runContext) {
    completeDrawingRun({
        profiler,
        pathData: runContext.pathData,
        audioFiles: runContext.audioFiles,
        canvasHeight: runContext.canvasHeight,
        minValue: runContext.minValue,
        maxValue: runContext.maxValue,
        scale: runContext.scale,
        padding: runContext.padding,
        yAxisFormatter: runContext.yAxisFormatter,
        isInverted_y_axis: runContext.isInverted_y_axis,
        isObjectifyEnabled: runContext.isObjectifyEnabled,
        isPolygonEnabled: runContext.isPolygonEnabled,
        maxDuration: runContext.maxDuration,
    });
}
