import {
    applyVisualFrameToFeature,
    buildDerivedFeatureFuncsForDrawing,
    buildDrawingFileContext,
    buildDrawingRunContext,
    buildVisualFrameOptions,
    completeDrawingRun,
    completeDrawingRunFromContext,
    finalizeDrawingFile,
    processDrawingAudioFile,
    processDrawingFeatureFrame,
    readDrawingCanvasContext,
    renderEligibleVisualFrame,
} from "./drawing-context.module.mjs?v=frontend-migration-122";

const drawingContext = {
    applyVisualFrameToFeature,
    buildDerivedFeatureFuncsForDrawing,
    buildDrawingFileContext,
    buildDrawingRunContext,
    buildVisualFrameOptions,
    completeDrawingRun,
    completeDrawingRunFromContext,
    finalizeDrawingFile,
    processDrawingAudioFile,
    processDrawingFeatureFrame,
    readDrawingCanvasContext,
    renderEligibleVisualFrame,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingContext = drawingContext;
Object.assign(window, drawingContext);
