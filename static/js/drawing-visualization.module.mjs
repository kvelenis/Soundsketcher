import {
    buildDrawingRunContext,
    completeDrawingRunFromContext,
    processDrawingAudioFile,
} from "./drawing-context.module.mjs?v=frontend-migration-122";

export function drawVisualization() {
    const profiler = window.createDrawingProfiler("drawVisualization");
    const controlState = window.readDrawingControlState();
    const audioFiles = window.globalAudioData?.data || [];

    console.log("DRAW division_value =", controlState.division_value);

    //! Tried to move filtering to frontend, didn't get good results
    // const applyFiltering = document.getElementById("filter_button").checked;
    // const overlap = percentages[document.getElementById("window_overlap_selector").value];
    // const kernelSize = calculate_optimal_length(overlap);

    const runContext = buildDrawingRunContext({
        profiler,
        controlState,
        audioFiles,
    });

    audioFiles.forEach((audioData, fileIndex) => {
        processDrawingAudioFile({
            audioData,
            fileIndex,
            profiler,
            ...runContext,
        });
    });

    completeDrawingRunFromContext(profiler, runContext);
}
