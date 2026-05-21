export function appendCompletedPathGroup(svgContainer, pathGroup) {
    svgContainer.appendChild(pathGroup);
}

export function renderObjectifierClusters({
    audioData,
    fileIndex,
    svgContainer,
    canvasWidth,
    canvasHeight,
    maxDuration,
}) {
    drawClusterOverlays(audioData.clusters,audioData.features,svgContainer,canvasWidth,canvasHeight,maxDuration,fileIndex,audioData);
}

export function finalizeDrawingFile({
    audioData,
    fileIndex,
    profiler,
    svgContainer,
    pathGroup,
    canvasWidth,
    canvasHeight,
    maxDuration,
    isObjectifyEnabled,
}) {
    if(!isObjectifyEnabled)
    {
        profiler.measure("append path group", () => {
            appendCompletedPathGroup(svgContainer, pathGroup);
        });
        return;
    }

    profiler.measure("objectifier render", () => {
        renderObjectifierClusters({
            audioData,
            fileIndex,
            svgContainer,
            canvasWidth,
            canvasHeight,
            maxDuration,
        });
    });
}
