(function () {
    window.SoundSketcherApp.onReady("mode controls", () => {
        if (!thresholdCircle || !joinDataPoints || !linePolygonMode || !objectifierMode) return;

        let circleState = thresholdCircle.checked;
        let linesState = joinDataPoints.checked;

        window.handleModeChange = function (changedCheckbox, otherCheckbox) {
            const changedState = changedCheckbox.checked;
            const otherState = otherCheckbox.checked;

            if (!thresholdCircle.disabled) circleState = thresholdCircle.checked;
            if (!joinDataPoints.disabled) linesState = joinDataPoints.checked;

            if (changedState && otherState) {
                otherCheckbox.checked = false;
                otherCheckbox.dispatchEvent(new Event("change"));
            }

            const anyModeActive = changedState || otherState;
            thresholdCircle.disabled = anyModeActive;
            joinDataPoints.disabled = anyModeActive;

            if (anyModeActive) {
                thresholdCircle.checked = false;
                joinDataPoints.checked = false;
            } else {
                thresholdCircle.checked = circleState;
                joinDataPoints.checked = linesState;
            }
        };

        objectifierMode.addEventListener("change", () => {
            window.handleModeChange(objectifierMode, linePolygonMode);
        });

        const objectifierDetailMode = document.getElementById("objectifierDetailMode");
        objectifierDetailMode?.addEventListener("change", () => {
            if (objectifierMode.checked && window.SoundSketcher?.state?.globalAudioData) {
                window.drawVisualization?.();
            }
        });
    });
})();
