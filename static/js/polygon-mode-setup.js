(function () {
    const separateStates = [true, true, false, false, true, true, false];

    window.SoundSketcherApp.onReady("polygon mode controls", () => {
        const labels = {
            lineLength: document.getElementById("svg-characteristic-label-line-length"),
            lineWidth: document.getElementById("svg-characteristic-label-line-width"),
            angle: document.getElementById("svg-characteristic-label-angle"),
            dashArray: document.getElementById("svg-characteristic-label-dash-array"),
        };

        function setModeLabels(isPolygonMode) {
            if (!labels.lineLength || !labels.lineWidth || !labels.angle || !labels.dashArray) return;

            labels.lineLength.innerText = isPolygonMode ? "Radius" : "Line Length";
            labels.lineWidth.innerText = isPolygonMode ? "Corners" : "Line Width";
            labels.angle.innerText = isPolygonMode ? "Skew" : "Angle";
            labels.dashArray.innerText = isPolygonMode ? "Texture" : "Dash Array";
        }

        function swapFeatureStates() {
            document.querySelectorAll(".featureSelect").forEach((element, index) => {
                if (!separateStates[index]) return;

                const currentState = element.value;
                element.value = features_state[index];
                features_state[index] = currentState;
            });
        }

        function swapInvertStates() {
            document.querySelectorAll(".invertMappingCheckbox").forEach((element, index) => {
                if (!separateStates[index]) return;

                const currentState = element.checked;
                element.checked = inverted_state[index];
                inverted_state[index] = currentState;
            });
        }

        function swapSliderStates() {
            const sliderElements = document.querySelectorAll(
                ".range-slider:not(#slider-clamp):not(#slider-softclip):not(#slider-gate):not(#period-slider):not(#gamma-slider):not(#osc-slider):not(#grain-slider)"
            );

            sliderElements.forEach((element, index) => {
                if (!separateStates[index] || !element?.noUiSlider || element.style.display === "none") return;

                const nextState = sliders_state[index];
                if (!nextState) return;

                const currentRange = element.noUiSlider.options.range;
                const currentStart = element.noUiSlider.get(true);
                element.noUiSlider.updateOptions(
                    {
                        range: {
                            min: nextState.min,
                            max: nextState.max,
                        },
                        start: [nextState.startMin, nextState.startMax],
                    },
                    false
                );

                sliders_state[index] = {
                    min: currentRange.min,
                    max: currentRange.max,
                    startMin: currentStart[0],
                    startMax: currentStart[1],
                };
            });
        }

        if (!linePolygonMode || !objectifierMode || typeof window.handleModeChange !== "function") return;

        linePolygonMode.addEventListener("change", () => {
            window.handleModeChange(linePolygonMode, objectifierMode);
            setModeLabels(linePolygonMode.checked);
            swapFeatureStates();
            swapInvertStates();
            swapSliderStates();
        });
    });
})();
