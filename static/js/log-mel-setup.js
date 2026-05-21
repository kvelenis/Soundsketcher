(function () {
    window.SoundSketcherApp.onReady("log mel controls", () => {
        const logLinearCheckbox = document.getElementById("log-linear");
        const melCheckbox = document.getElementById("mel-scale");
        const featureSelect = document.getElementById("featureSelect-5");

        if (!logLinearCheckbox || !melCheckbox || !featureSelect) return;

        let logState = logLinearCheckbox.checked;
        let melState = melCheckbox.checked;

        window.updateLogLinearCheckbox = function () {
            if (allowedLogFeatures.includes(featureSelect.value)) {
                logLinearCheckbox.disabled = false;
                logLinearCheckbox.checked = logState;
                melCheckbox.disabled = false;
                melCheckbox.checked = melState;
                return;
            }

            if (!logLinearCheckbox.disabled) {
                logState = logLinearCheckbox.checked;
            }
            if (!melCheckbox.disabled) {
                melState = melCheckbox.checked;
            }

            logLinearCheckbox.disabled = true;
            logLinearCheckbox.checked = false;
            melCheckbox.disabled = true;
            melCheckbox.checked = false;
        };

        featureSelect.addEventListener("change", window.updateLogLinearCheckbox);

        logLinearCheckbox.addEventListener("click", () => {
            logState = logLinearCheckbox.checked;
            if (logState) {
                melCheckbox.checked = false;
                melState = melCheckbox.checked;
            }
        });

        melCheckbox.addEventListener("click", () => {
            melState = melCheckbox.checked;
            if (melState) {
                logLinearCheckbox.checked = false;
                logState = logLinearCheckbox.checked;
            }
        });

        window.updateLogLinearCheckbox();
    });
})();
