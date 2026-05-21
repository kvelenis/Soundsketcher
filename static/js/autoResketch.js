(function () {
    function getAutoResketchState() {
        return window.SoundSketcher.state;
    }

    function bindClick(id, handler) {
        const element = document.getElementById(id);
        if (element) element.addEventListener("click", handler);
    }

    function bindWeightedPitchSlider(id, handler) {
        const slider = document.getElementById(id);
        if (!slider) return;

        slider.addEventListener("click", () => {
            const featureSelectors = document.querySelectorAll(".featureSelect");
            for (const selector of featureSelectors) {
                if (selector.value === "perceived_pitch_f0_or_SC_weighted") {
                    handler();
                    break;
                }
            }
        });
    }

    window.SoundSketcherApp.onReady("auto resketch", () => {
        const resketchButton = document.getElementById("submitButton");
        const autoResketchButton = document.getElementById("auto_resketch_button");
        if (!resketchButton || !autoResketchButton) return;

        function autoResketchHandler() {
            const state = getAutoResketchState();
            if (autoResketchButton.checked && state.globalAudioData && state.globalFile) {
                resketchButton.click();
            }
        }

        function sliderHandler(sliderElement) {
            const toggleId = sliderElement.dataset.toggle;
            if (toggleId) {
                const toggleButton = document.querySelector(toggleId);
                if (toggleButton && !toggleButton.checked) return;
            }
            autoResketchHandler();
        }

        autoResketchButton.addEventListener("click", autoResketchHandler);

        document.addEventListener("change", (event) => {
            if (event.target.classList.contains("color-slider")) {
                autoResketchHandler();
            }
        });

        document.querySelectorAll(".featureSelect").forEach((selector) => {
            selector.addEventListener("change", autoResketchHandler);
        });

        document.querySelectorAll(".invertMappingCheckbox").forEach((checkbox) => {
            checkbox.addEventListener("click", autoResketchHandler);
        });

        document.querySelectorAll(
            ".range-slider:not(#period-slider):not(#gamma-slider):not(#osc-slider):not(#grain-slider)"
        ).forEach((slider) => {
            if (slider.noUiSlider) slider.noUiSlider.on("set", () => sliderHandler(slider));
        });

        bindClick("randomizeFeaturesButton", autoResketchHandler);
        bindClick("resetButton", autoResketchHandler);

        document.querySelectorAll(".preset-button").forEach((presetButton) => {
            presetButton.addEventListener("click", autoResketchHandler);
        });

        document.querySelectorAll(".global-settings__checkbox:not(#auto_resketch_button):not(#filter_button)").forEach((globalSetting) => {
            globalSetting.addEventListener("click", autoResketchHandler);
        });

        const clamp = document.getElementById("toggleClamp");
        const softclip = document.getElementById("toggleSoftclip");
        bindClick("closeClampFeatureModal", () => {
            if (clamp?.checked || softclip?.checked) {
                autoResketchHandler();
            }
        });

        document.addEventListener("click", (event) => {
            const modal = document.getElementById("clampFeatureModal");
            if (event.target === modal && (clamp?.checked || softclip?.checked)) {
                autoResketchHandler();
            }
        });

        document.addEventListener("keydown", (event) => {
            if (["s", "d", "z", "x", "c"].includes(event.key.toLowerCase())) {
                autoResketchHandler();
            }
        });

        bindWeightedPitchSlider("period-slider", autoResketchHandler);
        bindWeightedPitchSlider("gamma-slider", autoResketchHandler);
    });
})();
