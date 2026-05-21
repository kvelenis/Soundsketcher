(function () {
    const RESETTABLE_SLIDER_SELECTOR = [
        ".range-slider",
        ":not(#slider-clamp)",
        ":not(#slider-softclip)",
        ":not(#slider-gate)",
        ":not(#period-slider)",
        ":not(#gamma-slider)",
        ":not(#osc-slider)",
        ":not(#grain-slider)",
    ].join("");

    function resetFeatureSelects() {
        document.querySelectorAll(".featureSelect").forEach((select) => {
            select.value = "none";
        });

        if (typeof window.updateLogLinearCheckbox === "function") {
            window.updateLogLinearCheckbox();
        }
    }

    function resetInvertCheckboxes() {
        document.querySelectorAll(".invertMappingCheckbox").forEach((checkbox) => {
            checkbox.checked = false;
        });
    }

    function resetRangeSliders() {
        document.querySelectorAll(RESETTABLE_SLIDER_SELECTOR).forEach((slider) => {
            const sliderInstance = slider.noUiSlider;

            if (!sliderInstance) {
                return;
            }

            const range = sliderInstance.options.range;
            sliderInstance.set([range.min, range.max], false);
        });
    }

    function resetAllFeatures() {
        resetFeatureSelects();
        resetInvertCheckboxes();
        resetRangeSliders();
        console.log("All select elements reset to 'none', checkboxes unchecked, and sliders reset.");
    }

    function bindResetButton() {
        const resetButton = document.getElementById("resetButton");

        if (!resetButton) {
            console.error("reset button element not found");
            return;
        }

        resetButton.addEventListener("click", resetAllFeatures);

        document.addEventListener("keydown", function (event) {
            if (event.key.toLowerCase() === "d") {
                resetButton.click();
            }
        });
    }

    window.SoundSketcherApp.onReady("reset button", bindResetButton);
})();
