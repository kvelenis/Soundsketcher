(function () {
    const rangeSliderConfigs = [
        { id: "slider-1", min: 0, max: 100, start: [0, 50], step: 1, label: "Slider-1" },
        { id: "slider-2", min: 0, max: 15, start: [2, 4], step: 1, label: "Slider-2" },
        { id: "slider-3", min: 0, max: 100, start: [0, 100], step: 1, label: "Slider-3" },
        { id: "slider-4", min: 0, max: 45, start: [2, 25], step: 1, label: "Slider-4" },
        { id: "slider-5", min: 0, max: 800, start: [0, 800], step: 1, label: "Slider-5", hidden: true },
        { id: "slider-6", min: 0, max: 100, start: [40, 100], step: 1, label: "Slider-6" },
        { id: "slider-7", min: 0, max: 10, start: [2, 6], step: 1, label: "Slider-7" },
    ];

    const singleSliderConfigs = [
        { id: "slider-gate", start: 5, min: 0, max: 20, step: 0.1, label: "gate scale" },
        { id: "period-slider", start: 0.75, min: 0, max: 1, step: 0.01, label: "periodicity scale" },
        { id: "gamma-slider", start: 0.14, min: 0, max: 1.5, step: 0.01, label: "gamma scale" },
        { id: "scdivision-slider", start: 0.6, min: 0, max: 1, step: 0.01, label: "division scale" },
        { id: "slider-softclip", start: 5, min: 0, max: 10, step: 0.1, label: "Softclip scale" },
    ];

    function getSliderElement(id, missingMessage) {
        const slider = document.getElementById(id);

        if (!slider) {
            console.error(missingMessage);
            return null;
        }

        if (!window.noUiSlider) {
            console.error("noUiSlider library is not loaded");
            return null;
        }

        return slider;
    }

    function createRangeSlider(config) {
        const slider = getSliderElement(config.id, `Elements for ${config.id} not found`);

        if (!slider) {
            return;
        }

        if (config.hidden) {
            slider.style.display = "none";
        }

        if (slider.noUiSlider) {
            return;
        }

        noUiSlider.create(slider, {
            start: config.start,
            connect: true,
            range: {
                min: config.min,
                max: config.max,
            },
            step: config.step,
            tooltips: true,
        });

        slider.noUiSlider.on("set", function (values) {
            const selectedMin = parseFloat(values[0]);
            const selectedMax = parseFloat(values[1]);
            console.log(`${config.label} selected range:`, selectedMin, selectedMax);
        });
    }

    function createSingleSlider(config) {
        const slider = getSliderElement(config.id, `${config.label} slider element not found`);

        if (!slider || slider.noUiSlider) {
            return;
        }

        noUiSlider.create(slider, {
            start: config.start,
            connect: [true, false],
            range: {
                min: config.min,
                max: config.max,
            },
            step: config.step,
            tooltips: true,
        });

        slider.noUiSlider.on("set", function (values) {
            console.log(`${config.label} set to:`, parseFloat(values[0]));
        });
    }

    function initializeSliders() {
        rangeSliderConfigs.forEach(createRangeSlider);
        singleSliderConfigs.forEach(createSingleSlider);

        createRangeSlider({
            id: "slider-clamp",
            min: 0,
            max: 100,
            start: [5, 95],
            step: 1,
            label: "Slider-clamp",
        });
    }

    window.SoundSketcherApp.onReady("sliders", initializeSliders);
})();
