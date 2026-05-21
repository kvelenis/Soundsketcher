(function () {
    const RANDOMIZABLE_SLIDER_SELECTOR = [
        ".range-slider",
        ":not(#slider-clamp)",
        ":not(#slider-softclip)",
        ":not(#slider-gate)",
        ":not(#period-slider)",
        ":not(#gamma-slider)",
        ":not(#osc-slider)",
        ":not(#grain-slider)",
    ].join("");

    function syncMenuButton(bottomMenu, toggleButton) {
        toggleButton.innerHTML = bottomMenu.classList.contains("open") ? "&#x25BC;" : "&#x25B2;";
    }

    function toggleBottomMenu(bottomMenu, toggleButton) {
        bottomMenu.classList.toggle("open");
        syncMenuButton(bottomMenu, toggleButton);
    }

    function bindBottomMenuToggle() {
        const bottomMenu = document.getElementById("bottomMenu");
        const toggleButton = document.getElementById("toggleMenuButton");

        if (!bottomMenu || !toggleButton) {
            console.error("bottom menu controls not found");
            return;
        }

        toggleButton.addEventListener("click", () => toggleBottomMenu(bottomMenu, toggleButton));

        document.addEventListener("keydown", (event) => {
            if (event.key.toLowerCase() === "f") {
                toggleBottomMenu(bottomMenu, toggleButton);
            }
        });
    }

    function bindResketchHotkey() {
        const resketchButton = document.getElementById("submitButton");

        if (!resketchButton) {
            console.error("resketch button element not found");
            return;
        }

        resketchButton.addEventListener("click", () => {
            console.log("Resketch button clicked!");
        });

        document.addEventListener("keydown", (event) => {
            if (event.key.toLowerCase() === "a" && !event.target.matches("input, textarea")) {
                event.preventDefault();
                resketchButton.click();
            }
        });
    }

    function randomizeFeatureSelects() {
        document.querySelectorAll(".featureSelect").forEach((select) => {
            const options = Array.from(select.options);
            const randomOption = options[Math.floor(Math.random() * options.length)];

            if (randomOption) {
                select.value = randomOption.value;
            }
        });

        if (typeof window.updateLogLinearCheckbox === "function") {
            window.updateLogLinearCheckbox();
        }
    }

    function randomizeInvertCheckboxes() {
        document.querySelectorAll(".invertMappingCheckbox").forEach((checkbox) => {
            checkbox.checked = Math.random() > 0.5;
        });
    }

    function randomizeRangeSliders() {
        document.querySelectorAll(RANDOMIZABLE_SLIDER_SELECTOR).forEach((slider) => {
            const sliderInstance = slider.noUiSlider;

            if (!sliderInstance) {
                return;
            }

            const range = sliderInstance.options.range;
            const randomMin = Math.random() * (range.max - range.min) + range.min;
            const randomMax = Math.random() * (range.max - randomMin) + randomMin;
            sliderInstance.set([randomMin, randomMax], false);
        });
    }

    function bindRandomizeButton() {
        const randomizeButton = document.getElementById("randomizeFeaturesButton");

        if (!randomizeButton) {
            console.error("randomize features button element not found");
            return;
        }

        function randomizeFeatures() {
            randomizeFeatureSelects();
            randomizeInvertCheckboxes();
            randomizeRangeSliders();
            console.log("Features have been randomized!");
        }

        randomizeButton.addEventListener("click", randomizeFeatures);

        document.addEventListener("keydown", (event) => {
            if (event.key.toLowerCase() === "s") {
                randomizeButton.click();
            }
        });
    }

    function bindShellControls() {
        bindBottomMenuToggle();
        bindResketchHotkey();
        bindRandomizeButton();
    }

    window.SoundSketcherApp.onReady("shell controls", bindShellControls);
})();
