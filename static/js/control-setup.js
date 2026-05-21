// Shared control references used by the remaining legacy setup blocks.
var featureSelect1;
var featureSelect2;
var featureSelect3;
var featureSelect4;
var featureSelect5;
var featureSelect6;
var featureSelect7;
var thresholdToggle;
var thresholdSlider;
var thresholdCircle;
var joinDataPoints;
var linePolygonMode;
var objectifierMode;
var submitButton;
var clamp_feature_select;
var feature_selectors;

(function () {
    function resetFormControls() {
        document.querySelectorAll('input[type="checkbox"]').forEach((element) => {
            element.checked = element.defaultChecked;
        });

        document.querySelectorAll('input[type="range"]').forEach((element) => {
            element.value = element.defaultValue;
        });

        document.querySelectorAll("select").forEach((element) => {
            const defaultOption = element.querySelector("option[selected]");
            if (defaultOption) {
                element.value = defaultOption.value;
            } else {
                element.selectedIndex = 0;
            }
        });

        document.querySelectorAll("select[multiple]").forEach((element) => {
            element.querySelectorAll("option").forEach((option) => {
                option.selected = false;
            });
            element.querySelectorAll("option[selected]").forEach((option) => {
                option.selected = true;
            });
        });
    }

    function bindPlaybackShortcuts() {
        const playback = window.SoundSketcher.playback;
        const state = window.SoundSketcher.state;
        const playStopButtonHeader = document.getElementById("playStopButtonHeader");
        if (playStopButtonHeader) {
            playStopButtonHeader.addEventListener("click", () => {
                if (!state.synthPlaying) {
                    playback.toggle();
                }
            });
        }

        document.addEventListener("keydown", (event) => {
            if (event.code === "Space") {
                event.preventDefault();
                if (!state.synthPlaying) {
                    playback.toggle();
                }
            }
        });
    }

    function captureSharedControls() {
        featureSelect1 = document.getElementById("featureSelect-1");
        featureSelect2 = document.getElementById("featureSelect-2");
        featureSelect3 = document.getElementById("featureSelect-3");
        featureSelect4 = document.getElementById("featureSelect-4");
        featureSelect5 = document.getElementById("featureSelect-5");
        featureSelect6 = document.getElementById("featureSelect-6");
        featureSelect7 = document.getElementById("featureSelect-7");
        thresholdToggle = document.getElementById("thresholdToggle");
        thresholdSlider = document.getElementById("thresholdSlider");
        thresholdCircle = document.getElementById("thresholdCircle");
        joinDataPoints = document.getElementById("joindatapoints");
        linePolygonMode = document.getElementById("linePolygonMode");
        objectifierMode = document.getElementById("objectifierMode");
        submitButton = document.getElementById("submitButton");
        clamp_feature_select = document.getElementById("clamp-feature-select");
        feature_selectors = document.querySelectorAll(".featureSelect");
    }

    function preventTypeAhead(element) {
        if (!element) return;
        element.addEventListener("keydown", (event) => {
            if (event.key.length === 1) {
                event.preventDefault();
            }
        });
    }

    window.SoundSketcherApp = window.SoundSketcherApp || {};
    window.SoundSketcherApp.initCoreControls = function () {
        resetFormControls();
        bindPlaybackShortcuts();
        captureSharedControls();
        preventTypeAhead(clamp_feature_select);
        feature_selectors.forEach(preventTypeAhead);
    };
})();
