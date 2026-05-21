(function () {
function bindPresetButtons() {
  const presets = window.SoundSketcherPresets;
  if (!presets) {
    console.error("SoundSketcher presets were not loaded.");
    return;
  }

  function applyPreset(preset) {
    const mode = document.getElementById("linePolygonMode").checked ? 1 : 0;

    const featuresConfig = preset.features[mode];
    features_state = preset.features[+!mode].slice();

    document.querySelectorAll(".featureSelect").forEach((select, index) => {
      const wanted = featuresConfig[index] || "none";
      const exists = Array.from(select.options).some((option) => option.value === wanted);
      select.value = exists ? wanted : "none";
    });
    updateLogLinearCheckbox();

    const invertedConfig = preset.inverted[mode];
    inverted_state = preset.inverted[+!mode].slice();
    document.querySelectorAll(".invertMappingCheckbox").forEach((checkbox, index) => {
      checkbox.checked = invertedConfig[index];
    });

    const sliderConfigs = preset.sliders[mode];
    sliders_state = JSON.parse(JSON.stringify(preset.sliders[+!mode]));
    document
      .querySelectorAll(
        ".range-slider:not(#slider-5):not(#slider-clamp):not(#slider-softclip):not(#slider-gate):not(#period-slider):not(#gamma-slider):not(#scdivision-slider):not(#osc-slider):not(#grain-slider)",
      )
      .forEach((slider, index) => {
        const sliderInstance = slider.noUiSlider;
        if (!sliderInstance) return;

        const { min, max, startMin, startMax } = sliderConfigs[index];
        sliderInstance.updateOptions(
          {
            range: { min, max },
            start: [startMin, startMax],
          },
          false,
        );
      });
  }

  Object.entries(presets).forEach(([name, preset]) => {
    const button = document.getElementById(`presetButton${name}`);
    if (!button) return;

    button.addEventListener("click", () => applyPreset(preset));
    document.addEventListener("keydown", (event) => {
      if (event.key.toLowerCase() === preset.key) button.click();
    });
  });

  document.getElementById("presetButtonA")?.click();
}

window.SoundSketcherApp.onReady("preset buttons", bindPresetButtons);
})();
