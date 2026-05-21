let clampConfig = {};

function updateClampConfig() {
    const clampSelect = document.getElementById("clamp-feature-select");
    const selected = Array.from(clampSelect.selectedOptions).map((option) => option.value);

    rawFeatureNames.forEach((name) => {
        clampConfig[name] = selected.includes(name);
    });
}

function bindDrawingControls() {
    const clampSelect = document.getElementById("clamp-feature-select");
    if (!clampSelect) {
        return;
    }

    clampSelect.addEventListener("change", updateClampConfig);
    updateClampConfig();
}

function readDrawingControlState() {
    const threshold_value = parseFloat(document.getElementById("period-slider").noUiSlider.get());
    const gamma_value = parseFloat(document.getElementById("gamma-slider").noUiSlider.get());
    const division_value = parseFloat(document.getElementById("scdivision-slider").noUiSlider.get());
    const periodicity_selector = globalAudioData.data[0].features[0].raw_periodicity !== undefined ? "raw_periodicity" : "yin_periodicity";

    const isGlobalClampEnabled = document.getElementById("toggleClamp").checked;
    const [lowerClampBound, upperClampBound] = document.getElementById("slider-clamp").noUiSlider.get().map(parseFloat);

    const isSoftclipEnabled = document.getElementById("toggleSoftclip").checked;
    const softclipScale = parseFloat(document.getElementById("slider-softclip").noUiSlider.get());

    const isLogSelected = document.getElementById("log-linear").checked;
    const isMelSelected = document.getElementById("mel-scale").checked;
    const scale = isMelSelected ? "mel" : isLogSelected ? "log" : "linear";

    const isInverted_y_axis = document.getElementById("invertMapping-5")?.checked;

    const isInverted_lineLength = document.getElementById("invertMapping-1")?.checked;
    const { startRange_lineLength, endRange_lineLength } = calculateDynamicRange(
        document.getElementById("slider-1"),
        isInverted_lineLength,
        "startRange_lineLength",
        "endRange_lineLength",
    );

    const isInverted_lineWidth = document.getElementById("invertMapping-2")?.checked;
    const { startRange_lineWidth, endRange_lineWidth } = calculateDynamicRange(
        document.getElementById("slider-2"),
        isInverted_lineWidth,
        "startRange_lineWidth",
        "endRange_lineWidth",
    );

    const isInverted_colorSaturation = document.getElementById("invertMapping-3")?.checked;
    const { startRange_colorSaturation, endRange_colorSaturation } = calculateDynamicRange(
        document.getElementById("slider-3"),
        isInverted_colorSaturation,
        "startRange_colorSaturation",
        "endRange_colorSaturation",
    );

    const isInverted_colorLightness = document.getElementById("invertMapping-6")?.checked;
    const { startRange_colorLightness, endRange_colorLightness } = calculateDynamicRange(
        document.getElementById("slider-6"),
        isInverted_colorLightness,
        "startRange_colorLightness",
        "endRange_colorLightness",
    );

    const isInverted_angle = document.getElementById("invertMapping-4")?.checked;
    const { startRange_angle, endRange_angle } = calculateDynamicRange(
        document.getElementById("slider-4"),
        isInverted_angle,
        "startRange_angle",
        "endRange_angle",
    );

    const isInverted_dashArray = document.getElementById("invertMapping-7")?.checked;
    const { startRange_dashArray, endRange_dashArray } = calculateDynamicRange(
        document.getElementById("slider-7"),
        isInverted_dashArray,
        "startRange_dashArray",
        "endRange_dashArray",
    );

    return {
        threshold_value,
        gamma_value,
        division_value,
        periodicity_selector,
        selectedFeature1: featureSelect1.value,
        selectedFeature2: featureSelect2.value,
        selectedFeature3: featureSelect3.value,
        selectedFeature4: featureSelect4.value,
        selectedFeature5: featureSelect5.value,
        selectedFeature6: featureSelect6.value,
        selectedFeature7: featureSelect7.value,
        isThresholdCircleEnabled: thresholdCircle.checked,
        isJoinPathsEnabled: joinDataPoints.checked,
        isLineSketchingEnabled: !document.getElementById("linePolygonMode").checked,
        isPolygonEnabled: document.getElementById("linePolygonMode").checked,
        isGlobalClampEnabled,
        lowerClampBound,
        upperClampBound,
        isSoftclipEnabled,
        softclipScale,
        scale,
        isInverted_y_axis,
        isInverted_angle,
        startRange_lineLength,
        endRange_lineLength,
        startRange_lineWidth,
        endRange_lineWidth,
        startRange_colorSaturation,
        endRange_colorSaturation,
        startRange_colorLightness,
        endRange_colorLightness,
        startRange_angle,
        endRange_angle,
        startRange_dashArray,
        endRange_dashArray,
    };
}

bindDrawingControls();
