export function buildDrawingFileContext(fileIndex, {
    featureConfig,
    isObjectifyEnabled,
    pathData,
}) {
    let previousDots = [];
    pathData[fileIndex] = [];

    //! Added this for "deterministic randomness"
    rng = random_engine(0); // rng() is a global function (dont add 'const' or anything)

    //! Added this
    const mixerToggle = document.getElementById(`toggle-${fileIndex}`);
    const isHidden = mixerToggle ? !mixerToggle.isActive : false;
    const pathGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    pathGroup.id = `audio-path-${fileIndex}`; // Add unique id
    pathGroup.style.display = isHidden ? "none" : "inline";
    // pathGroup.classList.add(`audio-path-${fileIndex}`); // Add unique class

    //let colorHue = hexToHSL(huePicker.value); // Convert hex to HSL
    // const colorHue = getFileBaseHue(fileIndex);
    const colorHue = Number(document.getElementById(`color-slider-${fileIndex}`).value);
    //! Added these
    const hue1 = ((colorHue - 30) + 360) % 360; // analogous color
    const hue2 = ((colorHue + 30) + 360) % 360; // analogous color
    const loudness_threshold = isObjectifyEnabled ? 0 : document.getElementById("slider-gate").noUiSlider.get()/100;

    return {
        previousDots,
        pathGroup,
        featureConfig,
        colorHue,
        hue1,
        hue2,
        loudness_threshold,
    };
}

export function buildVisualFrameOptions({
    maxDuration,
    canvasWidth,
    canvasHeight,
    padding,
    minValue,
    maxValue,
    selectedFeature1,
    selectedFeature2,
    selectedFeature3,
    selectedFeature4,
    selectedFeature5,
    selectedFeature6,
    selectedFeature7,
    isInverted_y_axis,
    isInverted_angle,
    scale,
    isSoftclipEnabled,
    softclipScale,
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
}) {
    return {
        maxDuration,
        canvasWidth,
        canvasHeight,
        padding,
        minValue,
        maxValue,
        selectedFeature1,
        selectedFeature2,
        selectedFeature3,
        selectedFeature4,
        selectedFeature5,
        selectedFeature6,
        selectedFeature7,
        isInverted_y_axis,
        isInverted_angle,
        scale,
        isSoftclipEnabled,
        softclipScale,
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
