//! Light modifications
document.addEventListener("DOMContentLoaded", function () {

    // Define the reset function //! !!! Create different function for each preset button !!!
    const presetAllFeatures = (features_preset,inverted_preset,sliders_preset) => {
        // --- 1) Preset Audio Feature <select>s (by row order) ---
        // Order must match the .featureSelect NodeList in your table

        const mode = document.getElementById("linePolygonMode").checked ? 1 : 0;

        const featuresConfig = features_preset[mode];
        features_state = features_preset[+!mode].slice();;

        const selectElements = document.querySelectorAll(".featureSelect");
        selectElements.forEach((select, i) => {
            const wanted = featuresConfig[i] || "none";
            const exists = Array.from(select.options).some(o => o.value === wanted);
            select.value = exists ? wanted : "none";
            // fire change so any listeners update mappings immediately
            // select.dispatchEvent(new Event("change", { bubbles: true })); //! Commented out because it triggers resketch multiple times
        });
        updateLogLinearCheckbox(); //! Added this

        const invertedConfig = inverted_preset[mode];
        inverted_state = inverted_preset[+!mode].slice();;

        // Uncheck all checkboxes with class "invertMappingCheckbox"
        const checkboxes = document.querySelectorAll(".invertMappingCheckbox");
        checkboxes.forEach((checkbox,index) =>
        {
            checkbox.checked = invertedConfig[index];
        });

        const sliderConfigs = sliders_preset[mode]
        sliders_state = JSON.parse(JSON.stringify(sliders_preset[+!mode]));

        // Reset all sliders to their initial startMin and startMax values
        const sliders = document.querySelectorAll(".range-slider:not(#slider-5):not(#slider-clamp):not(#slider-softclip):not(#slider-gate):not(#tmp-slider):not(#gamma-slider)"); //! Excluded Y-axis and clip sliders
        sliders.forEach((slider, index) => {
            const sliderInstance = slider.noUiSlider;
            if (sliderInstance) {
                const {min, max, startMin, startMax } = sliderConfigs[index]; // take from config
                sliderInstance.updateOptions(
                {
                    range:
                    {
                        min: min,
                        max: max
                    },
                    start: [startMin,startMax]
                },false); //! Added false to not trigger resketch multiple times
            }
        });

        //! Added these
        // const softclip_slider = document.getElementById("slider-softclip");
        // softclip_slider.noUiSlider.reset(false);
        // const clamp_checkbox = document.getElementById("toggleClamp");
        // clamp_checkbox.checked = true;
        // const clamp_select = document.getElementById("clamp-feature-select");
        // Array.from(clamp_select.options).forEach(opt => opt.selected = true);

        // Optional: Provide feedback
        console.log("All select elements reset to 'none', checkboxes unchecked, and sliders reset.");
    };

    const presetButtonA = document.getElementById("presetButtonA");

    const features_lineA = [
            "loudness_periodicity",        // row 1
            "mir_sharpness_zwicker",          // row 2
            "yin_periodicity",            // row 3
            "loudness",                    // row 4
            "yin_periodicity",             // row 5
            "mir_roughness_vassilakis",            // row 6
            "perceived_pitch_f0_or_SC_weighted"  // row 7
    ];
    const inverted_lineA = [false,true,false,true,true,false,false];
    const sliders_lineA = [
        { min: 0, max: 100, startMin: 0, startMax: 50 },   // Slider 1 Line Length
        { min: 0, max: 15, startMin: 2, startMax: 4 },      // Slider 2 Line Width
        { min: 0, max: 100, startMin: 0, startMax: 100 },   // Slider 3 Color Saturation
        { min: 0, max: 100, startMin: 40, startMax: 90  },     // Slider 6 Color Lightness
        { min: 0, max: 45, startMin: 2, startMax: 25  },   // Slider 4 Angle
        { min: 0, max: 10, startMin: 2, startMax: 6 },      // Slider 7 Dash Array
        // { min: 0, max: 100, startMin: 40, startMax: 100 },  // Slider 6 Yaxis //! No need anymore
    ];

    const features_polygonA = [
            "loudness_periodicity",        // row 1
            "mir_sharpness_zwicker",          // row 2
            "yin_periodicity",            // row 3
            "loudness",                    // row 4
            "yin_periodicity",             // row 5
            "mir_roughness_vassilakis",            // row 6
            "perceived_pitch_f0_or_SC_weighted"  // row 7
    ];
    const inverted_polygonA = [false,true,false,true,true,false,false];
    const sliders_polygonA = [
        { min: 3, max: 20, startMin: 3, startMax: 15 },   // Slider 1 Radius
        { min: 3, max: 12, startMin: 3, startMax: 8 },      // Slider 2 Corners
        { min: 0, max: 100, startMin: 0, startMax: 100 },   // Slider 3 Color Saturation
        { min: 0, max: 100, startMin: 40, startMax: 90  },     // Slider 6 Color Lightness
        { min: 0, max: 45, startMin: 0, startMax: 30  },   // Slider 4 Skew
        { min: 0, max: 100, startMin: 20, startMax: 100 },      // Slider 7 Texture
        // { min: 0, max: 100, startMin: 40, startMax: 100 },  // Slider 6 Yaxis //! No need anymore
    ];

    const featuresA = [features_lineA,features_polygonA]
    const invertedA = [inverted_lineA,inverted_polygonA]
    const slidersA = [sliders_lineA,sliders_polygonA]

    // Add click event listener to the reset button
    presetButtonA.addEventListener("click",() => { presetAllFeatures(featuresA,invertedA,slidersA) });

    // Add hotkey "z" to trigger reset functionality
    document.addEventListener("keydown", function (event) {
        if (event.key.toLowerCase() === "z")
        {
            presetButtonA.click();         }
    });

    const presetButtonB = document.getElementById("presetButtonB");

    //! Quick  preset for testing , change it when all features are functional
    const features_lineB = [
            "spectral_centroid",        // row 1
            "spectral_centroid",          // row 2
            "spectral_centroid",            // row 3
            "spectral_centroid",                    // row 4
            "spectral_centroid",             // row 5
            "spectral_centroid",            // row 6
            "spectral_centroid"  // row 7
    ];
    const inverted_lineB = inverted_lineA;
    const sliders_lineB = sliders_lineA;

    const features_polygonB = [
            "spectral_centroid",        // row 1
            "spectral_centroid",          // row 2
            "spectral_centroid",            // row 3
            "spectral_centroid",                    // row 4
            "spectral_centroid",             // row 5
            "spectral_centroid",            // row 6
            "spectral_centroid"  // row 7
    ];
    const inverted_polygonB = inverted_polygonA;
    const sliders_polygonB = sliders_polygonA;

    const featuresB = [features_lineB,features_polygonB]
    const invertedB = [inverted_lineB,inverted_polygonB]
    const slidersB = [sliders_lineB,sliders_polygonB]

    // Add click event listener to the reset button
    presetButtonB.addEventListener("click",() => { presetAllFeatures(featuresB,invertedB,slidersB) });

    // Add hotkey "z" to trigger reset functionality
    document.addEventListener("keydown", function (event) {
        if (event.key.toLowerCase() === "x")
        {
            presetButtonB.click();         }
    });

    const presetButtonC = document.getElementById("presetButtonC");

    //! Quick  preset for testing , change it when all features are functional
    const features_lineC = [
            "loudness",        // row 1
            "loudness",          // row 2
            "loudness",            // row 3
            "loudness",                    // row 4
            "loudness",             // row 5
            "loudness",            // row 6
            "loudness"  // row 7
    ];
    const inverted_lineC = inverted_lineA;
    const sliders_lineC = sliders_lineA;

    const features_polygonC = [
            "loudness",        // row 1
            "loudness",          // row 2
            "loudness",            // row 3
            "loudness",                    // row 4
            "loudness",             // row 5
            "loudness",            // row 6
            "loudness"  // row 7
    ];
    const inverted_polygonC = inverted_polygonA;
    const sliders_polygonC = sliders_polygonA;

    const featuresC = [features_lineC,features_polygonC]
    const invertedC = [inverted_lineC,inverted_polygonC]
    const slidersC = [sliders_lineC,sliders_polygonC]

    // Add click event listener to the reset button
    presetButtonC.addEventListener("click",() => { presetAllFeatures(featuresC,invertedC,slidersC) });

    // Add hotkey "z" to trigger reset functionality
    document.addEventListener("keydown", function (event) {
        if (event.key.toLowerCase() === "c")
        {
            presetButtonC.click();         }
    });

    presetButtonA.click();
});