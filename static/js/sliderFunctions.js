document.addEventListener("DOMContentLoaded", function () {
    const numberOfSliders = 7; // Total number of sliders

    // Define custom ranges for each slider
    const sliderConfigs = [
        { min: 0, max: 150 },  // Slider 1 Line Length
        { min: 1, max: 5 }, // Slider 2 Line Width
        { min: 0, max: 100 }, // Slider 3 Color Saturation
        { min: 0, max: 45 },     // Slider 4 Angle
        { min: 0, max: 800 },   // Slider 5 Y Axis
        { min: 0, max: 100 },   // Slider 6 Color Lightness
        { min: 0, max: 10 }    // Slider 7 Dash Array
    ];

    for (let i = 1; i <= numberOfSliders; i++) {
        const slider = document.getElementById(`slider-${i}`);

        if (!slider) {
            console.error(`Elements for slider-${i} not found`);
            continue;
        }

        const { min, max } = sliderConfigs[i - 1]; // Get the min and max for the current slider

        // Initialize the noUiSlider
        noUiSlider.create(slider, {
            start: [min, max], // Initial min and max values
            connect: true,
            range: {
                min: min,
                max: max
            },
            step: 1, // Adjust step size dynamically (optional)
            tooltips: true
        });

        // Use the slider values for mapping
        slider.noUiSlider.on("set", function (values) {
            const selectedMin = parseFloat(values[0]);
            const selectedMax = parseFloat(values[1]);

            // Update your mapping logic here
            console.log(`Slider-${i} selected range:`, selectedMin, selectedMax);
        });
    }
});