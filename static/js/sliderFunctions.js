let softclipSlider;

document.addEventListener("DOMContentLoaded", function () {
    const numberOfSliders = 7; // Total number of sliders

    // Define custom ranges for each slider
    const sliderConfigs = [
        { min: 0, max: 100, startMin: 0, startMax: 50  },  // Slider 1 Line Length
        { min: 0, max: 15, startMin: 2, startMax: 4  }, // Slider 2 Line Width
        { min: 0, max: 100, startMin: 0, startMax: 100  }, // Slider 3 Color Saturation
        { min: 0, max: 45, startMin: 2, startMax: 25  },     // Slider 4 Angle
        { min: 0, max: 800, startMin: 0, startMax: 800  },   // Slider 5 Y Axis
        { min: 0, max: 100, startMin: 40, startMax: 100  },   // Slider 6 Color Lightness
        { min: 0, max: 10, startMin: 2, startMax: 6  },    // Slider 7 Dash Array
    ];

    for (let i = 1; i <= numberOfSliders; i++) {
        const slider = document.getElementById(`slider-${i}`);

        if (!slider) {
            console.error(`Elements for slider-${i} not found`);
            continue;
        }

        const { min, max, startMin, startMax } = sliderConfigs[i - 1]; // Get the min and max for the current slider

        // Initialize the noUiSlider
        noUiSlider.create(slider, {
            start: [startMin, startMax], // Initial min and max values
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

        if (i==5) {
            slider.style.display = "none";
        }
    }

    // New single-handle slider
    const gateSlider = document.getElementById("slider-gate");

    if (gateSlider) {
        noUiSlider.create(gateSlider,
        {
            start: 5,   // Good default scale
            connect: [true, false], // Only first handle connected
            range: {
                min: 0,
                max: 20
            },
            step: 0.1,
            tooltips: true
        });

        gateSlider.noUiSlider.on("set", function (values) {
            const gateScale = parseFloat(values[0]);
            console.log("gate scale set to:", gateScale);
        });
    } else {
        console.error("gate slider element not found");
    }

    const tmpSlider = document.getElementById("tmp-slider");

    if (tmpSlider) {
        noUiSlider.create(tmpSlider,
        {
            start: 0.5,   // Good default scale
            connect: [true, false], // Only first handle connected
            range: {
                min: 0,
                max: 1
            },
            step: 0.01,
            tooltips: true
        });

        tmpSlider.noUiSlider.on("set", function (values) {
            const tmpScale = parseFloat(values[0]);
            console.log("tmp scale set to:", tmpScale);
        });
    } else {
        console.error("tmp slider element not found");
    }

    const gammaSlider = document.getElementById("gamma-slider");

    if (gammaSlider) {
        noUiSlider.create(gammaSlider,
        {
            start: 1,   // Good default scale
            connect: [true, false], // Only first handle connected
            range: {
                min: 0,
                max: 1.5
            },
            step: 0.01,
            tooltips: true
        });

        gammaSlider.noUiSlider.on("set", function (values) {
            const gammaScale = parseFloat(values[0]);
            console.log("gamma scale set to:", gammaScale);
        });
    } else {
        console.error("gamma slider element not found");
    }


    // New single-handle slider for softclip scale
    const softclipSlider = document.getElementById("slider-softclip");

    if (softclipSlider) {
        noUiSlider.create(softclipSlider,
        {
            start: 5,   // Good default scale
            connect: [true, false], // Only first handle connected
            range: {
                min: 0,
                max: 10
            },
            step: 0.1,
            tooltips: true
        });

        softclipSlider.noUiSlider.on("set", function (values) {
            const softclipScale = parseFloat(values[0]);
            console.log("Softclip scale set to:", softclipScale);
            // Store or pass this value to your mapWithSoftClipping()
        });
    } else {
        console.error("Softclip slider element not found");
    }

    //! Added this
    // New single-handle slider for clamp scale
    const clampSlider = document.getElementById("slider-clamp");

    if (clampSlider) {
        noUiSlider.create(clampSlider,
        {
            start: [5,95],
            connect: true,
            range:
            {
                min: 0,
                max: 100
            },
            step: 1,
            tooltips: true
        });
    }
    else
    {
        console.error("Clamping slider element not found");
    }

});