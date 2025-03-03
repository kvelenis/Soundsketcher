document.addEventListener("DOMContentLoaded", function () {
    // Add event listener to the reset button
    const resetButton = document.getElementById("resetButton");

    // Define the reset function
    const resetAllFeatures = () => {
        // Reset all select elements with class "featureSelect" to "none"
        const selectElements = document.querySelectorAll(".featureSelect");
        selectElements.forEach(select => {
            select.value = "none";
        });

        // Uncheck all checkboxes with class "invertMappingCheckbox"
        const checkboxes = document.querySelectorAll(".invertMappingCheckbox");
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        // Reset all sliders with class "range-slider" to their initial min and max values
        const sliders = document.querySelectorAll(".range-slider");
        sliders.forEach(slider => {
            const sliderInstance = slider.noUiSlider;
            if (sliderInstance) {
                const range = sliderInstance.options.range;
                sliderInstance.set([range.min, range.max]); // Reset to min and max values
            }
        });

        // Optional: Provide feedback
        console.log("All select elements reset to 'none', checkboxes unchecked, and sliders reset.");
    };

    // Add click event listener to the reset button
    resetButton.addEventListener("click", resetAllFeatures);

    // Add hotkey "x" to trigger reset functionality
    document.addEventListener("keydown", function (event) {
        if (event.key.toLowerCase() === "d") {
            resetAllFeatures();
        }
    });
});