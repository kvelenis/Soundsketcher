//! Light modifications
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
        updateLogLinearCheckbox(); //! Added this

        // Uncheck all checkboxes with class "invertMappingCheckbox"
        const checkboxes = document.querySelectorAll(".invertMappingCheckbox"); //! Exclude normalize button
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        // Reset all sliders with class "range-slider" to their initial min and max values
        const sliders = document.querySelectorAll(".range-slider:not(#slider-clamp):not(#slider-softclip):not(#slider-softclip)");
        sliders.forEach(slider => {
            const sliderInstance = slider.noUiSlider;
            if (sliderInstance) {
                const range = sliderInstance.options.range;
                sliderInstance.set([range.min, range.max],false); // Reset to min and max values //! Added false to not trigger resketch multiple times
            }
        });

        // Optional: Provide feedback
        console.log("All select elements reset to 'none', checkboxes unchecked, and sliders reset.");
    };

    // Add click event listener to the reset button
    resetButton.addEventListener("click", resetAllFeatures);

    // Add hotkey "d" to trigger reset functionality
    document.addEventListener("keydown", function (event) {
        if (event.key.toLowerCase() === "d")
        {
            resetButton.click(); //! Changed from calling resetAllFeatures directly
        }
    });
});