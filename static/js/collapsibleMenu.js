

// JavaScript to toggle the menu
const bottomMenu = document.getElementById("bottomMenu");
const toggleButton = document.getElementById("toggleMenuButton");

toggleButton.addEventListener("click", () => {
    // Toggle the "open" class on the menu
    bottomMenu.classList.toggle("open");

    // Toggle the arrow direction
    if (bottomMenu.classList.contains("open")) {
        toggleButton.innerHTML = "&#x25BC;"; // Down arrow
    } else {
        toggleButton.innerHTML = "&#x25B2;"; // Up arrow
    }
});

// HOTKEY SECTION

document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "c") {
        // Toggle the "open" class on the menu
        bottomMenu.classList.toggle("open");// Toggle the arrow direction
        if (bottomMenu.classList.contains("open")) {
            toggleButton.innerHTML = "&#x25BC;"; // Down arrow
        } else {
            toggleButton.innerHTML = "&#x25B2;"; // Up arrow
        }

    }
});

document.addEventListener("DOMContentLoaded", () => {
    const resketchButton = document.getElementById("submitButton");

    // Add a click event for the button
    resketchButton.addEventListener("click", () => {
        console.log("Resketch button clicked!");
        // Add the resketch functionality here
    });

    // Add a keydown event for the hotkey
    document.addEventListener("keydown", (event) => {
        if (event.key.toLowerCase() === "s" && !event.target.matches("input, textarea")) {
            event.preventDefault(); // Prevent default browser action for the key
            resketchButton.click(); // Programmatically trigger the button click
        }
    });
});


// RANDOMISE BUTTON

document.addEventListener("DOMContentLoaded", () => {
const randomizeButton = document.getElementById("randomizeFeaturesButton");

const randomizeFeatures = () => {
// Randomize the feature selects
const selects = document.querySelectorAll(".featureSelect");
selects.forEach((select) => {
    const options = Array.from(select.options);
    const randomOption = options[Math.floor(Math.random() * options.length)];
    select.value = randomOption.value; // Set a random option
});

// Randomize the invert mapping checkboxes
const checkboxes = document.querySelectorAll(".invertMappingCheckbox");
        checkboxes.forEach((checkbox) => {
            checkbox.checked = Math.random() > 0.5; // Randomly check or uncheck
        });

        // Randomize the sliders
        const sliders = document.querySelectorAll(".range-slider");
        sliders.forEach((slider) => {
            if (slider.noUiSlider) {
                const range = slider.noUiSlider.options.range;
                const randomMin = (Math.random() * (range.max - range.min)) + range.min;
                const randomMax = (Math.random() * (range.max - randomMin)) + randomMin;
                slider.noUiSlider.set([randomMin, randomMax]); // Set random range
            }
        });

        console.log("Features have been randomized!");
    };

    // Add click event listener to the button
    randomizeButton.addEventListener("click", randomizeFeatures);

    // Add "r" hotkey to randomize features
    document.addEventListener("keydown", (event) => {
        if (event.key.toLowerCase() === "a") {
            randomizeFeatures();
        }
    });
});

// DROP FILE HANDLERS

// Get reference to the drop area
const dropArea = document.getElementById('drop_area');

// Event listener for file dragging over the drop area
function handleDragEnter(event) {
    // Prevent default behavior to allow drop
    event.preventDefault();

    // Apply solid border style when a file is being dragged over the drop area
    dropArea.style.borderStyle = 'solid';
}

// Event listener for file leaving the drop area
function handleDragLeave(event) {
    // Remove border style when a file leaves the drop area
    dropArea.style.borderStyle = '';
}