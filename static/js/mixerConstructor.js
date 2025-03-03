function createMixerUI() {
    const mixerContainer = document.getElementById("mixer");
    mixerContainer.innerHTML = ""; // Clear existing mixer controls

    audioPlayers.forEach((audio, index) => {
        const mixerItem = document.createElement("div");
        mixerItem.classList.add("mixer-item");

        const label = document.createElement("label");
        label.textContent = `Audio ${index + 1}`;

        const volumeControl = document.createElement("input");
        volumeControl.type = "range";
        volumeControl.min = 0;
        volumeControl.max = 1;
        volumeControl.step = 0.01;
        volumeControl.value = audio.volume;

        volumeControl.addEventListener("input", (event) => {
            audio.volume = parseFloat(event.target.value);
        });

        // Create toggle button for visibility with eye icon
        const toggleButton = document.createElement("button");
        toggleButton.classList.add("toggle-button");

        const eyeIcon = document.createElement("img");
        eyeIcon.src = "/static/assets/eye_open.png"; // Initial image (eye open)
        eyeIcon.alt = "Toggle visibility";
        eyeIcon.style.width = "20px"; // Set icon size
        eyeIcon.style.height = "20px";

        toggleButton.appendChild(eyeIcon);

        // Add toggle functionality
        toggleButton.addEventListener("click", () => {
            const paths = document.querySelectorAll(`.audio-path-${index}`);
            const isHidden = paths[0]?.style.display === "none"; // Check current visibility

            paths.forEach(path => {
                path.style.display = isHidden ? "inline" : "none"; // Toggle visibility
            });

            // Update the eye icon
            eyeIcon.src = isHidden ? "/static/assets/eye_open.png" : "/static/assets/eye_close.png";
        });

        mixerItem.appendChild(label);
        mixerItem.appendChild(volumeControl);
        mixerItem.appendChild(toggleButton);

        mixerContainer.appendChild(mixerItem);
    });
}