const savedHues = [];

function getMixerState() {
    return window.SoundSketcher.state;
}

function getPlayback() {
    return window.SoundSketcher.playback;
}

function getMixerLabel(index) {
    const filename = getMixerState().globalFile?.[index]?.name || `Audio ${index + 1}`;
    const lastDotIndex = filename.lastIndexOf(".");

    if (lastDotIndex === -1) {
        return filename;
    }

    const basename = filename.slice(0, lastDotIndex);
    const extension = filename.slice(lastDotIndex + 1);
    return extension === "webm" ? "Recording" : basename;
}

function createVisibilityToggle(index) {
    const toggleButton = document.createElement("button");
    toggleButton.classList.add("toggle-button");
    toggleButton.id = `toggle-${index}`;
    toggleButton.type = "button";
    toggleButton.isActive = true;

    const eyeIcon = document.createElement("img");
    eyeIcon.src = window.SoundSketcher.url("/sandbox-static/assets/eye_open.png");
    eyeIcon.alt = "Toggle visibility";
    eyeIcon.style.width = "20px";
    eyeIcon.style.height = "20px";

    toggleButton.appendChild(eyeIcon);

    toggleButton.addEventListener("click", () => {
        toggleButton.isActive = !toggleButton.isActive;
        const isHidden = !toggleButton.isActive;
        const path = document.getElementById(`audio-path-${index}`);

        if (path) {
            path.style.display = isHidden ? "none" : "inline";
        }

        eyeIcon.src = window.SoundSketcher.url(isHidden ? "/sandbox-static/assets/eye_close.png" : "/sandbox-static/assets/eye_open.png");
    });

    return toggleButton;
}

function getFileBaseHue(fileIndex) {
    const fileBaseHues = [0, 120, 240, 300, 180, 60];
    return fileBaseHues[fileIndex % fileBaseHues.length];
}

function createColorPicker(index = 0) {
    let hue = savedHues.length > index ? savedHues[index] : getFileBaseHue(index);

    const picker = document.createElement("div");
    picker.className = "color-picker";

    const button = document.createElement("button");
    button.className = "color-button";
    button.type = "button";
    picker.appendChild(button);

    const popup = document.createElement("div");
    popup.className = "popup";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = 360;
    slider.value = hue;
    slider.id = `color-slider-${index}`;
    slider.className = "color-slider";

    popup.appendChild(slider);
    document.body.appendChild(popup);

    function updateColor() {
        const color = `hsl(${hue},100%,50%)`;
        button.style.backgroundColor = color;
        slider.style.color = color;
    }

    updateColor();

    button.addEventListener("click", () => {
        document.querySelectorAll(".popup").forEach((popupElement) => {
            if (popupElement !== popup) {
                popupElement.style.display = "none";
            }
        });

        const rect = button.getBoundingClientRect();
        popup.style.left = `${rect.left + window.scrollX + rect.width / 2 - 35 / 2}px`;
        popup.style.top = `${rect.top + window.scrollY - 200}px`;
        popup.style.display = popup.style.display === "block" ? "none" : "block";
    });

    slider.addEventListener("input", (event) => {
        hue = parseFloat(event.target.value);
        savedHues[index] = hue;
        updateColor();
    });

    return picker;
}

export function createMixerUI() {
    const mixerContainer = document.getElementById("mixer");
    const audioPlayers = getPlayback().getAudioPlayers();

    if (!mixerContainer || !Array.isArray(audioPlayers)) {
        return;
    }

    mixerContainer.innerHTML = "";
    document.querySelectorAll(".color-picker,.popup").forEach((element) => element.remove());

    audioPlayers.forEach((audio, index) => {
        const mixerItem = document.createElement("div");
        mixerItem.classList.add("mixer-item");

        const label = document.createElement("label");
        label.textContent = getMixerLabel(index);

        const volumeControl = document.createElement("input");
        volumeControl.id = `volume-slider-${index}`;
        volumeControl.className = "volume-slider";
        volumeControl.type = "range";
        volumeControl.min = -30;
        volumeControl.max = 0;
        volumeControl.step = 0.5;
        volumeControl.value = 0;
        volumeControl.addEventListener("input", (event) => {
            audio.volume = dBToGain(parseFloat(event.target.value), parseFloat(volumeControl.min));
        });

        mixerItem.appendChild(label);
        mixerItem.appendChild(volumeControl);
        mixerItem.appendChild(createVisibilityToggle(index));
        mixerItem.appendChild(createColorPicker(index));
        mixerContainer.appendChild(mixerItem);
    });
}

document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".color-button") && !event.target.closest(".popup")) {
        document.querySelectorAll(".popup").forEach((popup) => {
            popup.style.display = "none";
        });
    }
});

export const mixer = {
    createMixerUI,
};

window.SoundSketcher.mixer = mixer;
