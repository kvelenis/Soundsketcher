function createMixerUI() {
    const mixerContainer = document.getElementById("mixer");
    mixerContainer.innerHTML = ""; // Clear existing mixer controls
    document.querySelectorAll('.color-picker,.popup').forEach(el => el.remove());

    audioPlayers.forEach((audio, index) => {
        const mixerItem = document.createElement("div");
        mixerItem.classList.add("mixer-item");

        //! Added this
        let label_text;
        const filename = globalFile[index].name
        const lastDotIndex = filename.lastIndexOf(".");
        if(lastDotIndex !== -1)
        {
            const basename = filename.slice(0,lastDotIndex);
            const extension = filename.slice(lastDotIndex + 1);
            if(extension === "webm")
            {
                label_text = "Recording";
            }
            else
            {
                label_text = basename;
            }
        }
        else
        {
            label_text = filename
        }

        const label = document.createElement("label");
        label.textContent = label_text;
        // label.textContent = `Audio ${index + 1}`; //! Changed this

        const volumeControl = document.createElement("input");
        volumeControl.id = `volume-slider-${index}`;
        volumeControl.className = "volume-slider";
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
        toggleButton.id = `toggle-${index}`; //! Added this
        toggleButton.isActive = true; //! Added this

        const eyeIcon = document.createElement("img");
        eyeIcon.src = "static/assets/eye_open.png"; // Initial image (eye open)
        eyeIcon.alt = "Toggle visibility";
        eyeIcon.style.width = "20px"; // Set icon size
        eyeIcon.style.height = "20px";

        toggleButton.appendChild(eyeIcon);

        // Add toggle functionality
        //! Reworked this
        toggleButton.addEventListener("click", () =>
        {
            toggleButton.isActive = !toggleButton.isActive;
            const isHidden = !toggleButton.isActive;
            // const paths = document.querySelectorAll(`.audio-path-${index}`);
            // paths.forEach(path => {
            //     path.style.display = isHidden ? "none" : "inline"; // Toggle visibility
            // });
            const path = document.getElementById(`audio-path-${index}`);
            path.style.display = isHidden ? "none" : "inline";

            // Update the eye icon
            eyeIcon.src = isHidden ? "static/assets/eye_close.png" : "static/assets/eye_open.png";
        });
    
        mixerItem.appendChild(label);
        mixerItem.appendChild(volumeControl);
        mixerItem.appendChild(toggleButton);
        mixerItem.appendChild(createColorPicker(index));

        mixerContainer.appendChild(mixerItem);

        
    });
}

document.addEventListener('pointerdown',(event) =>
{
    if(!event.target.closest('.color-button') && !event.target.closest('.popup'))
    {
        document.querySelectorAll('.popup').forEach(p => p.style.display = 'none');
    }
});

function createColorPicker(index = 0)
{
    let initialHue = getFileBaseHue(index);

    const picker = document.createElement('div');
    picker.className = 'color-picker';

    const button = document.createElement('button');
    button.className = 'color-button';
    button.type = 'button';

    picker.appendChild(button);

    const popup = document.createElement('div');
    popup.className = 'popup';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 0;
    slider.max = 360;
    slider.value = initialHue;
    slider.id = `color-slider-${index}`;
    slider.className = "color-slider";

    popup.appendChild(slider);
    document.body.appendChild(popup);
    
    let hue = initialHue;
    function updateColor()
    {
        const color = `hsl(${hue},100%,50%)`;
        button.style.backgroundColor = color;
        slider.style.color = color;
    }
    updateColor();

    button.addEventListener('click',() =>
    {
        document.querySelectorAll('.popup').forEach
        (
            (p) =>
            {
                if(p !== popup)
                {
                    p.style.display='none';
                }
            }
        );

        const rect = button.getBoundingClientRect();
        popup.style.left = `${rect.left + window.scrollX + rect.width/2 - 35/2}px`;
        popup.style.top = `${rect.top + window.scrollY - 200}px`;
        popup.style.display = popup.style.display==='block' ? 'none' : 'block';
    });

    slider.addEventListener('input',(event) =>
    {
        hue = parseFloat(event.target.value);
        updateColor();
    });

    return picker;
}

function getFileBaseHue(fileIndex)
{
    const fileBaseHues =
    [
        0,   // Red
        120, // Green
        240, // Blue
        300, // Magenta
        180, // Cyan
        60,  // Yellow
    ];
    return fileBaseHues[fileIndex % fileBaseHues.length];
}