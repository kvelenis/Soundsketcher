let isScrollableMode = false;


submitButton.addEventListener('click', function () {

    if (!globalAudioData)
    {
        return;
    }

    //! Added this
    if(isPlaying)
    {
        togglePlayStop();
    }
    // Show the spinner
    const spinner = document.getElementById('spinner');
    //! Temporarily disabling objectifier
    const isObjectifyEnabled = false;
    // const isObjectifyEnabled = document.getElementById("objectifyToggle").checked
    isScrollableMode = document.getElementById("scrollModeToggle").checked;

    spinner.style.display = 'block';
    let existingSvg = document.querySelector('svg');

    
    if (existingSvg)
    {
        existingSvg.remove();
    } //! This bracket was at line ~50
    // Perform visualization or other tasks for all uploaded files
    const maxDuration = Math.max(
        ...globalAudioData.data.map(file =>
            file.features[file.features.length - 1]?.timestamp || 0
        )
    );

    //! Added this to show scroll even for short audios
    let canvasWidth;
    if(isScrollableMode)
    {
        pixelsPerSecond = 100;
        let width = Math.ceil(maxDuration * pixelsPerSecond)
        while(width <= window.innerWidth)
        {
            pixelsPerSecond += 100;
            width = Math.ceil(maxDuration * pixelsPerSecond)
        }
        canvasWidth = width;
    }
    else
    {
        canvasWidth = window.innerWidth;
    }
    pixelsPerSecond = canvasWidth/maxDuration;


    // canvasWidth = isScrollableMode
    //     ? Math.ceil(maxDuration * pixelsPerSecond)
    //     : window.innerWidth;

    console.log("canvasWidth", canvasWidth);

    const canvasHeight = Math.floor(window.innerHeight * 0.85); // 85% of viewport height
    const padding = 60;
    const base_height = canvasHeight - 2*padding;
    // Create SVG container

    var svgContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgContainer.setAttribute("id", "svgCanvas");
    svgContainer.setAttribute("width", canvasWidth);
    svgContainer.setAttribute("padding",padding);
    svgContainer.setAttribute("base_height",base_height);
    
    svgContainer.setAttribute("height", canvasHeight);

    svgContainer.addEventListener("dragover",(event) => event.preventDefault()); //! Added this
    svgContainer.addEventListener("drop",(event) => handleDrop(event)); //! Added this

    const wrapper = document.getElementById('svgWrapper');
    wrapper.innerHTML = ''; // clear previous if any
    wrapper.style.overflowX = isScrollableMode ? 'scroll' : 'hidden';
    wrapper.appendChild(svgContainer);

    // Perform tasks (e.g., draw visualization and handle playback)
    setTimeout(() => {
        try {
            console.log("globalAudioData:", globalAudioData)
            // drawVisualization(globalAudioData);
            // Determine the maximum duration among all audio data
            // maxDuration = Math.max(
            //         ...globalAudioData.data.map(file =>
            //             file.features[file.features.length - 1]?.timestamp || 0
            //         )
            //     );
            
            //! Triggering drawVisualization once for all files
            drawVisualization();
            // globalAudioData.data.forEach((fileData, fileIndex) => {
            //         console.log(`File: ${fileData.filename}`);
            //         svgContainer = document.getElementById("svgCanvas");
            //         canvasWidth = svgContainer.getAttribute("width");
            //         const baseHue = getFileBaseHue(fileIndex);
            //         drawVisualization(fileData, svgContainer, canvasWidth, maxDuration, fileIndex, baseHue, isObjectifyEnabled);
            //     });

            document.getElementById("svgCanvas").addEventListener("click", (event) => {
                const svgCanvas = event.currentTarget;
                const boundingRect = svgCanvas.getBoundingClientRect();

                // Calculate x-coordinate relative to the canvas
                const clickX = event.clientX - boundingRect.left;

                // Calculate the corresponding timestamp
                cursorTime = (clickX / boundingRect.width) * maxDuration;

                // Ensure the clicked position is within valid bounds
                cursorTime = Math.max(0, Math.min(cursorTime, maxDuration));

                // Move the red vertical line to the clicked position
                moveVerticalLine(clickX);
                changeAudioCurrentTime() //! Added this
            });

            // After dynamically adding or updating paths
            initializeTooltip();

        } finally {
            // Hide the spinner when done
            if(!isLoading)
            {
                spinner.style.display = 'none';
            }
        }
    }, 100); // Optional delay for better visual feedback
});

let resizeTimeout;
window.addEventListener('resize', () =>
{
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() =>
    {
        if (globalAudioData)
        {
            submitButton.click(); // Re-trigger visualization
        }
    }, 300); // Adjust debounce delay as needed
});