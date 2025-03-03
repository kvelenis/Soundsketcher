submitButton.addEventListener('click', function () {
    // Show the spinner
    const spinner = document.getElementById('spinner');
    spinner.style.display = 'block';
    let existingSvg = document.querySelector('svg');
    if (existingSvg) {
        existingSvg.remove();
        // Perform visualization or other tasks for all uploaded files
        var canvasWidth = 2000;
        // Create SVG container
        var svgContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svgContainer.setAttribute("id", "svgCanvas");
        svgContainer.setAttribute("width", canvasWidth);
        svgContainer.setAttribute("height", 800);
        
        document.body.appendChild(svgContainer);
    }
    // Perform tasks (e.g., draw visualization and handle playback)
    setTimeout(() => {
        try {
            console.log("globalAudioData:", globalAudioData)
            // drawVisualization(globalAudioData);
            // Determine the maximum duration among all audio data
            maxDuration = Math.max(
                    ...globalAudioData.features.map(file =>
                        file.features[file.features.length - 1]?.timestamp || 0
                    )
                );
            
            globalAudioData.features.forEach((fileData, fileIndex) => {
                    console.log(`File: ${fileData.filename}`);
                    svgContainer = document.getElementById("svgCanvas");
                    canvasWidth = svgContainer.getAttribute("width");
                    drawVisualization(fileData, svgContainer, canvasWidth, maxDuration, fileIndex);
                });

            document.getElementById("svgCanvas").addEventListener("click", (event) => {
                const svgCanvas = event.currentTarget;
                const boundingRect = svgCanvas.getBoundingClientRect();

                // Calculate x-coordinate relative to the canvas
                const clickX = event.clientX - boundingRect.left;

                // Calculate the corresponding timestamp
                clickedPosition = (clickX / boundingRect.width) * maxDuration;

                // Ensure the clicked position is within valid bounds
                clickedPosition = Math.max(0, Math.min(clickedPosition, maxDuration));

                // Move the red vertical line to the clicked position
                moveVerticalLine(clickX);
            });

            // After dynamically adding or updating paths
            initializeTooltip();

        } finally {
            // Hide the spinner when done
            spinner.style.display = 'none';
        }
    }, 100); // Optional delay for better visual feedback
});