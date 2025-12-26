function handleDrop(event) {
    event.preventDefault();

    // Show the spinner
    const spinner = document.getElementById('spinner');
    spinner.style.display = 'block';

    var files = event.dataTransfer.files; // Get dropped files
    if (files.length > 0) {
        // Create FormData for the POST request
        var formData = new FormData();

        // Loop through all files and append them to FormData
        for (var i = 0; i < files.length; i++) {
            if (files[i].type === 'audio/wav') {
                formData.append('wav_files', files[i]); // Append files under the key 'wav_files'
            } else {
                alert('Only WAV files are supported. Skipping: ' + files[i].name);
            }
        }

        // Show the spinner while uploading
        spinner.style.display = 'block';

        // Send the POST request
        fetch('/upload_wavs', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to upload WAV files.');
                }
                return response.json(); // Parse JSON response
            })
            .then(data => {
                dropArea.style.display = "none";
                Swal.fire({
                    position: "top-end",
                    icon: "success",
                    title: "WAV files uploaded successfully.",
                    showConfirmButton: false,
                    timer: 1500
                });

                // Handle the response data
                globalAudioData = data;
                globalFile = files;
                console.log('globalFile:', globalFile);

                // Perform visualization or other tasks for all uploaded files
                var canvasWidth = 2000;
                // Create SVG container
                var svgContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                svgContainer.setAttribute("id", "svgCanvas");
                svgContainer.setAttribute("width", canvasWidth);
                svgContainer.setAttribute("height", 800);
                
                document.body.appendChild(svgContainer);
                document.getElementById("controls").style.display = "flex";

                loadAudioFiles(Array.from(files));
                createMixerUI();
                // Bind the spacebar key to toggle play/stop
                document.addEventListener("keydown", function (event) {
                    // Check if the pressed key is the spacebar (keyCode 32 or event.code 'Space')
                    if (event.code === "Space" || event.keyCode === 32) {
                        event.preventDefault(); // Prevent default behavior (like page scrolling)
                        const playStopButton = document.getElementById("playStopButton");
                        if (playStopButton) {
                            togglePlayStop(); // Call your existing play/stop function
                        }
                    }
                });
                var maxDuration = Math.max(
                    ...globalAudioData.features.map(file =>
                        file.features[file.features.length - 1]?.timestamp || 0
                    )
                );
                console.log("Max Duration:", maxDuration);

                data.features.forEach((fileData,fileIndex) => {
                    console.log(`File: ${fileData.filename}`);
                    drawVisualization(fileData, svgContainer, canvasWidth, maxDuration, fileIndex);
                    console.log("fileData:", fileData);
                    drawClusterOverlays(fileData.clusters, svgContainer, canvasWidth, 800, maxDuration);
                    
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

                initializeTooltip();


                // createToggleButtons(); // Create buttons after paths are added
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to upload WAV files.');
            })
            .finally(() => {
                // Hide the spinner when done
                spinner.style.display = 'none';
            });
    } else {
        alert('No files were dropped.');
        // Hide the spinner immediately if no files were provided
        spinner.style.display = 'none';
    }
}

function handleDragOver(event) {
    event.preventDefault();
}