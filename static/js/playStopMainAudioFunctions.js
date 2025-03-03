let audioPlayers = []; // Array to store audio players
let audioFiles = []; // Array to store uploaded audio files
let isPlaying = false; // Playback state


// Function to load audio files into players
function loadAudioFiles(files) {
    console.log("files inside loaAudioFiles", files)
    audioPlayers = []; // Clear existing players
    audioFiles = files; // Store files globally

    files.forEach(file => {
        const audio = new Audio(URL.createObjectURL(file));
        audioPlayers.push(audio);
    });
}





function moveVerticalLine(x) {
    const svgCanvas = document.getElementById("svgCanvas");
    let progressLine = document.getElementById("progressLine");

    // If the vertical line doesn't exist, create it
    if (!progressLine) {
        progressLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        progressLine.setAttribute("id", "progressLine");
        progressLine.setAttribute("y1", 0);
        progressLine.setAttribute("y2", 800); // Full height of the canvas
        progressLine.setAttribute("stroke", "red");
        progressLine.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(progressLine);
    }

    // Update the line's position
    progressLine.setAttribute("x1", x);
    progressLine.setAttribute("x2", x);
}

// Toggle play/stop state
function togglePlayStop() {
    const button = document.getElementById("playStopButton");

    if (!isPlaying) {
        // Calculate the maximum duration dynamically
        const maxDuration = Math.max(...audioPlayers.map(audio => audio.duration));

        // Play all audio files from the clicked position
        audioPlayers.forEach(audio => {
            if (clickedPosition <= audio.duration) {
                audio.currentTime = clickedPosition; // Start from the clicked position
                audio.play();
            } else {
                audio.pause();
                audio.currentTime = 0; // Reset shorter files
            }
        });

        // Draw the vertical line if it doesn't already exist
        let line = document.getElementById("progressLine");
        if (!line) {
            drawVerticalLine();
        }

        // Draw vertical line based on maxDuration
        // drawVerticalLine();
        // Start the smooth animation
        smoothUpdateLinePosition(audioPlayers, maxDuration);

        // Update play/stop button to "Stop"
        isPlaying = true;
        playStopIcon.src = "static/assets/stop.png"; // Change to stop icon
        playStopIcon.alt = "Stop";
    } else {
        // Stop all audio files
        audioPlayers.forEach(audio => {
            audio.pause();
            audio.currentTime = 0; // Reset playback position
        });

        // Stop the smooth animation
        cancelAnimationFrame(animationFrameId);

        // Remove the vertical line
        const line = document.getElementById("progressLine");
        if (line) {
            line.remove(); // Remove the line from the SVG canvas
        }

        // Reset the clicked position to the start
        clickedPosition = 0;

        // Update play/stop button to "Play"
        isPlaying = false;
        playStopIcon.src = "static/assets/play.png"; // Change back to play icon
        playStopIcon.alt = "Play";
    }
}

// Function to draw the vertical line
function drawVerticalLine() {
    const svgCanvas = document.getElementById("svgCanvas");

    if (!document.getElementById("progressLine")) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("id", "progressLine");
        line.setAttribute("x1", 0);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", 0);
        line.setAttribute("y2", 800);
        line.setAttribute("stroke", "red");
        line.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(line);
    }
}

function smoothUpdateLinePosition(audioPlayers, maxDuration) {
    const svgCanvas = document.getElementById("svgCanvas");
    const line = document.getElementById("progressLine");
    const canvasWidth = svgCanvas.getBoundingClientRect().width;

    // Find the audio player corresponding to the longest duration
    const longestAudio = audioPlayers.find(audio => audio.duration === maxDuration);

    function update() {
        if (line && longestAudio) {
            // Calculate the current progress of the longest audio file
            const x = (longestAudio.currentTime / maxDuration) * canvasWidth;

            // Ensure the line doesn't move outside the canvas
            const clampedX = Math.max(0, Math.min(x, canvasWidth));

            // Update line position
            line.setAttribute("x1", clampedX);
            line.setAttribute("x2", clampedX);

            // Optional: Animate intersected paths
            // animateIntersectedPaths(clampedX);
        }

        // Request the next frame if playback is ongoing
        if (longestAudio && !longestAudio.paused) {
            animationFrameId = requestAnimationFrame(update);
        }
    }

    // Start the animation
    update();
}

function animateIntersectedPaths(lineX) {
    const svgCanvas = document.getElementById("svgCanvas");

    // Get all path elements in the SVG
    const paths = svgCanvas.querySelectorAll("path");

    paths.forEach((path) => {
        const pathBoundingBox = path.getBBox(); // Get the bounding box of the path

        // Check if the line intersects with the path
        if (lineX >= pathBoundingBox.x && lineX <= pathBoundingBox.x + pathBoundingBox.width) {
            // Calculate transform-origin based on bounding box
            const transformOriginX = pathBoundingBox.x + pathBoundingBox.width / 2;
            const transformOriginY = pathBoundingBox.y + pathBoundingBox.height / 2;

            // Apply the transform-origin directly on the path
            path.style.transformOrigin = `${transformOriginX}px ${transformOriginY}px`;

            // Apply the animation
            path.classList.add("zoom-animation");

            // Remove the class after animation completes
            setTimeout(() => {
                path.classList.remove("zoom-animation");
            }, 300); // Animation duration in milliseconds
        }
    });
}

// Function to update the position of the vertical line
function updateLinePosition(audioPlayers, maxDuration) {
    const svgCanvas = document.getElementById("svgCanvas");
    const line = document.getElementById("progressLine");

    if (line) {
        const canvasWidth = svgCanvas.getBoundingClientRect().width;

        // Calculate the average current time across all audio files
        const averageTime =
            audioPlayers.reduce((sum, audio) => sum + audio.currentTime, 0) /
            audioPlayers.length;

        // Normalize the position based on maxDuration
        const x = (averageTime / maxDuration) * canvasWidth;

        // Update line position
        line.setAttribute("x1", x);
        line.setAttribute("x2", x);
    }
}