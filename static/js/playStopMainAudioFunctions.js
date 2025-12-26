let audioPlayers = []; // Array to store audio players
let audioFiles = []; // Array to store uploaded audio files
let isPlaying = false;

//! Added these
const playStopIcon = document.getElementById("playStopIcon");
const playStopIconHeader = document.getElementById("playStopIconHeader");

//! Added this for microphone files
async function getAudioDuration(file) {
    // const audioContext = new AudioContext(); // replaced by global audioContext variable in index.html
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.duration;
}

//! Changed this for microphone files
// Function to load audio files into players
async function loadAudioFiles(files) {
    console.log("files inside loaAudioFiles", files)
    audioPlayers = []; // Clear existing players
    audioFiles = files; // Store files globally

    for (const file of files)
    {
        const audio = new Audio(URL.createObjectURL(file));
        if(!isFinite(audio.duration))
        {
            try
            {
                const duration = await getAudioDuration(file);
                console.log('Duration:', duration);
                audio._duration = duration;
            } catch (err) {
                console.warn('Could not get duration:', err.message);
            }
        }
        else
        {
            audio._duration = audio.duration;
        }
        audioPlayers.push(audio);
    }
}

//! New function to wait for file metadata
function loadMetadata(audio)
{
    return new Promise(resolve =>
    {
        audio.addEventListener("loadedmetadata",() => resolve(),{once: true});
    });
}

//! Light modifications (moved this here from handleDropAudio.js)
async function loadAudioFromUrls(audioFileList)
{
    audioPlayers = [];
    audioFiles = audioFileList;

    for (const file of audioFileList)
    {
        const audio = new Audio("/app1/" + file.audio_url);

        //! Added this
        await loadMetadata(audio);
        audio._duration = audio.duration;
        
        audioPlayers.push(audio);
    }

    if (audioFileList.length > 0)
    {
        Swal.fire(
        {
            icon: "info",
            title: "Loaded example",
            text: `${audioFileList[0].name}`, //! changed .filename (undefined) to .name
            timer: 1500,
            showConfirmButton: false
        });
    }
}

function moveVerticalLine(x) {
    const svgCanvas = document.getElementById("svgCanvas");
    let progressLine = document.getElementById("progressLine");

    // If the vertical line doesn't exist, create it
    if (!progressLine) {
        progressLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        progressLine.setAttribute("id", "progressLine");
        progressLine.setAttribute("y1", 0);
        progressLine.setAttribute("y2", svgCanvas.getAttribute("height")); // Full height of the canvas
        progressLine.setAttribute("stroke", "red");
        progressLine.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(progressLine);
    }

    // Update the line's position
    progressLine.setAttribute("x1", x);
    progressLine.setAttribute("x2", x);
}

function waitMetadata(el) {
  return new Promise(res => {
    if (el.readyState >= 1 && Number.isFinite(el.duration)) return res();
    el.addEventListener("loadedmetadata", res, { once: true });
  });
}


// Toggle play/stop state
async function togglePlayStop()
{
    if(globalAudioData && globalFile)
    {
        if (!isPlaying)
        {
            // ensure all durations are known
            // await Promise.all(audioPlayers.map(waitMetadata));
            // Calculate the maximum duration dynamically
            const maxDuration = Math.max(...audioPlayers.map(audio => audio._duration));
    
            // Play all audio files from the clicked position
            audioPlayers.forEach(audio => {
                if (cursorTime <= audio._duration) {
                    audio.currentTime = cursorTime; // Start from the clicked position
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
            isScrollableMode = document.getElementById("scrollModeToggle").checked;
            console.log("isScrollableMode", isScrollableMode);
            console.log("audioPlayers exists", audioPlayers, "maxDuration", maxDuration)
            smoothUpdateLinePosition(audioPlayers, maxDuration, isScrollableMode);
    
            // Update play/stop button to "Stop"
            isPlaying = true;
            playStopIcon.src = "static/assets/stop.png"; // Change to stop icon
            playStopIcon.alt = "Stop";
            //! Added this
            playStopIconHeader.src = "static/assets/stop.png"; // Change to stop icon
            playStopIconHeader.alt = "Stop";
        }
        else
        {
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
            cursorTime = 0;
    
            // Update play/stop button to "Play"
            isPlaying = false;
            playStopIcon.src = "static/assets/play.png"; // Change back to play icon
            playStopIcon.alt = "Play";
            playStopIconHeader.src = "static/assets/play.png"; // Change back to play icon
            playStopIconHeader.alt = "Play";
        }
    }
}

//! New function to move audio player to clicked position
function changeAudioCurrentTime()
{
    audioPlayers.forEach(audio =>
    {
        if (cursorTime <= audio._duration)
        {
            audio.currentTime = cursorTime; // Start from the clicked position
            if(isPlaying)
            {
                audio.play();
            }
        }
        else
        {
            audio.pause();
            audio.currentTime = 0; // Reset shorter files
        }
    });
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
        line.setAttribute("y2", svgCanvas.getAttribute("height"));
        line.setAttribute("stroke", "red");
        line.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(line);
    }
}

//! Modified the scrolling logic
function smoothUpdateLinePosition(audioPlayers, maxDuration, isScrollableMode) {
    const line = document.getElementById("progressLine");
    const svgCanvas = document.getElementById("svgCanvas");
    const canvasWidth = svgCanvas.getBoundingClientRect().width;
    const svgWrapper = document.getElementById("svgWrapper");
    const longestAudio = audioPlayers.find(audio => audio._duration === maxDuration);

    let init = true;
    const padding = 100;
    // let newCursorTime = true;
    // let lastCursorTime = cursorTime;
    
    function update()
    {
        if (line && longestAudio)
        {
            const x_canvas = (longestAudio.currentTime / maxDuration) * canvasWidth;
            const x_screen = x_canvas - svgWrapper.scrollLeft;
            // if(cursorTime != lastCursorTime)
            // {
            //     newCursorTime = false;
            //     lastCursorTime = cursorTime
            // }

            // Update line position
            line.setAttribute("x1", x_canvas);
            line.setAttribute("x2", x_canvas);
 
            // === Auto-scroll logic ===
            if (isScrollableMode && svgWrapper)
            {
                // if(newCursorTime)
                // {
                //     if(x_screen < 0 || x_screen > padding)
                //     {
                //         svgWrapper.scrollLeft = x_canvas;
                //     }
                //     newCursorTime = false;
                // }
                // else
                // {
                    if(x_screen > padding)
                    {
                        svgWrapper.scrollLeft = x_canvas - padding;
                    }
                    if(init)
                    {
                        svgWrapper.scrollLeft = x_canvas;
                        init = false;
                    }
                // }
            }
        }

        // Continue animation while audio is playing
        if(longestAudio && !longestAudio.paused)
        {
            animationFrameId = requestAnimationFrame(update);
        }
        else
        {
            //! Added these
            line.remove()
            isPlaying = false;
            playStopIcon.src = "static/assets/play.png";
            playStopIcon.alt = "Play";
            playStopIconHeader.src = "static/assets/play.png";
            playStopIconHeader.alt = "Play";
            cursorTime = 0;
        }
    }

    update(); // Start animation
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

// document.addEventListener("keydown", function(event)
// {
//     if(event.code === "Space") 
//     {
//         event.preventDefault(); // Prevent scrolling the page
//         togglePlayStop();
//     }
// });

if('mediaSession' in navigator)
{
    navigator.mediaSession.setActionHandler('play',null);
    navigator.mediaSession.setActionHandler('play',null);
    // navigator.mediaSession.setActionHandler('play', function()
    // {
    //     // Prevent the default play
    //     console.log('Play key pressed');
    //     togglePlayStop();
    // });

    // navigator.mediaSession.setActionHandler('pause', function()
    // {
    //     // Prevent the default pause
    //     console.log('Pause key pressed');
    //     togglePlayStop();
    // });
}