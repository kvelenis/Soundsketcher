let audioContexttmp;
let audioBuffer;
let animationFrameId;
let grainSourcePosition = 0;
let activeGrains = [];
let masterGainNode;
let playbackMarkerX = 0;
let isPlayingtmp = false;

function drawRedLine(position) {
    const canvasWidth = canvas.width;
    const xAxis = (position / audioBuffer.duration) * canvasWidth;

    ctx.save();
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(xAxis, 0);
    ctx.lineTo(xAxis, canvas.height);
    ctx.stroke();
    ctx.restore();
}

function drawSpreadMarkers(grainSource, spread) {
    const canvasWidth = canvas.width;
    const grainWidth = canvasWidth / audioBuffer.duration;

    const startX = (grainSource - spread / 2) * grainWidth;
    const endX = (grainSource + spread / 2) * grainWidth;

    ctx.save();
    ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
    ctx.fillRect(Math.max(0, startX), 0, Math.min(endX - startX, canvasWidth), canvas.height);
    ctx.restore();
}

function updateCanvas(currentGrainSource, grainSpread) {
    drawWaveform(audioBuffer);

    if (playbackSpreads.length > 0) {
        playbackSpreads.forEach(({ grainSource, spread }) => {
            drawSpreadMarkers(grainSource, spread);
        });
    }

    drawRedLine(currentGrainSource);
}

function animateGrainSpread(totalDuration, grainSpread) {
    const startTime = performance.now();

    function update() {
        const elapsedTime = (performance.now() - startTime) / 1000;

        if (elapsedTime > totalDuration || !isPlayingtmp) {
            isPlayingtmp = false;
            return;
        }

        const currentGrainSource = grainSourcePosition + Math.sin(elapsedTime * 2 * Math.PI) * grainSpread / 2;

        playbackSpreads = [
            {
                grainSource: currentGrainSource,
                spread: grainSpread,
            },
        ];
        updateCanvas(currentGrainSource, grainSpread);

        requestAnimationFrame(update);
    }

    update();
}

function drawVerticalLine() {
    const svgCanvas = document.getElementById("svgCanvas");
    if (!svgCanvas) return;

    let line = document.getElementById("progressLine");
    if (!line) {
        line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("id", "progressLine");
        line.setAttribute("x1", 0);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", 0);
        line.setAttribute("y2", svgCanvas.getBoundingClientRect().height);
        line.setAttribute("stroke", "red");
        line.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(line);
    }
}

function updateLinePosition(currentTimestamp, totalDuration, canvasWidth, isScrollableMode) {
    const line = document.getElementById("progressLine");
    if (!line) return;

    const xAxis = (currentTimestamp / totalDuration) * canvasWidth;
    line.setAttribute("x1", xAxis);
    line.setAttribute("x2", xAxis);

    console.log("isScrollableMode", isScrollableMode);
    if (isScrollableMode) {
        const scrollOffset = xAxis - window.innerWidth / 2;
        window.scrollTo({ left: scrollOffset, behavior: "smooth" });
    }
}

function smoothUpdateLinePositionForPathData(totalDuration) {
    const svgCanvas = document.getElementById("svgCanvas");
    const canvasWidth = svgCanvas.getBoundingClientRect().width;

    let startTime = performance.now();

    function update() {
        const elapsedTime = (performance.now() - startTime) / 1000;
        if (elapsedTime > totalDuration) {
            cancelAnimationFrame(animationFrameId);
            stopAllGrains();
            return;
        }
        isScrollableMode = document.getElementById("scrollModeToggle").checked;
        updateLinePosition(elapsedTime, totalDuration, canvasWidth, isScrollableMode);

        function calculatePlaybackRate(yAxis, isInverted = false) {
            return isInverted
                ? map(yAxis, 0, 1, 2, 0.5)
                : map(yAxis, 0, 1, 0.5, 2);
        }

        normalisedPathData.forEach((path) => {
            if (elapsedTime >= path.timestamp && elapsedTime < path.timestamp + 0.1) {
                const playbackRate = calculatePlaybackRate(path.yAxis, true);
                playGrain(grainSourcePosition, path.dashArray, path.lineWidth, path.lineLength * 0.05, playbackRate);
            }
        });

        animationFrameId = requestAnimationFrame(update);
    }

    update();
}

function playGrain(baseSourcePosition, dashArraySpread, duration, volume, playbackRate) {
    const source = audioContexttmp.createBufferSource();
    source.buffer = audioBuffer;

    const randomSpread = (Math.random() - 0.5) * dashArraySpread;
    let grainPosition = baseSourcePosition + randomSpread;
    grainPosition = Math.max(0, grainPosition % audioBuffer.duration);

    source.playbackRate.value = playbackRate;

    const gainNode = audioContexttmp.createGain();
    const now = audioContexttmp.currentTime;

    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(volume, now + duration * 0.1);
    gainNode.gain.setValueAtTime(volume, now + duration * 0.9);
    gainNode.gain.linearRampToValueAtTime(0, now + duration);

    source.connect(gainNode).connect(masterGainNode);

    source.start(now, grainPosition, duration);
    source.stop(now + duration);

    console.log(
        `Grain started at ${grainPosition}s with duration ${duration}s, volume ${volume}, playbackRate ${playbackRate}`
    );

    activeGrains.push(source);

    setTimeout(() => {
        activeGrains = activeGrains.filter((grain) => grain !== source);
    }, duration * 1000);
}

function stopAllGrains() {
    activeGrains.forEach((grain) => grain.stop());
    activeGrains = [];
}

function startGranulation() {
    if (!audioBuffer) {
        console.error("No audio loaded.");
        return;
    }

    if (normalisedPathData.length === 0) {
        console.error("Path data not computed. Please compute path attributes first.");
        return;
    }

    const totalDuration = normalisedPathData[normalisedPathData.length - 1].timestamp;
    drawVerticalLine();
    smoothUpdateLinePositionForPathData(totalDuration);
}
