function downmixBufferToMono(audioContext, buffer) {
    const length = buffer.length;
    const sampleRate = buffer.sampleRate;
    const numChannels = buffer.numberOfChannels;
    const monoBuffer = audioContext.createBuffer(1, length, sampleRate);
    const monoData = monoBuffer.getChannelData(0);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let index = 0; index < length; index++) {
            monoData[index] += channelData[index];
        }
    }

    for (let index = 0; index < length; index++) {
        monoData[index] /= numChannels;
    }

    return monoBuffer;
}

function normalizeMonoBuffer(audioContext, buffer) {
    const length = buffer.length;
    const sampleRate = buffer.sampleRate;
    const source = buffer.getChannelData(0);

    let peak = 0;
    for (let index = 0; index < length; index++) {
        const abs = Math.abs(source[index]);
        if (abs > peak) {
            peak = abs;
        }
    }

    const normalizedBuffer = audioContext.createBuffer(1, length, sampleRate);
    const destination = normalizedBuffer.getChannelData(0);

    if (peak === 0) {
        destination.set(source);
        return normalizedBuffer;
    }

    const scale = 1 / peak;
    for (let index = 0; index < length; index++) {
        destination[index] = source[index] * scale;
    }

    return normalizedBuffer;
}

export function processBuffer(audioContext, buffer) {
    return normalizeMonoBuffer(audioContext, downmixBufferToMono(audioContext, buffer));
}

function drawSampleLines(context, fileData, canvasWidth, yCenter) {
    context.strokeStyle = "rgba(0, 0, 128, 1)";
    context.fillStyle = context.strokeStyle;

    const step = canvasWidth / fileData.length;
    let x = step / 2;

    for (let index = 0; index < fileData.length; index++) {
        const sample = fileData[index];
        const y = (1 - sample) * yCenter;

        context.beginPath();
        context.moveTo(x, yCenter);
        context.lineTo(x, y);
        context.stroke();

        context.beginPath();
        context.arc(x, y, 2, 0, Math.PI * 2);
        context.fill();

        x += step;
    }
}

function drawEnvelope(context, fileData, canvasWidth, yCenter) {
    context.strokeStyle = "rgba(0, 0, 128, 1)";
    context.fillStyle = "rgba(0, 96, 255, 1)";

    const envelope = new Array(canvasWidth);
    const step = Math.ceil(fileData.length / canvasWidth);

    for (let x = 0; x < canvasWidth; x++) {
        let min = Infinity;
        let max = -Infinity;
        const start = Math.min(x * step, fileData.length - 1);
        const end = Math.min(start + step, fileData.length);

        for (let index = start; index < end; index++) {
            const sample = fileData[index];
            min = Math.min(min, sample);
            max = Math.max(max, sample);
        }
        envelope[x] = { min, max };
    }

    const path = new Path2D();
    path.moveTo(0, (1 - envelope[0].max) * yCenter);

    for (let x = 0; x < canvasWidth; x++) {
        path.lineTo(x, (1 - envelope[x].max) * yCenter);
    }
    for (let x = canvasWidth - 1; x >= 0; x--) {
        path.lineTo(x, (1 - envelope[x].min) * yCenter);
    }

    path.closePath();
    context.fill(path);
    context.stroke(path);
}

function drawSourcePositions(context, canvasWidth, canvasHeight) {
    const colorSliders = document.querySelectorAll(".color-slider");
    colorSliders.forEach((slider) => {
        const hue = slider.value;
        const x = canvasWidth * hue / 360;
        context.strokeStyle = `hsl(${hue},100%,50%)`;
        context.lineWidth = 3;
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, canvasHeight);
        context.stroke();
    });
}

export function drawWaveform({ canvas, container, fileData }) {
    if (!fileData) return;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const context = canvas.getContext("2d");
    const yCenter = canvasHeight / 2;

    context.clearRect(0, 0, canvasWidth, canvasHeight);
    context.beginPath();
    context.lineWidth = 1;
    context.strokeStyle = getComputedStyle(container).borderColor;
    context.moveTo(0, yCenter);
    context.lineTo(canvasWidth, yCenter);
    context.stroke();

    if (fileData.length <= canvasWidth / 2) {
        drawSampleLines(context, fileData, canvasWidth, yCenter);
    } else {
        drawEnvelope(context, fileData, canvasWidth, yCenter);
    }

    drawSourcePositions(context, canvasWidth, canvasHeight);
}

window.SoundSketcher.granularHelpers = {
    processBuffer,
    drawWaveform,
};
