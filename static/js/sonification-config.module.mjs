function formatDecibelTooltip(value, silentAtThreshold) {
    if (silentAtThreshold ? value <= -36 : value < -36) {
        return "-Infinity dB";
    }
    if (value === 0) {
        return "0.0 dB";
    }
    return `${value > 0 ? "+" : "-"}${Math.abs(value).toFixed(1)} dB`;
}

const root = window.SoundSketcher = window.SoundSketcher || {};

export const sonificationConfig = root.sonificationConfig || {
    waveforms: [
        { name: "sine", image: window.SoundSketcher.url("/sandbox-static/assets/sine-wave.svg") },
        { name: "triangle", image: window.SoundSketcher.url("/sandbox-static/assets/triangle-wave.svg") },
        { name: "square", image: window.SoundSketcher.url("/sandbox-static/assets/square-wave.svg") },
        { name: "sawtooth", image: window.SoundSketcher.url("/sandbox-static/assets/sawtooth-wave.svg") },
    ],
    createMasterVolumeSliderOptions({ silentAtThreshold = false } = {}) {
        return {
            start: 0,
            connect: [true, false],
            range: {
                min: -36.5,
                max: 18,
            },
            step: 0.5,
            tooltips: {
                to: (value) => formatDecibelTooltip(value, silentAtThreshold),
                from: (value) => parseFloat(value),
            },
        };
    },
};

root.sonificationConfig = sonificationConfig;
