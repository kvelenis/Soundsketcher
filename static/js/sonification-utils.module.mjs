let seed = 0;
let rng = randomEngine(seed);

export function dBToGain(dB, threshold = -100) {
    return dB <= threshold ? 0 : Math.pow(10, dB / 20);
}

export function randomEngine(initialSeed) {
    return function () {
        let t = initialSeed += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function getRandomSign() {
    return rng() < 0.5 ? -1 : 1;
}

export function resetRandomSeed(nextSeed = 0) {
    seed = nextSeed;
    rng = randomEngine(seed);
}

export function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
}

export function map(value, start1, stop1, start2, stop2) {
    if (stop1 === start1) {
        return (start2 + stop2) / 2;
    }

    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}

export function perceivedBrightness(r, g, b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function hslToRgb(h, s, l) {
    s /= 100;
    l /= 100;
    const k = (n) => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => l - a * Math.max(Math.min(k(n) - 3, 9 - k(n), 1), -1);
    return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.sonificationUtils = {
    dBToGain,
    randomEngine,
    getRandomSign,
    resetRandomSeed,
    clamp,
    map,
    perceivedBrightness,
    hslToRgb,
};
