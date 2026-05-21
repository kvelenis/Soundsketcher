const root = window.SoundSketcher = window.SoundSketcher || {};

export const sonificationState = root.sonificationState || {
    synthsInitialized: false,
    engineMap: null,
    synthData: [],
};

root.sonificationState = sonificationState;
