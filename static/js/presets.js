(function () {
  const lineSlidersA = [
    { min: 0, max: 100, startMin: 0, startMax: 50 },
    { min: 0, max: 15, startMin: 2, startMax: 4 },
    { min: 0, max: 100, startMin: 0, startMax: 100 },
    { min: 0, max: 100, startMin: 40, startMax: 90 },
    { min: 0, max: 45, startMin: 2, startMax: 25 },
    { min: 0, max: 10, startMin: 2, startMax: 6 },
  ];

  const polygonSlidersA = [
    { min: 3, max: 20, startMin: 3, startMax: 15 },
    { min: 3, max: 12, startMin: 3, startMax: 8 },
    { min: 0, max: 100, startMin: 0, startMax: 100 },
    { min: 0, max: 100, startMin: 40, startMax: 90 },
    { min: 0, max: 45, startMin: 0, startMax: 30 },
    { min: 0, max: 100, startMin: 20, startMax: 100 },
  ];

  const invertedA = [false, true, false, true, true, false, false];

  window.SoundSketcherPresets = {
    A: {
      key: "z",
      features: [
        ["loudness", "spectral_centroid", "loudness", "loudness", "yin_f0_librosa", "loudness", "spectral_centroid"],
        ["loudness", "spectral_centroid", "loudness", "loudness", "yin_f0_librosa", "loudness", "spectral_centroid"],
      ],
      inverted: [invertedA, invertedA],
      sliders: [lineSlidersA, polygonSlidersA],
    },
    B: {
      key: "x",
      features: [
        Array(7).fill("spectral_centroid"),
        Array(7).fill("spectral_centroid"),
      ],
      inverted: [invertedA, invertedA],
      sliders: [lineSlidersA, polygonSlidersA],
    },
    C: {
      key: "c",
      features: [
        Array(7).fill("loudness"),
        Array(7).fill("loudness"),
      ],
      inverted: [invertedA, invertedA],
      sliders: [lineSlidersA, polygonSlidersA],
    },
  };
})();
