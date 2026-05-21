var audioContext = new AudioContext();
var pathData = {};
var pixelsPerSecond = 100;
var synthPlaying = false;
var isLoading = false;

var rawFeatureNames = window.SoundSketcherFeatureConfig.rawFeatureNames;
var visibleFeatureNames = window.SoundSketcherFeatureConfig.visibleFeatureNames;
var PITCH_FEATURES = new Set(window.SoundSketcherFeatureConfig.pitchFeatureNames);
var allowedLogFeatures = window.SoundSketcherFeatureConfig.allowedLogFeatures;
var percentages = window.SoundSketcherFeatureConfig.percentages;

var audioPlayer = document.getElementById("audioPlayer");
var audioDuration = 0;
var globalAudioData = null;
var globalFile = null;
var cursorTime = 0;

var features_state;
var inverted_state;
var sliders_state;
