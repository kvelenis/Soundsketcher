import "./feature-config.module.mjs?v=frontend-migration-84";
import "./app-state.module.mjs?v=frontend-migration-85";
import "./audio-cache-client.module.mjs?v=frontend-migration-114";
import "./audio-upload-client.module.mjs?v=frontend-migration-84";
import "./playback-controller.module.mjs?v=frontend-migration-134";
import "./mixer.module.mjs?v=frontend-migration-95";
import "./audio-visualization-orchestrator.module.mjs?v=frontend-migration-95";
import "./sonification-utils.module.mjs?v=frontend-migration-114";
import "./sonification-config.module.mjs?v=frontend-migration-113";
import "./sonification-state.module.mjs?v=frontend-migration-112";
import "./granular-helpers.module.mjs?v=frontend-migration-111";
import "./sonification-data-prep.module.mjs?v=frontend-migration-111";
import "./sonification-engine-orchestrator.module.mjs?v=frontend-migration-114";
import "./oscillator-controller.module.mjs?v=frontend-migration-112";
import "./granular-controller.module.mjs?v=frontend-migration-112";
import "./sonification-playback-controller.module.mjs?v=frontend-migration-113";

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.main = {
    audioModulesLoaded: true,
    featureConfigLoaded: true,
    appStateLoaded: true,
    sonificationOrchestratorLoaded: true,
    sonificationHelpersLoaded: true,
    sonificationControllersLoaded: true,
};
