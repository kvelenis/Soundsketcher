(function () {
    const root = window.SoundSketcher = window.SoundSketcher || {};

    root.audio = root.audio || {};
    root.featureConfig = window.SoundSketcherFeatureConfig;
    const audioClientWaiters = root.__audioClientWaiters || {};
    root.__audioClientWaiters = audioClientWaiters;

    root.state = root.state || {};

    root.basePath = function () {
        const pathname = window.location.pathname || "/";
        if (pathname === "/soundsketcher" || pathname.startsWith("/soundsketcher/")) return "/soundsketcher";
        if (pathname === "/app1" || pathname.startsWith("/app1/")) return "/app1";
        return "";
    };

    root.url = function (path) {
        const normalizedPath = path.startsWith("/") ? path : `/${path}`;
        const basePath = root.basePath();
        if (!basePath) return normalizedPath;
        return normalizedPath === "/" ? `${basePath}/` : `${basePath}${normalizedPath}`;
    };

    window.SoundSketcherUrl = root.url;
    [
        "audioContext",
        "pathData",
        "pixelsPerSecond",
        "synthPlaying",
        "isLoading",
        "audioPlayer",
        "audioDuration",
        "globalAudioData",
        "globalFile",
        "cursorTime",
        "features_state",
        "inverted_state",
        "sliders_state",
    ].forEach((name) => {
        if (Object.prototype.hasOwnProperty.call(root.state, name)) return;

        Object.defineProperty(root.state, name, {
            configurable: true,
            get() {
                return window[name];
            },
            set(value) {
                window[name] = value;
            },
        });
    });

    root.registerAudioClient = function (name, client) {
        root.audio[name] = client;
        if (audioClientWaiters[name]) {
            audioClientWaiters[name].splice(0).forEach((resolve) => resolve(client));
        }
        return client;
    };

    root.whenAudioClient = function (name) {
        if (root.audio[name]) {
            return Promise.resolve(root.audio[name]);
        }

        return new Promise((resolve) => {
            audioClientWaiters[name] = audioClientWaiters[name] || [];
            audioClientWaiters[name].push(resolve);
        });
    };

    root.registerApp = function (app) {
        root.app = app;
        return app;
    };

    if (window.SoundSketcherApp) {
        root.registerApp(window.SoundSketcherApp);
    }
})();
