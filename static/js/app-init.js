(function () {
    const app = window.SoundSketcherApp = window.SoundSketcherApp || {};
    window.SoundSketcher?.registerApp?.(app);

    const initializers = [];
    let hasInitialized = false;

    function runInitializer(name, callback) {
        try {
            callback();
        } catch (error) {
            console.error(`SoundSketcher initializer failed: ${name}`, error);
        }
    }

    app.registerInitializer = function (name, callback) {
        if (typeof name === "function") {
            callback = name;
            name = callback.name || "anonymous";
        }

        if (typeof callback !== "function") {
            console.error(`SoundSketcher initializer is not a function: ${name}`);
            return;
        }

        if (hasInitialized) {
            runInitializer(name, callback);
            return;
        }

        initializers.push({ name, callback });
    };

    app.onReady = app.registerInitializer;

    function initialize() {
        if (!app.initCoreControls) {
            console.error("SoundSketcher core controls initializer is not available.");
            return;
        }

        app.initCoreControls();
        hasInitialized = true;
        initializers.splice(0).forEach(({ name, callback }) => runInitializer(name, callback));
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})();
