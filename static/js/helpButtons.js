(function () {
    function bindModal(openId, modalId, closeId, display = "flex") {
        const openButton = document.getElementById(openId);
        const modal = document.getElementById(modalId);
        const closeButton = document.getElementById(closeId);
        if (!openButton || !modal || !closeButton) return;

        openButton.addEventListener("click", (event) => {
            if (openButton.tagName === "A") event.preventDefault();
            modal.style.display = display;
        });

        closeButton.addEventListener("click", () => {
            modal.style.display = "none";
        });

        window.addEventListener("click", (event) => {
            if (event.target === modal) {
                modal.style.display = "none";
            }
        });
    }

    function bindHelpButtons() {
        bindModal("audioFeaturesHelp", "helpAudioFeatureModal", "closeAudioFeatureModal");
        bindModal("mappindSettingsHelp", "helpMappingModal", "closeMappingModal");
        bindModal("sonificatorsHelp", "helpSonificatorsModal", "closeSonificatorsModal");
        bindModal("openClampFeatureModal", "clampFeatureModal", "closeClampFeatureModal", "block");
        bindModal("openAdvancedSettingsModal", "advancedSettingsModal", "closeAdvancedSettingsModal", "block");
        bindModal("contactInfoLink", "contactInfoModal", "closeContactInfoModal", "block");
    }

    window.SoundSketcherApp.onReady("help buttons", bindHelpButtons);

    window.openContactInfoModal = function (event) {
        if (event) event.preventDefault();
        const modal = document.getElementById("contactInfoModal");
        if (modal) modal.style.display = "block";
    };
})();
