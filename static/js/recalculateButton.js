(function () {
    function getRecalculateState() {
        return window.SoundSketcher.state;
    }

    function getRecalculateFeatureConfig() {
        return window.SoundSketcher.featureConfig;
    }

    function appendCachedFiles(formData, audioData) {
        const filesProcessed = audioData.files_processed || 0;
        for (let index = 0; index < filesProcessed; index += 1) {
            formData.append("filenames", audioData.filename[index]);
            formData.append("hashes", audioData.hash[index]);
        }
    }

    function shouldRunObjectifier() {
        return Boolean(document.getElementById("objectifierMode")?.checked);
    }

    function showFeatureViewWhileObjectifierProcesses(data) {
        const hasPendingObjectifier = (data?.data || []).some((fileData) => {
            const status = fileData?.objectifier_job?.status;
            return !Array.isArray(fileData?.clusters) && ["queued", "running"].includes(status);
        });
        const objectifierMode = document.getElementById("objectifierMode");
        if (!hasPendingObjectifier || !objectifierMode?.checked) return false;

        objectifierMode.checked = false;
        objectifierMode.dispatchEvent(new Event("change"));
        return true;
    }

    window.SoundSketcherApp.onReady("recalculate button", () => {
        const recalculateButton = document.getElementById("recalculate_button");
        const lengthSelector = document.getElementById("window_length_selector");
        const overlapSelector = document.getElementById("window_overlap_selector");
        const normalizeButton = document.getElementById("normalize_button");
        const filterButton = document.getElementById("filter_button");
        const resketchButton = document.getElementById("submitButton");
        const spinner = document.getElementById("spinner");

        if (!recalculateButton || !lengthSelector || !overlapSelector || !filterButton || !resketchButton) {
            return;
        }

        recalculateButton.addEventListener("click", async () => {
            const state = getRecalculateState();
            if (!state.globalAudioData) return;

            if (spinner) spinner.style.display = "block";

            const formData = new FormData();
            formData.append("n_fft", lengthSelector.value);
            formData.append("overlap", getRecalculateFeatureConfig().percentages[overlapSelector.value]);
            formData.append("normalize_audio", Boolean(normalizeButton?.checked));
            formData.append("apply_filter", filterButton.checked);
            formData.append("save_json", shouldRunObjectifier());
            formData.append("run_objectifier", shouldRunObjectifier());
            appendCachedFiles(formData, state.globalAudioData);

            try {
                const response = await fetch(window.SoundSketcher.url("/recalculate_features"), {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error("Feature recalculation failed.");
                }

                const data = await response.json();
                state.globalAudioData = data;
                window.SoundSketcherFeatureStatus?.updateFeatureExtractionStatus();
                window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();

                Swal.fire({
                    icon: "success",
                    title: "Recalculation started — canvas will update when ready.",
                    position: "top-end",
                    showConfirmButton: false,
                    timer: 2000,
                });
            } catch (error) {
                console.error(error);
                alert("Something went wrong during recalculations.");
                if (spinner) spinner.style.display = "none";
            }
        });
    });
})();
