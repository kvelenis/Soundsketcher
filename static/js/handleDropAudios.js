function getUploadDropArea() {
    return document.getElementById("drop_area");
}

function getUploadState() {
    return window.SoundSketcher.state;
}

function getUploadFeatureConfig() {
    return window.SoundSketcher.featureConfig;
}

function setUploadLoading(spinner, nextIsLoading) {
    if (spinner) {
        spinner.style.display = nextIsLoading ? "block" : "none";
    }
    getUploadState().isLoading = nextIsLoading;
}

function alertIfUploadIsBusy() {
    if (!getUploadState().isLoading) return false;

    alert("Please wait for loading to finish");
    return true;
}

function getAudioUploadClient() {
    return window.SoundSketcher?.whenAudioClient?.("upload") || Promise.resolve(window.SoundSketcherAudioUploadClient);
}

function getAudioCacheClient() {
    return window.SoundSketcher?.whenAudioClient?.("cache") || Promise.resolve(window.SoundSketcherAudioCacheClient);
}

function getAudioVisualizationClient() {
    return window.SoundSketcher?.whenAudioClient?.("visualize") || Promise.resolve(window.SoundSketcherAudioVisualizationOrchestrator);
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
    if (!hasPendingObjectifier || !objectifierMode?.checked) return;

    objectifierMode.checked = false;
    objectifierMode.dispatchEvent(new Event("change"));
}

function hasPendingFeatureExtraction(data) {
    return (data?.data || []).some((fileData) => {
        const status = fileData?.feature_job?.status;
        return ["queued", "running"].includes(status);
    });
}

async function handleDrop(event) {
    event.preventDefault();

    if (alertIfUploadIsBusy()) {
        return;
    }

    const spinner = document.getElementById("spinner");
    setUploadLoading(spinner, true);

    const n_fft = document.getElementById("window_length_selector").value;
    const overlap = getUploadFeatureConfig().percentages[document.getElementById("window_overlap_selector").value];
    const normalize_audio = document.getElementById("normalize_button").checked;
    const apply_filter = document.getElementById("filter_button").checked;

    const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("audio/"));
    if (files.length === 0) {
        alert("No valid audio files were dropped.");
        setUploadLoading(spinner, false);
        getUploadDropArea()?.classList.remove("solid");
        return;
    }

    const uploadClient = await getAudioUploadClient();
    const { formData, atLeastOneCached } = await uploadClient.prepareUploadRequest(
        files,
        {
            n_fft,
            overlap,
            normalize_audio,
            apply_filter,
            save_json: true,
            run_objectifier: shouldRunObjectifier(),
        }
    );

    let reuseCached = false;
    if (atLeastOneCached) {
        const reuseChoice = await Swal.fire({
            title: "Use cached analysis if available?",
            text: "Some files have already been analyzed. Reuse or reprocess?",
            icon: "question",
            showCancelButton: true,
            confirmButtonText: "Use Cached",
            cancelButtonText: "Reprocess All",
        });
        reuseCached = reuseChoice.isConfirmed;
    }

    try {
        const data = await uploadClient.uploadAudioForm(formData, reuseCached);

        getUploadDropArea()?.style.setProperty("display", "none");

        const state = getUploadState();
        state.globalAudioData = data;
        window.SoundSketcherFeatureStatus?.updateFeatureExtractionStatus();
        window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
        showFeatureViewWhileObjectifierProcesses(data);
        state.globalFile = files;
        const hasPendingExtraction = hasPendingFeatureExtraction(data);
        if (!hasPendingExtraction) {
            const visualizationClient = await getAudioVisualizationClient();
            await visualizationClient.visualizeAllFiles(state.globalFile);
        }

        if (hasPendingExtraction) {
            Swal.fire({
                icon: "info",
                title: "Feature extraction started.",
                text: "The sketch will update automatically when analysis finishes.",
                position: "top-end",
                showConfirmButton: false,
                timer: 2200,
            });
        } else {
            Swal.fire({
                icon: "success",
                title: "Audio files processed successfully.",
                position: "top-end",
                showConfirmButton: false,
                timer: 1500,
            });
        }
    } catch (err) {
        console.error(err);
        alert("Something went wrong during upload.");
        getUploadDropArea()?.classList.remove("solid");
    } finally {
        setUploadLoading(spinner, false);
    }
}


async function fetchPreviouslyProcessed(filename, hash) {
    if (alertIfUploadIsBusy()) {
        return;
    }

    const spinner = document.getElementById("spinner");
    setUploadLoading(spinner, true);

    try {
        const cacheClient = await getAudioCacheClient();
        const data = await cacheClient.loadCachedAudio(filename, hash);
        getUploadDropArea()?.style.setProperty("display", "none");
        const state = getUploadState();
        state.globalAudioData = data;
        window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
        state.globalFile = cacheClient.toCachedFileList(data);
        const visualizationClient = await getAudioVisualizationClient();
        await visualizationClient.visualizeAllFiles(state.globalFile);
    } catch (err) {
        console.error(err);
        alert("Failed to load cached file.");
    } finally {
        setUploadLoading(spinner, false);
    }
}


window.SoundSketcherApp.onReady("upload drop controls", () => {
    const dropArea = document.getElementById('drop_area');
    const hiddenInput = document.getElementById('hiddenFileInput');

    function markDropAreaActive(event) {
        event.preventDefault();
        dropArea?.classList.add("solid");
    }

    // When user clicks the drop area, trigger the file dialog.
    dropArea?.addEventListener('click', () => {
        hiddenInput?.click();
    });

    dropArea?.addEventListener("dragenter", markDropAreaActive);
    dropArea?.addEventListener("dragover", markDropAreaActive);
    dropArea?.addEventListener("dragleave", () => {
        dropArea.classList.remove("solid");
    });
    dropArea?.addEventListener("drop", (event) => {
        dropArea.classList.add("solid");
        handleDrop(event);
    });

    // When files are selected via the dialog, call handleDrop.
    hiddenInput?.addEventListener('change', (event) => {
        const files = event.target.files;

        const fakeEvent = {
            preventDefault: () => {},
            dataTransfer: { files }
        };

        dropArea?.classList.add('solid');
        handleDrop(fakeEvent);

        // Clear input so the same file can be selected again if needed.
        hiddenInput.value = '';
    });
});
