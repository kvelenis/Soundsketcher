const MAX_REC_SEC = 10;

let mediaRecorder = null;
let recordedChunks = [];
let recStartTs = 0;
let recTimerInt = null;
let recStream = null;
let recStopTO = null;
let recUIActive = false;

function getRecCircleButton() {
    return document.getElementById("recCircleBtn");
}

function getRecordingState() {
    return window.SoundSketcher.state;
}

function getRecordingFeatureConfig() {
    return window.SoundSketcher.featureConfig;
}

function getRecordingUploadClient() {
    return window.SoundSketcher?.whenAudioClient?.("upload") || Promise.resolve(window.SoundSketcherAudioUploadClient);
}

function getRecordingVisualizationClient() {
    return window.SoundSketcher?.whenAudioClient?.("visualize") || Promise.resolve(window.SoundSketcherAudioVisualizationOrchestrator);
}

function formatMMSS(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function updateRecButton(countdown = null) {
    const recCircleBtn = getRecCircleButton();

    if (!recCircleBtn) {
        return;
    }

    if (!recUIActive) {
        recCircleBtn.classList.remove("is-recording");
        recCircleBtn.textContent = "";
    } else {
        recCircleBtn.classList.add("is-recording");
        if (countdown !== null) {
            recCircleBtn.textContent = String(countdown);
        }
    }
}

function startTimer() {
    const timer = document.getElementById("recTimer");
    recStartTs = Date.now();

    if (timer) {
        timer.style.display = "inline";
        timer.textContent = formatMMSS(MAX_REC_SEC);
    }

    if (recTimerInt) clearInterval(recTimerInt);
    if (recStopTO) clearTimeout(recStopTO);

    recTimerInt = setInterval(() => {
        const elapsed = (Date.now() - recStartTs) / 1000;
        const remaining = Math.max(0, MAX_REC_SEC - elapsed);

        if (timer) {
            timer.textContent = formatMMSS(Math.ceil(remaining));
        }

        updateRecButton(Math.ceil(remaining));
    }, 250);

    recStopTO = setTimeout(() => {
        stopRecording();
    }, MAX_REC_SEC * 1000);
}

function stopTimer() {
    const timer = document.getElementById("recTimer");

    clearInterval(recTimerInt);
    recTimerInt = null;
    clearTimeout(recStopTO);
    recStopTO = null;

    if (timer) {
        timer.style.display = "none";
        timer.textContent = "00:00";
    }

    recUIActive = false;
    updateRecButton();
}

async function getBestAudioMimeType() {
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
    ];

    for (const type of candidates) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }

    return "";
}

async function startRecording() {
    if (getRecordingState().isLoading) {
        alert("Please wait for loading to finish");
        return;
    }

    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = await getBestAudioMimeType();

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(recStream, mimeType ? { mimeType } : undefined);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        try {
            const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
            const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
            const fname = `recording_${new Date().toISOString().replace(/[:.]/g, "-")}.${ext}`;
            const fileFromBlob = new File([blob], fname, { type: blob.type });

            await processRecordedFile(fileFromBlob);
        } catch (err) {
            console.error(err);
            Swal.fire({ icon: "error", title: "Recording failed", text: err?.message || String(err) });
        } finally {
            if (recStream) {
                recStream.getTracks().forEach((track) => track.stop());
                recStream = null;
            }

            const btnRecord = document.getElementById("btnRecord");
            const btnStopRecord = document.getElementById("btnStopRecord");
            if (btnRecord) btnRecord.disabled = false;
            if (btnStopRecord) btnStopRecord.disabled = true;
            stopTimer();
        }
    };

    mediaRecorder.start(250);

    const btnRecord = document.getElementById("btnRecord");
    const btnStopRecord = document.getElementById("btnStopRecord");
    if (btnRecord) btnRecord.disabled = true;
    if (btnStopRecord) btnStopRecord.disabled = false;

    recUIActive = true;
    updateRecButton(MAX_REC_SEC);
    startTimer();
}

function stopRecording() {
    stopTimer();

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

function shouldRunRecordingObjectifier() {
    return Boolean(document.getElementById("objectifierMode")?.checked);
}

function showRecordingFeatureViewWhileObjectifierProcesses(data) {
    const hasPendingObjectifier = (data?.data || []).some((fileData) => {
        const status = fileData?.objectifier_job?.status;
        return !Array.isArray(fileData?.clusters) && ["queued", "running"].includes(status);
    });
    const objectifierMode = document.getElementById("objectifierMode");
    if (!hasPendingObjectifier || !objectifierMode?.checked) return;

    objectifierMode.checked = false;
    objectifierMode.dispatchEvent(new Event("change"));
}

async function processRecordedFile(file) {
    const spinner = document.getElementById("spinner");
    if (spinner) spinner.style.display = "block";
    getRecordingState().isLoading = true;

    try {
        const uploadClient = await getRecordingUploadClient();
        const hash = await uploadClient.computeSHA256(file);
        const exists = await uploadClient.checkFileExists(hash);
        let reuseCached = false;

        if (exists) {
            const reuseChoice = await Swal.fire({
                title: "Use cached analysis?",
                text: "A recording with identical content already exists.",
                icon: "question",
                showCancelButton: true,
                confirmButtonText: "Use Cached",
                cancelButtonText: "Reprocess",
            });
            reuseCached = reuseChoice.isConfirmed;
        }

        const n_fft = document.getElementById("window_length_selector").value;
        const overlap = getRecordingFeatureConfig().percentages[document.getElementById("window_overlap_selector").value];
        const normalize_audio = document.getElementById("normalize_button").checked;
        const apply_filter = document.getElementById("filter_button").checked;

        const formData = new FormData();
        formData.append("audio_files", file);
        formData.append("filenames", file.name);
        formData.append("hashes", hash);
        formData.append("n_fft", n_fft);
        formData.append("overlap", overlap);
        formData.append("normalize_audio", normalize_audio);
        formData.append("apply_filter", apply_filter);
        formData.append("save_json", true);
        formData.append("run_objectifier", shouldRunRecordingObjectifier());

        const response = await fetch(window.SoundSketcher.url(`/upload_wavs?reuse_cached=${reuseCached}`), {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`Upload failed (HTTP ${response.status})`);

        const data = await response.json();
        const currentDropArea = typeof dropArea !== "undefined" ? dropArea : document.getElementById("drop_area");
        if (currentDropArea) currentDropArea.style.display = "none";

        Swal.fire({
            icon: "success",
            title: "Recording processed.",
            position: "top-end",
            showConfirmButton: false,
            timer: 1500,
        });

        const state = getRecordingState();
        state.globalAudioData = data;
        window.SoundSketcherFeatureStatus?.updateFeatureExtractionStatus();
        window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
        showRecordingFeatureViewWhileObjectifierProcesses(data);
        state.globalFile = [file];
        const visualizationClient = await getRecordingVisualizationClient();
        await visualizationClient.visualizeAllFiles(state.globalFile);
    } catch (err) {
        console.error(err);
        Swal.fire({ icon: "error", title: "Processing error", text: err?.message || String(err) });
    } finally {
        if (spinner) spinner.style.display = "none";
        getRecordingState().isLoading = false;
    }
}

function initRecordingFeature() {
    const btnRec = document.getElementById("btnRecord");
    const btnStop = document.getElementById("btnStopRecord");
    const recCircleBtn = getRecCircleButton();

    if (!navigator.mediaDevices?.getUserMedia) {
        if (btnRec) btnRec.disabled = true;
        if (btnStop) btnStop.disabled = true;
        if (recCircleBtn) recCircleBtn.disabled = true;
        console.warn("getUserMedia not supported in this browser.");
        return;
    }

    btnRec?.addEventListener("click", async () => {
        try {
            await startRecording();
        } catch (err) {
            console.error(err);
            Swal.fire({ icon: "error", title: "Microphone access denied?", text: err?.message || String(err) });
        }
    });

    btnStop?.addEventListener("click", () => stopRecording());

    recCircleBtn?.addEventListener("click", () => {
        if (!recUIActive) {
            startRecording().catch((err) => {
                console.error(err);
                Swal.fire({ icon: "error", title: "Microphone access denied?", text: err?.message || String(err) });
            });
        } else {
            stopRecording();
        }
    });
}

window.SoundSketcherApp.onReady("recording controls", initRecordingFeature);
