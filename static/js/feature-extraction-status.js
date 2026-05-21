(function () {
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLL_ATTEMPTS = 150; // 5 minutes max
    const activePolls = new Map();
    const pollAttempts = new Map();

    function getStatusElement() {
        return document.getElementById("objectifierStatus");
    }

    function getFloatingStatusElement() {
        return document.getElementById("featureProgressStatus");
    }

    function loadedFileRefs() {
        const audioData = window.SoundSketcher?.state?.globalAudioData;
        const files = audioData?.data || [];
        return files.map((fileData, index) => ({
            fileData,
            index,
            filename: audioData?.filename?.[index],
            hash: audioData?.hash?.[index],
        })).filter((ref) => ref.filename && ref.hash);
    }

    function isPendingFeatureJob(fileData) {
        return ["queued", "running"].includes(fileData?.feature_job?.status);
    }

    function fileKey(ref) {
        return `feat::${ref.hash}::${ref.filename}`;
    }

    function formatElapsed(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return "";
        if (seconds < 1) return "0s";
        if (seconds < 60) return `${Math.round(seconds)}s`;
        const minutes = Math.floor(seconds / 60);
        const rest = Math.round(seconds % 60).toString().padStart(2, "0");
        return `${minutes}:${rest}`;
    }

    function jobProgress(job) {
        const progress = Number(job?.progress);
        if (Number.isFinite(progress)) {
            return Math.max(0, Math.min(100, Math.round(progress)));
        }
        return job?.status === "done" ? 100 : 0;
    }

    function renderProgressStatus(element, { label, job, state = "processing" }) {
        if (!element) return;
        const progress = jobProgress(job);
        const message = job?.message || job?.stage || job?.status || "";
        const elapsed = formatElapsed(Number(job?.elapsed_seconds));
        element.dataset.state = state;
        element.textContent = "";

        const line = document.createElement("div");
        line.className = "objectifier-status__line";
        if (element.id === "featureProgressStatus") {
            const spinner = document.createElement("span");
            spinner.className = "feature-progress-status__spinner";
            spinner.setAttribute("aria-hidden", "true");
            line.appendChild(spinner);
        }
        const text = document.createElement("span");
        text.className = "objectifier-status__text";
        text.textContent = `${label}: ${progress}%${message ? ` — ${message}` : ""}${elapsed ? ` (${elapsed})` : ""}`;
        const value = document.createElement("span");
        value.className = "objectifier-status__percent";
        value.textContent = `${progress}%`;
        line.append(text, value);

        const bar = document.createElement("div");
        bar.className = "objectifier-status__bar";
        const fill = document.createElement("div");
        fill.className = "objectifier-status__bar-fill";
        fill.style.width = `${progress}%`;
        bar.appendChild(fill);

        element.append(line, bar);
    }

    function renderFeatureProgress({ label, job, state = "processing" }) {
        const pageSpinner = document.getElementById("spinner");
        if (pageSpinner) pageSpinner.style.display = "none";

        const panelStatus = getStatusElement();
        renderProgressStatus(panelStatus, { label, job, state });

        const floatingStatus = getFloatingStatusElement();
        if (floatingStatus) {
            floatingStatus.classList.remove("app-hidden");
            renderProgressStatus(floatingStatus, { label, job, state });
        }
    }

    function hideFloatingStatus() {
        const floatingStatus = getFloatingStatusElement();
        if (!floatingStatus) return;
        floatingStatus.classList.add("app-hidden");
        floatingStatus.textContent = "";
    }

    function renderFeatureFailure(message) {
        const panelStatus = getStatusElement();
        if (panelStatus) {
            panelStatus.textContent = message;
            panelStatus.dataset.state = "missing";
        }

        const floatingStatus = getFloatingStatusElement();
        if (floatingStatus) {
            floatingStatus.classList.remove("app-hidden");
            floatingStatus.textContent = message;
            floatingStatus.dataset.state = "missing";
        }
    }

    async function fetchFeatureJobStatus(ref) {
        const params = new URLSearchParams({ filename: ref.filename, audio_hash: ref.hash });
        const response = await fetch(window.SoundSketcher.url(`/feature_extraction_job_status?${params}`));
        if (!response.ok) throw new Error(`Feature job status failed (${response.status})`);
        return response.json();
    }

    async function applyFeatureJobStatus(ref, status) {
        if (!ref.fileData) return;

        if (status.status === "done") {
            renderFeatureProgress({
                label: "Rendering sketch",
                job: {
                    progress: 100,
                    message: "Rendering sketch",
                    elapsed_seconds: status.elapsed_seconds,
                },
                state: "processing",
            });

            ref.fileData.feature_job = null;
            if (Array.isArray(status.features) && status.features.length > 0) {
                ref.fileData.features = status.features;
            }
            // Wire up objectifier job so objectifier polling can start
            if (status.objectifier_job) {
                ref.fileData.objectifier_job = status.objectifier_job;
            }
            // Use the full visualization path if we have an audio_url for the file,
            // otherwise fall back to a plain button click.
            const audioUrl = ref.fileData.audio_url;
            if (audioUrl && typeof window.visualizeAllFiles === "function") {
                const urlFileList = [{ name: ref.filename, audio_url: audioUrl }];
                await window.visualizeAllFiles(urlFileList).catch((err) => {
                    console.warn("visualizeAllFiles failed after feature extraction, falling back:", err);
                    triggerRedraw();
                });
            } else {
                triggerRedraw();
            }
            hideFloatingStatus();
            // Hand off to objectifier polling
            window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
        } else {
            ref.fileData.feature_job = status;
        }
    }

    function stopPolling(key) {
        clearInterval(activePolls.get(key));
        activePolls.delete(key);
        pollAttempts.delete(key);
    }

    function triggerRedraw() {
        try {
            document.getElementById("submitButton")?.click();
        } catch (drawErr) {
            console.error("Canvas redraw failed after feature extraction:", drawErr);
        }
    }

    function pollFeatureJobStatus(ref) {
        const key = fileKey(ref);
        if (activePolls.has(key)) return;

        pollAttempts.set(key, 0);

        const poll = async () => {
            const attempts = (pollAttempts.get(key) || 0) + 1;
            pollAttempts.set(key, attempts);

            if (attempts > MAX_POLL_ATTEMPTS) {
                stopPolling(key);
                if (ref.fileData) ref.fileData.feature_job = { status: "failed", error: "timed out" };
                updateFeatureExtractionStatus();
                return;
            }

            let status;
            try {
                status = await fetchFeatureJobStatus(ref);
            } catch (err) {
                console.warn("Feature extraction status poll request failed", err);
                return;
            }

            // "unknown" means the server has no record of this job — skip this cycle but
            // keep polling (the worker may have restarted; the job may still be running).
            if (status.status === "unknown") {
                return;
            }

            await applyFeatureJobStatus(ref, status);
            if (["queued", "running"].includes(status.status)) {
                updateFeatureExtractionStatus();
            }

            if (!["queued", "running"].includes(status.status)) {
                stopPolling(key);
            }
        };

        activePolls.set(key, setInterval(poll, POLL_INTERVAL_MS));
        poll();
    }

    function startFeaturePolling() {
        loadedFileRefs()
            .filter((ref) => isPendingFeatureJob(ref.fileData))
            .forEach(pollFeatureJobStatus);
    }

    function updateFeatureExtractionStatus() {
        const refs = loadedFileRefs();
        const processingCount = refs.filter((ref) => isPendingFeatureJob(ref.fileData)).length;
        const failedCount = refs.filter((ref) => ref.fileData?.feature_job?.status === "failed").length;

        if (processingCount > 0) {
            const active = refs.find((ref) => isPendingFeatureJob(ref.fileData));
            renderFeatureProgress({
                label: `Feature extraction (${processingCount}/${refs.length})`,
                job: active?.fileData?.feature_job,
                state: "processing",
            });
        } else if (failedCount > 0) {
            renderFeatureFailure(`Feature extraction failed for ${failedCount} file${failedCount === 1 ? "" : "s"}`);
        } else {
            hideFloatingStatus();
        }

        startFeaturePolling();
    }

    window.SoundSketcherFeatureStatus = {
        updateFeatureExtractionStatus,
        isPendingFeatureJob,
        startFeaturePolling,
    };

    window.SoundSketcherApp.onReady("feature extraction status", () => {
        updateFeatureExtractionStatus();
    });
})();
