(function () {
    const POLL_INTERVAL_MS = 3000;
    const activePolls = new Map();

    function getObjectifierMode() {
        return document.getElementById("objectifierMode");
    }

    function getStatusElement() {
        return document.getElementById("objectifierStatus");
    }

    function getReadyButton() {
        return document.getElementById("objectifierReadyButton");
    }

    function loadedFiles() {
        return window.SoundSketcher?.state?.globalAudioData?.data || [];
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

    function hasObjectifierClusters(fileData) {
        return Array.isArray(fileData?.clusters) && fileData.clusters.length > 0;
    }

    function isProcessingJob(fileData) {
        return ["queued", "running"].includes(fileData?.objectifier_job?.status);
    }

    function fileKey(ref) {
        return `${ref.hash}::${ref.filename}`;
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
        const text = document.createElement("span");
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

    function getObjectifierAvailability() {
        const files = loadedFiles();
        if (!files.length) {
            return {
                state: "idle",
                text: "Objectifier data: no audio loaded",
                hasMissing: false,
            };
        }

        const processingCount = files.filter((fileData) => !hasObjectifierClusters(fileData) && isProcessingJob(fileData)).length;
        if (processingCount > 0) {
            const activeJob = files.find((fileData) => !hasObjectifierClusters(fileData) && isProcessingJob(fileData))?.objectifier_job;
            return {
                state: "processing",
                text: `Objectifier data: processing for ${processingCount}/${files.length} file${files.length === 1 ? "" : "s"}`,
                hasMissing: true,
                job: activeJob,
                processingCount,
                totalCount: files.length,
            };
        }

        const missingCount = files.filter((fileData) => !hasObjectifierClusters(fileData)).length;
        if (missingCount === 0) {
            return {
                state: "available",
                text: "Objectifier data: available",
                hasMissing: false,
            };
        }

        return {
            state: "missing",
            text: `Objectifier data: missing for ${missingCount}/${files.length} file${files.length === 1 ? "" : "s"}`,
            hasMissing: true,
        };
    }

    function setStatusElement(status) {
        const element = getStatusElement();
        if (!element) return;

        element.textContent = status.text;
        element.dataset.state = status.state;
        if (status.state === "processing" && status.job) {
            renderProgressStatus(element, {
                label: `Objectifier (${status.processingCount}/${status.totalCount})`,
                job: status.job,
                state: status.state,
            });
            return;
        }
    }

    function setReadyButton(status) {
        const button = getReadyButton();
        if (!button) return;

        const objectifierMode = getObjectifierMode();
        button.hidden = status.state !== "available" || Boolean(objectifierMode?.checked);
    }

    async function fetchObjectifierStatus(ref) {
        const params = new URLSearchParams({
            filename: ref.filename,
            audio_hash: ref.hash,
        });
        const response = await fetch(window.SoundSketcher.url(`/objectifier_status?${params}`));
        if (!response.ok) {
            throw new Error(`Objectifier status failed (${response.status})`);
        }
        return response.json();
    }

    function applyObjectifierStatus(ref, status) {
        if (!ref.fileData) return;
        ref.fileData.objectifier_job = status.status === "done" ? null : status;
        if (status.status === "done" && Array.isArray(status.clusters)) {
            ref.fileData.clusters = status.clusters;
            ref.fileData.cluster_labels = status.cluster_labels || {};
        }
    }

    function pollObjectifierStatus(ref) {
        const key = fileKey(ref);
        if (activePolls.has(key)) return;

        const poll = async () => {
            try {
                const status = await fetchObjectifierStatus(ref);
                applyObjectifierStatus(ref, status);
                updateObjectifierStatus(false);
                if (!["queued", "running"].includes(status.status)) {
                    clearInterval(activePolls.get(key));
                    activePolls.delete(key);
                }
            } catch (error) {
                console.warn("Objectifier status polling failed", error);
            }
        };

        activePolls.set(key, setInterval(poll, POLL_INTERVAL_MS));
        poll();
    }

    function startObjectifierPolling() {
        loadedFileRefs()
            .filter((ref) => !hasObjectifierClusters(ref.fileData) && isProcessingJob(ref.fileData))
            .forEach(pollObjectifierStatus);
    }

    function updateObjectifierStatus() {
        const status = getObjectifierAvailability();
        setStatusElement(status);
        setReadyButton(status);
        startObjectifierPolling();
        return status;
    }

    function showObjectifierView() {
        const status = updateObjectifierStatus();
        if (status.state !== "available") return;

        const objectifierMode = getObjectifierMode();
        const polygonMode = document.getElementById("linePolygonMode");
        if (polygonMode?.checked) {
            polygonMode.checked = false;
            polygonMode.dispatchEvent(new Event("change", { bubbles: true }));
        }
        if (objectifierMode && !objectifierMode.checked) {
            objectifierMode.checked = true;
            objectifierMode.dispatchEvent(new Event("change", { bubbles: true }));
        }
        document.getElementById("submitButton")?.click();
        updateObjectifierStatus();
    }

    function showMissingObjectifierNotice() {
        const status = updateObjectifierStatus();
        if (!getObjectifierMode()?.checked || !status.hasMissing) return false;

        if (window.Swal?.fire) {
            if (status.state === "processing") {
                Swal.fire({
                    icon: "info",
                    title: "Objectifier data is processing",
                    text: "The feature view is ready. Objectifier regions will become available when the background job finishes.",
                    confirmButtonText: "OK",
                });
                return true;
            }
            Swal.fire({
                icon: "info",
                title: "Objectifier data is missing",
                text: "Turn on Objectifier and press Recalculate to generate objectifier regions for the loaded audio.",
                confirmButtonText: "OK",
            });
        }
        return true;
    }

    window.SoundSketcherObjectifierStatus = {
        getObjectifierAvailability,
        hasObjectifierClusters,
        isProcessingJob,
        showMissingObjectifierNotice,
        showObjectifierView,
        startObjectifierPolling,
        updateObjectifierStatus,
    };

    window.SoundSketcherApp.onReady("objectifier status", () => {
        updateObjectifierStatus();
        getObjectifierMode()?.addEventListener("change", updateObjectifierStatus);
        getReadyButton()?.addEventListener("click", showObjectifierView);
    });
})();
