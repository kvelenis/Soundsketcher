// --- Live recording → upload → visualize ---

let mediaRecorder = null;
let recordedChunks = [];
let recStartTs = 0;
let recTimerInt = null;
let recStream = null;

function formatMMSS(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function startTimer() {
    const timer = document.getElementById('recTimer');
    timer.style.display = 'inline';
    recStartTs = Date.now();
    recTimerInt = setInterval(() => {
        const elapsed = (Date.now() - recStartTs) / 1000;
        timer.textContent = formatMMSS(elapsed);
    }, 250);
}

function stopTimer() {
    const timer = document.getElementById('recTimer');
    clearInterval(recTimerInt);
    recTimerInt = null;
    timer.style.display = 'none';
    timer.textContent = '00:00';
}

async function getBestAudioMimeType() {
    const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',              // Safari fallback
        'audio/ogg;codecs=opus'
    ];
    for (const t of candidates) {
        if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return ''; // Let browser choose default
}

async function startRecording() {
    // Ask for mic
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = await getBestAudioMimeType();

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(recStream, mimeType ? { mimeType } : undefined);

    mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
        try {
            // Build Blob & File
            const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            const ext = (blob.type.includes('mp4') ? 'm4a'
                      : blob.type.includes('ogg') ? 'ogg'
                      : 'webm');
            // Give the file a deterministic-ish name (timestamped)
            const fname = `recording_${new Date().toISOString().replace(/[:.]/g,'-')}.${ext}`;
            const fileFromBlob = new File([blob], fname, { type: blob.type });

            await processRecordedFile(fileFromBlob);
        } catch (err) {
            console.error(err);
            Swal.fire({ icon: 'error', title: 'Recording failed', text: err?.message || String(err) });
        } finally {
            // cleanup stream tracks
            if (recStream) {
                recStream.getTracks().forEach(t => t.stop());
                recStream = null;
            }
            document.getElementById('btnRecord').disabled = false;
            document.getElementById('btnStopRecord').disabled = true;
            stopTimer();
        }
    };

    mediaRecorder.start(250); // gather dataevery 250ms
    document.getElementById('btnRecord').disabled = true;
    document.getElementById('btnStopRecord').disabled = false;
    startTimer();
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

// Reuse your upload/caching pipeline for a single recorded file
async function processRecordedFile(file) {
    const spinner = document.getElementById('spinner');
    spinner.style.display = 'block';

    try {
        // Compute hash and check cache
        const hash = await computeSHA256(file);
        const exists = await checkFileExists(hash);

        let reuseCached = false;
        if (exists) {
            const reuseChoice = await Swal.fire({
                title: "Use cached analysis?",
                text: "A recording with identical content already exists.",
                icon: "question",
                showCancelButton: true,
                confirmButtonText: "Use Cached",
                cancelButtonText: "Reprocess"
            });
            reuseCached = reuseChoice.isConfirmed;
        }

        // Send to the same endpoint you already use
        const fd = new FormData();
        fd.append('audio_files', file); // backend already expects 'audio_files'
        const response = await fetch(`upload_wavs?reuse_cached=${reuseCached}&run_objectifier=false`, {
            method: 'POST',
            body: fd
        });

        if (!response.ok) throw new Error(`Upload failed (HTTP ${response.status})`);
        const data = await response.json();

        dropArea.style.display = "none";
        Swal.fire({
            icon: "success",
            title: "Recording processed.",
            position: "top-end",
            showConfirmButton: false,
            timer: 1500
        });

        // For consistency with your existing code:
        globalAudioData = data;
        globalFile = [file]; // a File, like your drag&drop case
        await visualizeAllFiles(globalFile); //! Changed arguments for new version of function

    } catch (err) {
        console.error(err);
        Swal.fire({ icon: 'error', title: 'Processing error', text: err?.message || String(err) });
    } finally {
        spinner.style.display = 'none';
    }
}

// Wire up the buttons once (call this after DOM is ready)
function initRecordingFeature() {
    const btnRec = document.getElementById('btnRecord');
    const btnStop = document.getElementById('btnStopRecord');

    if (!navigator.mediaDevices?.getUserMedia) {
        btnRec.disabled = true;
        btnStop.disabled = true;
        console.warn('getUserMedia not supported in this browser.');
        return;
    }

    btnRec.addEventListener('click', async () => {
        try {
            await startRecording();
        } catch (err) {
            console.error(err);
            Swal.fire({ icon: 'error', title: 'Microphone access denied?', text: err?.message || String(err) });
        }
    });

    btnStop.addEventListener('click', () => stopRecording());
}

// Call this once on page load OR right after you show #controls
document.addEventListener('DOMContentLoaded', initRecordingFeature);