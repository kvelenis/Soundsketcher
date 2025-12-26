// --- Live recording → upload → visualize ---

const MAX_REC_SEC = 10;          // <= cap at 10 seconds

let mediaRecorder = null;
let recordedChunks = [];
let recStartTs = 0;
let recTimerInt = null;
let recStream = null;
let recStopTO = null;            // <= timeout to auto-stop at MAX_REC_SEC

// NEW: UI state for the header record button
let recUIActive = false;
const recCircleBtn = document.getElementById('recCircleBtn');

function formatMMSS(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// Helper to update the new header Record UI
function updateRecButton(countdown = null) {
    if (!recUIActive) {
        recCircleBtn.classList.remove('is-recording');
        recCircleBtn.textContent = ''; // plain red circle
    } else {
        recCircleBtn.classList.add('is-recording');
        // Show remaining seconds inside the rectangle
        if (countdown !== null) {
            recCircleBtn.textContent = String(countdown);
        }
    }
}

// Reworked to be a reverse countdown (from MAX_REC_SEC → 0) and auto-stop
function startTimer() {
    const timer = document.getElementById('recTimer');
    timer.style.display = 'inline';
    recStartTs = Date.now();

    // initialize display at full duration
    timer.textContent = formatMMSS(MAX_REC_SEC);

    // clear any existing timers just in case
    if (recTimerInt) clearInterval(recTimerInt);
    if (recStopTO) clearTimeout(recStopTO);

    // update 4x/sec for smoothness
    recTimerInt = setInterval(() => {
        const elapsed = (Date.now() - recStartTs) / 1000;
        const remaining = Math.max(0, MAX_REC_SEC - elapsed);
        timer.textContent = formatMMSS(Math.ceil(remaining));

        // ALSO update the header rectangle text
        updateRecButton(Math.ceil(remaining));
    }, 250);

    // hard stop exactly at MAX_REC_SEC
    recStopTO = setTimeout(() => {
        stopRecording(); // triggers onstop → processing
    }, MAX_REC_SEC * 1000);
}

function stopTimer() {
    const timer = document.getElementById('recTimer');
    clearInterval(recTimerInt);
    recTimerInt = null;
    clearTimeout(recStopTO);
    recStopTO = null;
    timer.style.display = 'none';
    timer.textContent = '00:00';

    // Reset header record UI to idle circle
    recUIActive = false;
    updateRecButton();
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

    if(!isLoading)
    {
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
    
                // Ensure UI resets
                stopTimer();
            }
        };
    
        mediaRecorder.start(250); // gather data every 250ms
        document.getElementById('btnRecord').disabled = true;
        document.getElementById('btnStopRecord').disabled = false;
    
        // Activate header UI and start the countdown
        recUIActive = true;
        updateRecButton(MAX_REC_SEC);
        startTimer(); // start reverse countdown + auto-stop
    }
    else
    {
        alert("Please wait for loading to finish");
    }
}

function stopRecording() {
    // stop the countdown immediately for better UX
    stopTimer();

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

// Reuse your upload/caching pipeline for a single recorded file
async function processRecordedFile(file) {

    const spinner = document.getElementById('spinner');
    spinner.style.display = 'block';
    // isLoading = true;

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
        const n_fft = document.getElementById("window_length_selector").value;
        const overlap = percentages[document.getElementById("window_overlap_selector").value];
        const normalize_audio = document.getElementById("normalize_button").checked;
        const apply_filter = document.getElementById("filter_button").checked;

        const fd = new FormData();
        fd.append('audio_files',file); // backend already expects 'audio_files'
        fd.append("filenames",file.name)
        fd.append("hashes",hash)
        fd.append('n_fft',n_fft);
        fd.append('overlap',overlap);
        fd.append('normalize_audio',normalize_audio);
        fd.append('apply_filter',apply_filter);
        fd.append('save_json',true);
        fd.append("run_objectifier",false);
        const response = await fetch("upload_wavs",
        {
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
        isLoading = false;
    }
}

// Wire up the buttons once (call this after DOM is ready)
function initRecordingFeature() {

    console.log("INIT REC");
    const btnRec = document.getElementById('btnRecord');
    const btnStop = document.getElementById('btnStopRecord');

    if (!navigator.mediaDevices?.getUserMedia) {
        btnRec.disabled = true;
        btnStop.disabled = true;
        if (recCircleBtn) recCircleBtn.disabled = true;
        console.warn('getUserMedia not supported in this browser.');
        return;
    }

    // Existing buttons (kept intact)
    btnRec.addEventListener('click', async () => {
        try {
            await startRecording();
        } catch (err) {
            console.error(err);
            Swal.fire({ icon: 'error', title: 'Microphone access denied?', text: err?.message || String(err) });
        }
    });
    btnStop.addEventListener('click', () => stopRecording());

    // NEW: Header Record UI toggles start/stop
    if (recCircleBtn) {
        recCircleBtn.addEventListener('click', () => {
            if (!recUIActive) {
                startRecording().catch(err => {
                    console.error(err);
                    Swal.fire({ icon: 'error', title: 'Microphone access denied?', text: err?.message || String(err) });
                });
            } else {
                stopRecording();
            }
        });
    }
}

// Call this once on page load OR right after you show #controls
document.addEventListener('DOMContentLoaded', initRecordingFeature);