export function computeSHA256(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function () {
            const shaObj = new jsSHA("SHA-256", "ARRAYBUFFER");
            shaObj.update(reader.result);
            resolve(shaObj.getHash("HEX"));
        };

        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(file);
    });
}

export async function checkFileExists(hash) {
    const response = await fetch(window.SoundSketcher.url(`/check_file_exists?audio_hash=${hash}`));
    if (!response.ok) return false;

    const result = await response.json();
    return result.features_exists;
}

function appendUploadOptions(formData, options) {
    formData.append("n_fft", options.n_fft);
    formData.append("overlap", options.overlap);
    formData.append("normalize_audio", options.normalize_audio);
    formData.append("apply_filter", options.apply_filter);
    formData.append("save_json", options.save_json ?? true);
    formData.append("run_objectifier", options.run_objectifier ?? false);
}

export async function prepareUploadRequest(files, options) {
    const formData = new FormData();
    let atLeastOneCached = false;

    for (const file of files) {
        const hash = await computeSHA256(file);
        const exists = await checkFileExists(hash);
        if (exists) {
            atLeastOneCached = true;
        }

        formData.append("audio_files", file);
        formData.append("filenames", file.name);
        formData.append("hashes", hash);
        appendUploadOptions(formData, options);
    }

    return { formData, atLeastOneCached };
}

export async function uploadAudioForm(formData, reuseCached) {
    const response = await fetch(window.SoundSketcher.url(`/upload_wavs?reuse_cached=${reuseCached}`), {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Upload failed (HTTP ${response.status})`);
    }

    return response.json();
}

export const audioUploadClient = {
    computeSHA256,
    checkFileExists,
    prepareUploadRequest,
    uploadAudioForm,
};

window.SoundSketcher?.registerAudioClient?.("upload", audioUploadClient);
window.SoundSketcherAudioUploadClient = audioUploadClient;
window.computeSHA256 = computeSHA256;
window.checkFileExists = checkFileExists;
