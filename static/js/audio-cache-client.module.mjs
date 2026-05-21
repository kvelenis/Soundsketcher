function resolveAppEndpoint(path) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return window.SoundSketcher?.url?.(normalizedPath) || normalizedPath;
}

export async function listCachedFiles() {
    const response = await fetch(resolveAppEndpoint("/list_cached_files"));
    if (!response.ok) throw new Error("Could not load cached file list.");

    const payload = await response.json();
    return Array.isArray(payload.cached_files) ? payload.cached_files : [];
}

export async function loadCachedAudio(filename, hash) {
    const formData = new FormData();
    formData.append("filename", filename);
    formData.append("hash", hash);

    const response = await fetch(resolveAppEndpoint("/load_cached_audio"), {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error("Could not load cached audio.");
    return response.json();
}

export function toCachedFileList(loadCachedAudioPayload) {
    const filename = loadCachedAudioPayload.filename?.[0];
    const audioUrl = loadCachedAudioPayload.audio_url?.[0];
    if (!filename || !audioUrl) {
        throw new Error("Cached audio response is missing filename or audio_url.");
    }

    return [
        {
            name: filename,
            audio_url: audioUrl,
        },
    ];
}

export function resolveAudioUrl(audioUrl) {
    if (!audioUrl) return "";
    if (/^https?:\/\//.test(audioUrl)) return audioUrl;

    const normalizedUrl = audioUrl.startsWith("/") ? audioUrl : `/${audioUrl}`;
    return window.SoundSketcher?.url?.(normalizedUrl) || normalizedUrl;
}

export const audioCacheClient = {
    listCachedFiles,
    loadCachedAudio,
    toCachedFileList,
    resolveAudioUrl,
};

window.SoundSketcher?.registerAudioClient?.("cache", audioCacheClient);
window.SoundSketcherAudioCacheClient = audioCacheClient;
window.resolveCachedAudioUrl = resolveAudioUrl;
