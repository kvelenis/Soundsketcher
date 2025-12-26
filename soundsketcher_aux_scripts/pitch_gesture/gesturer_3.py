import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ---------- PEAKS PER FRAME ----------
def get_peaks_per_frame(y, sr, n_fft=4096, hop_length=512, peak_rel_height=0.03, max_peaks=30):
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).T
    window = np.hanning(n_fft)
    freqs_axis = np.fft.rfftfreq(n_fft, 1.0 / sr)
    peaks_per_frame = []
    for frame in frames:
        spec = np.abs(np.fft.rfft(frame * window))
        if spec.size == 0 or np.max(spec) <= 0:
            peaks_per_frame.append((np.array([]), np.array([])))
            continue
        thr = np.max(spec) * peak_rel_height
        idx, props = find_peaks(spec, height=thr)
        p_freqs = freqs_axis[idx]
        p_mags  = spec[idx]
        if len(p_mags) > max_peaks:
            keep = np.argsort(p_mags)[::-1][:max_peaks]
            p_freqs = p_freqs[keep]
            p_mags  = p_mags[keep]
        # sort by frequency ascending (helps modal tracking)
        order = np.argsort(p_freqs)
        peaks_per_frame.append((p_freqs[order], p_mags[order]))
    times = np.arange(len(peaks_per_frame)) * (hop_length / sr)
    return peaks_per_frame, times

# ---------- STRETCHED-HARMONIC FIT (monophonic virtual pitch) ----------
def fit_stretched_harmonics(freqs_hz, mags=None, max_harm=8):
    freqs = np.asarray(freqs_hz, dtype=float)
    if mags is None:
        mags = np.ones_like(freqs)
    else:
        mags = np.asarray(mags, dtype=float)
    ok = np.isfinite(freqs) & (freqs > 0)
    freqs = freqs[ok]; mags = mags[ok]
    if freqs.size < 2:
        return None, None, 0.0
    # Map the K lowest freqs to k=1..K (greedy). Good enough for bells.
    order = np.argsort(freqs)
    f_sorted = freqs[order]
    K = int(min(len(f_sorted), max_harm))
    if K < 2:
        return None, None, 0.0
    k = np.arange(1, K + 1, dtype=float)
    fK = f_sorted[:K]
    w  = np.linspace(1.0, 0.6, K)  # favor lower “partials”
    x = np.log(k)
    y = np.log(fK)
    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    A = X.T @ W @ X
    b = X.T @ W @ y
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None, 0.0
    log_f0, s = float(beta[0]), float(beta[1])
    f0 = np.exp(log_f0)
    # R^2
    y_hat = X @ beta
    ss_res = np.sum(w * (y - y_hat) ** 2)
    y_bar  = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - y_bar) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    if not np.isfinite(f0) or f0 <= 10 or f0 >= 6000 or not np.isfinite(s):
        return None, None, 0.0
    return f0, s, float(r2)

def hybrid_gesture_pitch_monophonic(peaks_per_frame, r2_thresh=0.85):
    """
    Returns f_pitch (Hz) and r2 confidence per frame.
    If fit is poor (< r2_thresh), we still return f0 but with low r2 (plot will fade it).
    """
    T = len(peaks_per_frame)
    f_pitch = np.full(T, np.nan, dtype=float)
    r2_track = np.zeros(T, dtype=float)
    for i, (pf, pm) in enumerate(peaks_per_frame):
        if len(pf) < 2:
            continue
        f0, s, r2 = fit_stretched_harmonics(pf, pm)
        if f0 is not None:
            f_pitch[i] = f0
            r2_track[i] = r2
    return f_pitch, r2_track

# ---------- SMALL-GAP FILL & SMOOTHING ----------
def fill_small_gaps(f, times, max_gap_s=0.08):
    f = f.astype(float).copy()
    if len(times) < 2: return f
    dt = np.mean(np.diff(times))
    max_gap = int(round(max_gap_s / dt)) if dt > 0 else 0
    if max_gap <= 0: return f
    idx = np.arange(len(f))
    valid = ~np.isnan(f)
    if valid.sum() < 2: return f
    isn = ~valid
    edges = np.diff(np.pad(isn.astype(int), (1,1), constant_values=0))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0] - 1
    for s, e in zip(starts, ends):
        run = e - s + 1
        if run <= max_gap and s-1 >= 0 and e+1 < len(f) and valid[s-1] and valid[e+1]:
            f[s:e+1] = np.linspace(f[s-1], f[e+1], run+2)[1:-1]
    return f

def smooth_contour(f, times, sigma_s=0.025):
    if len(times) < 2: return f
    dt = np.mean(np.diff(times))
    if dt <= 0: return f
    sigma_frames = max(1e-6, sigma_s / dt)
    mask = np.isnan(f)
    if np.all(mask): return f
    fin = f.copy()
    if np.any(mask):
        valid = np.where(~mask)[0]
        fin[mask] = np.interp(np.where(mask)[0], valid, fin[valid])  # nearest-ish for filter
    fs = gaussian_filter1d(fin, sigma=sigma_frames)
    fs[mask] = np.nan
    return fs

# ---------- MODAL RIDGES (top-N resonant modes) ----------
def extract_modal_ridges(peaks_per_frame, times, topN=3, freq_tolerance_hz=40, smooth_sigma_s=0.02):
    """
    Track top-N modal ridges by nearest-frequency continuation.
    Returns modes: array (topN, T) with NaNs when unmatched.
    """
    T = len(peaks_per_frame)
    modes = np.full((topN, T), np.nan, dtype=float)

    # init from frame 0: take topN by magnitude
    if T == 0: return modes
    f0, m0 = peaks_per_frame[0]
    if len(f0) > 0:
        keep = np.argsort(m0)[::-1][:topN]
        base = np.sort(f0[keep])
        for k in range(min(topN, len(base))):
            modes[k, 0] = base[k]

    # continuation
    for t in range(1, T):
        f, mag = peaks_per_frame[t]
        for k in range(topN):
            prev = modes[k, t-1]
            if np.isnan(prev) or len(f) == 0:
                continue
            d = np.abs(f - prev)
            j = np.argmin(d)
            if d[j] <= freq_tolerance_hz:
                modes[k, t] = f[j]

    # smooth each ridge a bit
    for k in range(topN):
        modes[k] = smooth_contour(modes[k], times, sigma_s=smooth_sigma_s)
    return modes

# ---------- PLOT (monophonic with modal fallback) ----------
def plot_monophonic_with_modal(times, f_pitch, r2, modes, max_freq=2000,
                               r2_emphasis=0.85, save_path="monophonic_with_modal.png"):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Modal ridges: opacity increases when pitch confidence is low
    if modes is not None and modes.size > 0:
        inv_conf = 1.0 - np.clip(r2, 0.0, 1.0)  # 0..1
        for k in range(modes.shape[0]):
            mk = modes[k]
            for i in range(1, len(times)):
                f1, f2 = mk[i-1], mk[i]
                if np.isfinite(f1) and np.isfinite(f2):
                    alpha = 0.15 + 0.85 * inv_conf[i]   # strong when r2 is low
                    ax.plot(times[i-1:i+1], [f1, f2], color='gray', alpha=alpha, linewidth=1.5)

    # Pitch line: opacity follows confidence
    for i in range(1, len(times)):
        f1, f2 = f_pitch[i-1], f_pitch[i]
        if np.isfinite(f1) and np.isfinite(f2):
            alpha = 0.15 + 0.85 * np.clip(r2[i], 0.0, 1.0)
            ax.plot(times[i-1:i+1], [f1, f2], color='black', alpha=alpha, linewidth=2.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, max_freq)
    ax.set_title("Monophonic Perceptual Pitch with Modal View (opacity = confidence)")
    ax.grid(True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

# ---------- END-TO-END EXAMPLE ----------
if __name__ == "__main__":
    audio_path = "noises_test_with_diff_bandwidths_moving_Q.wav"  # <- set this
    y, sr = librosa.load(audio_path, sr=None)

    # 1) peaks & times
    peaks_per_frame, times = get_peaks_per_frame(
        y, sr,
        n_fft=4096, hop_length=512,
        peak_rel_height=0.03, max_peaks=40
    )

    # 2) monophonic virtual pitch + confidence
    f_pitch, r2 = hybrid_gesture_pitch_monophonic(peaks_per_frame, r2_thresh=0.85)

    # 3) gap-fill & smooth the monophonic line
    f_pitch = fill_small_gaps(f_pitch, times, max_gap_s=0.08)
    f_pitch = smooth_contour(f_pitch, times, sigma_s=0.025)

    # 4) modal ridges (top 3)
    modes = extract_modal_ridges(peaks_per_frame, times, topN=3, freq_tolerance_hz=40, smooth_sigma_s=0.02)

    # 5) plot & save
    plot_monophonic_with_modal(
        times, f_pitch, r2, modes,
        max_freq=2000,
        save_path="monophonic_with_modal.png"
    )