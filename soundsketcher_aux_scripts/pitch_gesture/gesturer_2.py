import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def extract_peaks(frame, sr, n_fft):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    peak_indices, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.05)
    peak_freqs = freqs[peak_indices]
    peak_mags = spectrum[peak_indices]

    sorted_idx = np.argsort(peak_mags)[::-1]
    return peak_freqs[sorted_idx], peak_mags[sorted_idx]

def group_partials(freqs, freq_tolerance_cents=30):
    groups = []
    assigned = np.full(len(freqs), False)

    for i, f in enumerate(freqs):
        if assigned[i] or f <= 0:
            continue
        group = [i]
        assigned[i] = True

        for j in range(i + 1, len(freqs)):
            if assigned[j] or freqs[j] <= 0:
                continue
            ratio = freqs[j] / f
            if ratio <= 0:
                continue
            nearest = round(ratio)
            if nearest == 0:
                continue
            cents = 1200 * np.log2(ratio / nearest)
            if abs(cents) <= freq_tolerance_cents:
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups

def partial_tracker_with_groups(y, sr, n_fft=2048, hop_length=512, max_groups=8):
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).T
    window = np.hanning(n_fft)

    FREQ = np.full((max_groups, len(frames)), np.nan)
    AMP = np.full((max_groups, len(frames)), np.nan)
    times = np.arange(len(frames)) * (hop_length / sr)

    prev_freqs = None

    for t_idx, frame in enumerate(frames):
        windowed = frame * window
        freqs, mags = extract_peaks(windowed, sr, n_fft)

        groups = group_partials(freqs)

        for g_idx, group in enumerate(groups[:max_groups]):
            main_peak = group[0]
            FREQ[g_idx, t_idx] = freqs[main_peak]
            AMP[g_idx, t_idx] = mags[main_peak]

        # Safe prev_freqs update
        means = []
        for g in range(max_groups):
            vals = FREQ[g, t_idx:t_idx+1]
            means.append(np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan)
        prev_freqs = np.array(means)

    return FREQ, AMP, times

def plot_group_matrix_colored(FREQ, AMP, delta_freq, times, title="Partial Groups Over Time", save_path=None):
    AMP_norm = AMP / np.nanmax(AMP)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-50, vmax=50)

    fig, ax = plt.subplots(figsize=(12, 6))

    for g_idx in range(FREQ.shape[0]):
        for t_idx in range(FREQ.shape[1]):
            if not np.isnan(FREQ[g_idx, t_idx]):
                ax.scatter(
                    times[t_idx], FREQ[g_idx, t_idx],
                    color=cmap(norm(delta_freq[g_idx, t_idx])),
                    alpha=AMP_norm[g_idx, t_idx]
                )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="Δ Frequency (Hz/s)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------
# 1) Choose the main group
# ---------------------------
def select_main_group(FREQ, AMP, times, w_lifespan=1.0, w_energy=1.0):
    """
    Score each group by a combo of lifespan (active seconds) and energy (sum amplitude).
    Returns index of the main group.
    """
    dt = np.mean(np.diff(times)) if len(times) > 1 else 0.0
    active = ~np.isnan(FREQ)
    lifespan_sec = active.sum(axis=1) * dt
    energy = np.nan_to_num(AMP, nan=0.0).sum(axis=1)

    # Normalize to avoid bias by scale
    ls_norm = (lifespan_sec / (lifespan_sec.max() + 1e-12)) if lifespan_sec.max() > 0 else lifespan_sec
    en_norm = (energy / (energy.max() + 1e-12)) if energy.max() > 0 else energy

    score = w_lifespan * ls_norm + w_energy * en_norm
    main_idx = int(np.nanargmax(score)) if np.any(np.isfinite(score)) else 0
    return main_idx, {"lifespan_sec": lifespan_sec, "energy": energy, "score": score}

# -----------------------------------------
# 2) Build contour & fill small gaps
# -----------------------------------------
def build_monophonic_contour(FREQ, group_idx, times, max_gap_s=0.100):
    """
    Extract F0 contour from chosen group and linearly interpolate gaps
    up to max_gap_s (leave larger gaps as NaN).
    """
    f = FREQ[group_idx].astype(float).copy()
    dt = np.mean(np.diff(times)) if len(times) > 1 else 0.0
    if dt == 0 or np.all(np.isnan(f)):
        return f

    max_gap_frames = int(round(max_gap_s / dt))

    # Indices where data exists
    idx = np.arange(len(f))
    valid = ~np.isnan(f)

    if valid.sum() < 2:
        return f  # not enough points to interpolate

    # Identify NaN runs and only inpaint small ones
    isn = ~valid
    # Find start/end of NaN runs
    edges = np.diff(np.pad(isn.astype(int), (1,1), constant_values=0))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0] - 1  # inclusive

    f_filled = f.copy()
    for s, e in zip(starts, ends):
        run_len = e - s + 1
        if run_len <= max_gap_frames:
            left = s-1
            right = e+1
            if left >= 0 and right < len(f) and valid[left] and valid[right]:
                # Linear interpolate
                f_filled[s:right] = np.linspace(f[left], f[right], run_len+2)[1:-1]

    return f_filled

# -----------------------------------------
# 3) Smooth (gesture > jitter)
# -----------------------------------------
def smooth_contour(f_contour, times, sigma_s=0.030):
    """
    Gaussian smooth with sigma in seconds (converted to frames).
    """
    if len(times) < 2:
        return f_contour
    dt = np.mean(np.diff(times))
    if dt <= 0:
        return f_contour
    sigma_frames = max(1e-6, sigma_s / dt)
    f = f_contour.copy()
    # Don’t smear NaNs: mask, smooth, then put NaNs back
    mask = np.isnan(f)
    if np.all(mask):
        return f
    # simple inpainting for smoothing only
    f_in = f.copy()
    if np.any(mask):
        # fill NaNs with nearest values for filter, then restore NaNs
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) >= 1:
            # nearest fill
            nearest = np.interp(np.arange(len(f)), valid_idx, f[valid_idx])
            f_in[mask] = nearest[mask]
    f_sm = gaussian_filter1d(f_in, sigma=sigma_frames)
    f_sm[mask] = np.nan
    return f_sm

# ------------------------------------------------
# 4) Optional: freq → Y (normalized log mapping)
# ------------------------------------------------
def freq_to_y_log(f_hz, f_min=50.0, f_max=2000.0, invert=True):
    """
    Map frequency to [0,1] on log scale for screen Y.
    invert=True -> higher freq at lower Y (music staff feel).
    """
    f = np.array(f_hz, dtype=float)
    y = np.full_like(f, np.nan, dtype=float)
    safe = (f > 0)
    if not np.any(safe):
        return y
    lo, hi = np.log10(f_min), np.log10(f_max)
    lo, hi = (lo, hi) if hi > lo else (0.0, 1.0)
    y[safe] = (np.log10(f[safe]) - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    if invert:
        y = 1.0 - y
    return y

# ------------------------------------------------
# 5) Convenience: full pipeline
# ------------------------------------------------
def make_monophonic_gesture_y(FREQ, AMP, times,
                              f_min=50, f_max=2000,
                              w_lifespan=1.0, w_energy=1.0,
                              max_gap_s=0.100, sigma_s=0.030,
                              return_debug=False):
    """
    Returns:
      y_norm (0..1), f_contour (Hz), main_group_idx, debug (optional)
    """
    main_idx, dbg_score = select_main_group(FREQ, AMP, times, w_lifespan, w_energy)
    f_raw = FREQ[main_idx]

    f_filled = build_monophonic_contour(FREQ, main_idx, times, max_gap_s=max_gap_s)
    f_smooth = smooth_contour(f_filled, times, sigma_s=sigma_s)
    y_norm = freq_to_y_log(f_smooth, f_min=f_min, f_max=f_max, invert=True)

    if return_debug:
        return y_norm, f_smooth, main_idx, {"score": dbg_score, "f_raw": f_raw, "f_filled": f_filled}
    return y_norm, f_smooth, main_idx

# === Example usage ===
if __name__ == "__main__":
    y, sr = librosa.load("polyphony_2.wav", sr=None)
    FREQ, AMP, times = partial_tracker_with_groups(y, sr)

    # Δ frequency
    delta_freq = np.diff(FREQ, axis=1) / (times[1] - times[0])
    delta_freq = np.hstack([np.zeros((FREQ.shape[0], 1)), delta_freq])

    plot_group_matrix_colored(
            FREQ, AMP, delta_freq, times,
            save_path="/mnt/ssd1/kvelenis/soundsketcher/soundsketcher_aux_scripts/pitch_gesture/group_matrix.png"
        )

    # 2) Generate monophonic gesture Y
    y_norm, f_pitch, main_group = make_monophonic_gesture_y(
        FREQ, AMP, times,
        f_min=50, f_max=2000,   # ui range
        w_lifespan=1.0, w_energy=1.0,
        max_gap_s=0.08,         # fill tiny dropouts up to 80 ms
        sigma_s=0.025           # ~25 ms smoothing
    )

    # 3) Plot and save
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, f_pitch, label=f"Main group {main_group} (Hz)", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 2000)
    ax.grid(True)
    ax.legend()

    save_path = "/mnt/ssd1/kvelenis/soundsketcher/soundsketcher_aux_scripts/pitch_gesture/monophonic_gesture.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gesture plot saved to {save_path}")