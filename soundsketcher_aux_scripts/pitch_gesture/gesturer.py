import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === Extract spectral peaks ===
def extract_peaks(frame, sr, n_fft):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    peak_indices, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.05)
    peak_freqs = freqs[peak_indices]
    peak_mags = spectrum[peak_indices]

    sorted_idx = np.argsort(peak_mags)[::-1]
    return peak_freqs[sorted_idx], peak_mags[sorted_idx]

# === Partial tracker (frame-to-frame peak continuation) ===
def partial_tracker(y, sr, n_fft=2048, hop_length=512, max_tracks=10, freq_tolerance=20):
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).T
    window = np.hanning(n_fft)
    tracks = [[] for _ in range(max_tracks)]

    prev_freqs = None
    prev_amps = None
    time_step = hop_length / sr

    for i, frame in enumerate(frames):
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        peak_indices, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.05)
        peak_freqs = freqs[peak_indices]
        peak_mags = spectrum[peak_indices]

        # Sort by magnitude, keep top-N
        sorted_idx = np.argsort(peak_mags)[::-1][:max_tracks]
        freqs = peak_freqs[sorted_idx]
        mags = peak_mags[sorted_idx]
        t = i * time_step

        for j in range(max_tracks):
            if j < len(freqs):
                f, amp = freqs[j], mags[j]
                if prev_freqs is not None:
                    dists = np.abs(prev_freqs - f)
                    if np.min(dists) < freq_tolerance:
                        tracks[j].append((t, f, amp))
                    else:
                        tracks[j].append((t, np.nan, 0))
                else:
                    tracks[j].append((t, f, amp))
            else:
                tracks[j].append((t, np.nan, 0))

        prev_freqs = freqs
        prev_amps = mags

    return tracks

def compute_frequency_derivatives(tracks):
    derivatives = []

    for track in tracks:
        freqs = np.array([f for t, f, a in track])
        times = np.array([t for t, f, a in track])
        dfdt = []

        for i in range(1, len(freqs)):
            if not np.isnan(freqs[i]) and not np.isnan(freqs[i - 1]):
                delta_f = freqs[i] - freqs[i - 1]
                delta_t = times[i] - times[i - 1]
                dfdt.append(delta_f / delta_t)
            else:
                dfdt.append(np.nan)

        derivatives.append(dfdt)

    return derivatives

def is_harmonic(freq, f0, max_deviation_cents=30):
    if f0 == 0 or np.isnan(freq):
        return False
    ratio = freq / f0
    nearest = round(ratio)
    if nearest == 0:
        return False
    cents_error = 1200 * np.log2(ratio / nearest)
    return abs(cents_error) < max_deviation_cents

def find_harmonic_groups_at_frame(freqs, amps, min_f0=50, max_f0=1000, step_hz=1.0):
    candidates = np.arange(min_f0, max_f0, step_hz)
    best_f0 = None
    best_score = -1
    best_group = []

    for f0 in candidates:
        group = []
        for i, freq in enumerate(freqs):
            if is_harmonic(freq, f0):
                group.append((i, freq, amps[i]))
        score = sum([amp for _, __, amp in group])
        if score > best_score and len(group) >= 2:
            best_f0 = f0
            best_score = score
            best_group = group

    return best_f0, best_group

def group_partials_by_harmonics(tracks, sr, hop_length):
    n_frames = len(tracks[0])
    grouped = []

    for i in range(n_frames):
        freqs_at_t = []
        amps_at_t = []
        for j in range(len(tracks)):
            t, f, a = tracks[j][i]
            if not np.isnan(f):
                freqs_at_t.append(f)
                amps_at_t.append(a)

        if len(freqs_at_t) < 2:
            grouped.append((None, []))
            continue

        f0, group = find_harmonic_groups_at_frame(freqs_at_t, amps_at_t)
        grouped.append((f0, group))

    return grouped

# === Estimate a single perceived pitch from partials ===
def estimate_perceived_pitch(tracks):
    n_frames = len(tracks[0])
    pitch_contour = []

    for i in range(n_frames):
        freqs_at_t = [tracks[j][i][1] for j in range(len(tracks)) if not np.isnan(tracks[j][i][1])]
        if len(freqs_at_t) == 0:
            pitch_contour.append(np.nan)
        else:
            # Option 1: Use lowest frequency (common psychoacoustic heuristic)
            pitch_contour.append(min(freqs_at_t))
    return pitch_contour

# === Plot partial tracks ===
def plot_tracks(tracks, save_path=None, max_freq=2000):
    for track in tracks:
        t, f = zip(*track)
        plt.plot(t, f, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Partial Tracks (Librosa + SciPy)")
    plt.ylim(0, max_freq)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Partial tracks plot saved to: {save_path}")
    plt.show()

def plot_tracks_with_amplitude(tracks, save_path=None, max_freq=2000):
    plt.figure(figsize=(12, 6))

    for track in tracks:
        t, f, a = zip(*track)
        t = np.array(t)
        f = np.array(f)
        a = np.array(a)

        # Normalize amplitude for line width
        a_norm = a / (np.max(a) + 1e-9)
        for i in range(1, len(t)):
            if not np.isnan(f[i - 1]) and not np.isnan(f[i]):
                plt.plot(t[i - 1:i + 1], f[i - 1:i + 1], linewidth=1 + 2 * a_norm[i], alpha=0.7)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, max_freq)
    plt.title("Partial Tracks with Amplitude-Weighted Lines")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Amplitude-weighted track plot saved to: {save_path}")
    plt.show()

def perceived_pitch_from_harmonic_groups(harmonic_groups):
    pitch_contour = []
    for f0, group in harmonic_groups:
        pitch_contour.append(f0 if f0 is not None else np.nan)
    return pitch_contour

# === Plot perceived pitch contour ===
def plot_pitch_contour(pitch_contour, hop_length, sr, save_path=None, max_freq=2000):
    times = np.arange(len(pitch_contour)) * (hop_length / sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, pitch_contour, label="Perceived Pitch", color='red', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Estimated Perceptual Pitch Contour")
    plt.ylim(0, max_freq)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pitch contour plot saved to: {save_path}")
    plt.show()

import matplotlib.cm as cm

def plot_harmonic_groups(tracks, harmonic_groups, save_path=None, max_freq=2000):
    plt.figure(figsize=(12, 6))
    n_tracks = len(tracks)
    n_frames = len(tracks[0])
    time_step = tracks[0][1][0] - tracks[0][0][0]  # Assumes regular spacing

    cmap = cm.get_cmap('tab10')

    for j in range(n_tracks):
        t_vals = []
        f_vals = []
        color_vals = []

        for i in range(n_frames):
            t, f, a = tracks[j][i]
            f0, group = harmonic_groups[i]

            # Check if this partial is part of the harmonic group at this frame
            is_in_group = any(abs(f - freq) < 1e-6 for _, freq, _ in group)

            t_vals.append(t)
            f_vals.append(f)
            color_vals.append('tab:red' if is_in_group else 'gray')

        # Plot with colored segments
        for k in range(1, len(t_vals)):
            if not np.isnan(f_vals[k - 1]) and not np.isnan(f_vals[k]):
                color = color_vals[k]
                alpha = 0.9 if color != 'gray' else 0.4
                plt.plot(t_vals[k - 1:k + 1], f_vals[k - 1:k + 1], color=color, alpha=alpha, linewidth=2 if color != 'gray' else 1)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Harmonic Group Visualization")
    plt.ylim(0, max_freq)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Harmonic group plot saved to: {save_path}")
    plt.show()

# === Main usage ===
if __name__ == "__main__":
    audio_path = "noises.wav"
    y, sr = librosa.load(audio_path, sr=None)

    tracks = partial_tracker(y, sr)
    plot_tracks_with_amplitude(tracks, save_path="partials_amplitude.png")

    dfdt_per_track = compute_frequency_derivatives(tracks)

    

    # Compute harmonic groups
    harmonic_groups = group_partials_by_harmonics(tracks, sr=sr, hop_length=512)

    # After harmonic_groups = ...
    pitch_contour = perceived_pitch_from_harmonic_groups(harmonic_groups)
    plot_pitch_contour(
        pitch_contour,
        hop_length=512,
        sr=sr,
        save_path="perceived_pitch_from_harmonics.png",
        max_freq=200
    )
    # Plot with harmonic group coloring
    plot_harmonic_groups(tracks, harmonic_groups, save_path="harmonic_groups.png", max_freq=2000)

    # Print average slope per track (rough motion indicator)
    for i, dfdt in enumerate(dfdt_per_track):
        slopes = [x for x in dfdt if not np.isnan(x)]
        if slopes:
            avg_slope = np.mean(slopes)
            print(f"Track {i}: avg df/dt = {avg_slope:.2f} Hz/s")