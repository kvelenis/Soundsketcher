from __future__ import annotations

import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.config import Settings


# ---------------------------------------------------------------------------
# MATLAB engine worker — one per process, lazy-started
# ---------------------------------------------------------------------------

class _MatlabWorker:
    """Daemon thread that owns a single MATLAB engine and processes jobs serially."""

    _instance: "_MatlabWorker | None" = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self._q: queue.Queue = queue.Queue()
        self._settings = settings
        self._thread = threading.Thread(target=self._run, daemon=True, name="matlab-worker")
        self._thread.start()

    @classmethod
    def get(cls, settings: Settings) -> "_MatlabWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    def submit(self, audio_path: str, timestamps: np.ndarray) -> dict[str, np.ndarray]:
        """Block until MATLAB returns results for this audio file."""
        result_container: dict[str, Any] = {}
        done = threading.Event()
        self._q.put((audio_path, timestamps, result_container, done))
        done.wait()
        if "error" in result_container:
            raise RuntimeError(result_container["error"])
        return result_container["results"]

    def _run(self) -> None:
        import matlab.engine  # noqa: F401 — available after matlabengine install

        eng = matlab.engine.start_matlab()
        s = self._settings

        if s.matlab_toolbox_dir and s.matlab_toolbox_dir.exists():
            eng.addpath(eng.genpath(str(s.matlab_toolbox_dir)), nargout=0)
        if s.matlab_scripts_dir and s.matlab_scripts_dir.exists():
            eng.addpath(eng.genpath(str(s.matlab_scripts_dir)), nargout=0)
            eng.cd(str(s.matlab_scripts_dir), nargout=0)

        eng.eval("clear all; rehash;", nargout=0)

        while True:
            audio_path, timestamps, container, done = self._q.get()
            try:
                container["results"] = _run_matlab_mir(eng, audio_path, timestamps)
            except Exception as exc:
                container["error"] = str(exc)
                n = len(timestamps)
                container["results"] = _zero_mir_results(n)
            finally:
                done.set()


def _zero_mir_results(n: int) -> dict[str, np.ndarray]:
    return {
        "MPS_roughness": np.zeros(n),
        "sharpness_Zwicker": np.zeros(n),
        "roughness_vassilakis": np.zeros(n),
        "weighted_spectral_centroid": np.zeros(n),
    }


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.array(x)
    arr = np.squeeze(arr)
    if arr.ndim > 1 and arr.shape[0] == 2:
        arr = arr[0]
    return arr


def _run_matlab_mir(eng: Any, audio_path: str, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    (
        mps_roughness, _rough_z, sharp_z, rough_v, _rough_s, loudness_sc_hz,
        time_mps, time_zwicker, time_v, _time_s
    ) = eng.roughnessTimeSeries(audio_path, nargout=10)

    def interp(vals: Any, t: Any) -> np.ndarray:
        v = _to_numpy(vals)
        t_arr = _to_numpy(t)
        return np.interp(timestamps, t_arr, v)

    return {
        "MPS_roughness": interp(mps_roughness, time_mps),
        "sharpness_Zwicker": interp(sharp_z, time_zwicker),
        "roughness_vassilakis": interp(rough_v, time_v),
        "weighted_spectral_centroid": interp(loudness_sc_hz, time_zwicker),
    }


# ---------------------------------------------------------------------------
# Sonic Annotator
# ---------------------------------------------------------------------------

def _run_sonic_annotator(wav_path: str, output_dir: str, sonic_annotator_dir: Path) -> None:
    import subprocess
    exe = str(sonic_annotator_dir / "sonic-annotator")
    periodicity_n3 = str(sonic_annotator_dir / "periodicity.n3")
    cmd = [exe, "-t", periodicity_n3, wav_path, "-w", "csv", "--csv-basedir", output_dir]
    subprocess.run(cmd, check=True, capture_output=True)


def _read_sonic_annotator_csvs(output_dir: str, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    import pandas as pd
    result = {"yin_periodicity": np.zeros_like(timestamps)}
    for fname in os.listdir(output_dir):
        if "periodicity" in fname and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(output_dir, fname), header=None)
            vals = np.interp(timestamps, df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
            result["yin_periodicity"] = np.maximum(vals, 0.0)
            break
    return result


# ---------------------------------------------------------------------------
# Per-tool extractors
# ---------------------------------------------------------------------------

def _extract_librosa(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict[str, np.ndarray]:
    import librosa

    f_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (50 <= f_bins) & (f_bins <= 200)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
    mag = np.abs(stft)
    fullness = np.sum(np.abs(np.diff(mag[mask, :], axis=1)) ** 2, axis=0) ** 0.5
    fullness = np.insert(fullness, 0, 0)

    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        frames = stft.shape[1]
        f0 = np.zeros(frames)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False)[0]
    return {"spectral_centroid": sc, "f0_librosa": f0, "fullness": fullness}


def _extract_aubio(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict[str, np.ndarray]:
    import aubio
    from scipy.ndimage import median_filter

    detector = aubio.pitch("yin", n_fft, hop_length, sr)
    detector.set_unit("Hz")
    detector.set_silence(-40)

    f0_list = []
    for i in range(0, len(y), hop_length):
        frame = y[i: i + hop_length].astype(np.float32)
        if len(frame) < hop_length:
            frame = np.pad(frame, (0, hop_length - len(frame)))
        p = detector(frame)[0]
        f0_list.append(float(p) if 50 <= p <= 500 else 0.0)

    f0 = median_filter(f0_list, size=3)
    t = np.arange(len(f0)) * (hop_length / sr)
    return {"aubio_f0": f0, "aubio_timestamps": t}


def _extract_crepe(y: np.ndarray, sr: int, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    import crepe
    t_c, freq_c, conf_c, _ = crepe.predict(y, sr, viterbi=True, model_capacity="tiny")
    return {
        "crepe_f0": np.interp(timestamps, t_c, freq_c),
        "crepe_confidence": np.interp(timestamps, t_c, conf_c),
    }


def _extract_mosqito(
    y: np.ndarray, sr: int, timestamps: np.ndarray, n_fft: int, hop_length: int
) -> dict[str, np.ndarray]:
    from mosqito.sq_metrics import sharpness_din_perseg, loudness_zwst_perseg

    noverlap = n_fft - hop_length
    sharp_vals, sharp_t = sharpness_din_perseg(signal=y, fs=sr, nperseg=n_fft, noverlap=noverlap, field_type="free")
    loud_vals, _N_spec, _bark, loud_t = loudness_zwst_perseg(signal=y, fs=sr, nperseg=n_fft, noverlap=noverlap, field_type="free")

    return {
        "loudness": np.interp(timestamps, loud_t, loud_vals),
        "sharpness": np.interp(timestamps, sharp_t, sharp_vals),
    }


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def run_legacy_extraction(
    wav_path: Path,
    settings: Settings,
    n_fft: int = 2048,
    hop_length: int = 1024,
    normalize_audio: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, float]]:
    """Run the full legacy feature extraction pipeline.

    Returns a list of per-timestamp feature dicts matching the 17-feature schema.
    Raises RuntimeError if required tools are missing.
    """
    import librosa

    def progress(stage: str) -> None:
        if progress_callback:
            progress_callback(stage)

    progress("loading_audio")
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        return []

    if normalize_audio:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak

    frames_needed = int(np.ceil((len(y) - n_fft) / hop_length)) + 1
    target_length = hop_length * (frames_needed - 1) + n_fft
    y = librosa.util.fix_length(y, size=target_length)
    timestamps = librosa.frames_to_time(range(frames_needed), sr=sr, hop_length=hop_length)

    # Sonic Annotator (subprocess)
    progress("sonic_annotator")
    sa_results: dict[str, np.ndarray] = {"yin_periodicity": np.zeros_like(timestamps)}
    if settings.sonic_annotator_dir and settings.sonic_annotator_dir.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _run_sonic_annotator(str(wav_path), tmpdir, settings.sonic_annotator_dir)
                sa_results = _read_sonic_annotator_csvs(tmpdir, timestamps)
            except Exception:
                pass

    progress("librosa")
    lib = _extract_librosa(y, sr, n_fft, hop_length)
    progress("aubio")
    aub = _extract_aubio(y, sr, n_fft, hop_length)
    progress("crepe")
    crp = _extract_crepe(y, sr, timestamps)
    progress("mosqito")
    mosq = _extract_mosqito(y, sr, timestamps, n_fft, hop_length)

    # Use aubio f0 as fallback when librosa pyin returns all zeros/nan
    yin_f0_librosa = lib["f0_librosa"]
    if np.all(yin_f0_librosa == 0):
        yin_f0_librosa = np.interp(timestamps, aub["aubio_timestamps"], aub["aubio_f0"])

    # MATLAB MIR features (daemon thread — starts on first call)
    progress("matlab")
    try:
        mir = _MatlabWorker.get(settings).submit(str(wav_path), timestamps)
    except Exception:
        mir = _zero_mir_results(len(timestamps))

    # Derived features
    progress("derived")
    raw_periodicity = sa_results["yin_periodicity"]
    yin_periodicity = np.where(raw_periodicity > 0.3, raw_periodicity, 0.0)
    weighted_sc = mir["weighted_spectral_centroid"]
    crepe_f0 = crp["crepe_f0"]
    crepe_conf = crp["crepe_confidence"]
    loudness = mosq["loudness"]

    def perceived_pitch(periodicity: np.ndarray, f0: np.ndarray, sc: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = np.clip((threshold - periodicity) / threshold, 0.0, 1.0)
        return f0 * (1.0 - p) + sc * p * 0.4

    perceived = perceived_pitch(raw_periodicity, crepe_f0, weighted_sc)
    loudness_period = loudness * (1.0 - yin_periodicity)
    loudness_pitch_conf = loudness * (1.0 - crepe_conf)

    n = len(timestamps)

    def _align(arr: np.ndarray) -> np.ndarray:
        if len(arr) < n:
            return np.pad(arr, (0, n - len(arr)))
        return arr[:n]

    rows = []
    sc_a = _align(lib["spectral_centroid"])
    wsc_a = _align(weighted_sc)
    crepe_f0_a = _align(crepe_f0)
    yin_f0_a = _align(yin_f0_librosa)
    perceived_a = _align(perceived)
    loudness_a = _align(loudness)
    lp_a = _align(loudness_period)
    lpc_a = _align(loudness_pitch_conf)
    sharp_a = _align(mosq["sharpness"])
    mps_a = _align(mir["MPS_roughness"])
    shz_a = _align(mir["sharpness_Zwicker"])
    rv_a = _align(mir["roughness_vassilakis"])
    full_a = _align(lib["fullness"])
    yinp_a = _align(yin_periodicity)
    cc_a = _align(crepe_conf)
    rawp_a = _align(raw_periodicity)

    for i, ts in enumerate(timestamps):
        rows.append({
            "timestamp": float(ts),
            "spectral_centroid": float(sc_a[i]),
            "weighted_spectral_centroid": float(wsc_a[i]),
            "crepe_f0": float(crepe_f0_a[i]),
            "yin_f0_librosa": float(yin_f0_a[i]),
            "perceived_pitch_f0_or_SC_weighted": float(perceived_a[i]),
            "loudness": float(loudness_a[i]),
            "loudness_periodicity": float(lp_a[i]),
            "loudness_pitchConf": float(lpc_a[i]),
            "sharpness": float(sharp_a[i]),
            "mir_mps_roughness": float(mps_a[i]),
            "mir_sharpness_zwicker": float(shz_a[i]),
            "mir_roughness_vassilakis": float(rv_a[i]),
            "fullness": float(full_a[i]),
            "yin_periodicity": float(yinp_a[i]),
            "crepe_confidence": float(cc_a[i]),
            "raw_periodicity": float(rawp_a[i]),
            # Aliases for frontend compatibility
            "f0_librosa": float(yin_f0_a[i]),
            "rms": float(loudness_a[i]),
        })

    return rows


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def legacy_dependency_report(settings: Settings) -> list[dict[str, Any]]:
    import importlib.util
    report = []

    for pkg in ("librosa", "numpy", "soundfile", "aubio", "crepe", "mosqito", "scipy", "matlab", "tensorflow"):
        available = importlib.util.find_spec(pkg) is not None
        report.append({"name": pkg, "available": available})

    sa = settings.sonic_annotator_dir
    report.append({
        "name": "sonic_annotator",
        "available": bool(sa and (sa / "sonic-annotator").exists()),
        "detail": str(sa) if sa else "not configured",
    })

    ml = settings.matlab_toolbox_dir
    report.append({
        "name": "matlab_toolbox",
        "available": bool(ml and ml.exists()),
        "detail": str(ml) if ml else "not configured",
    })

    ms = settings.matlab_scripts_dir
    report.append({
        "name": "matlab_scripts",
        "available": bool(ms and ms.exists()),
        "detail": str(ms) if ms else "not configured",
    })

    return report


def legacy_is_available(settings: Settings) -> bool:
    required = {"librosa", "numpy", "soundfile", "aubio", "crepe", "mosqito", "matlab", "tensorflow"}
    report = legacy_dependency_report(settings)
    by_name = {r["name"]: r["available"] for r in report}
    return all(by_name.get(name, False) for name in required)
