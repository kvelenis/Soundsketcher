# python main_with_event_detection.py

#! Imports
import re
import codecs
import psutil
import asyncio
import matlab.engine
from threading import Thread
import matplotlib.pyplot as plt
from multiprocessing import Queue,Manager
from starlette.middleware.base import BaseHTTPMiddleware
from soundsketcher_aux_scripts import feature_filtering as ff

import uvicorn
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi import Query

from concurrent.futures import ProcessPoolExecutor

from starlette.responses import RedirectResponse
from sklearn.manifold import TSNE
from typing import List
import numpy as np
import os
import pandas as pd
import json
import math
import uuid
from datetime import datetime
from typing import List
import time
from pathlib import Path
import hashlib
from typing import Optional

from soundsketcher_aux_scripts import sonic_annotator_call as sac
from soundsketcher_aux_scripts import clustering_objects as clobj
from soundsketcher_aux_scripts import matlab_extractor as matlextract
from soundsketcher_aux_scripts import json_creator as jsc
import torch
from transformers import pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PIL import Image
import clip  # Import the CLIP library

import librosa
import librosa.display

import soundfile as sf
import timbral_models

import mosqito
from scipy.signal import butter, lfilter, savgol_filter

import crepe

import pandas as pd
import numpy as np
import os

import librosa
import numpy as np
import pandas as pd
import os
import aubio
from scipy.ndimage import median_filter
from pydub import AudioSegment
from io import BytesIO


from mosqito.sq_metrics import sharpness_din_perseg, loudness_zwst_perseg  # Correct imports for features

from soundsketcher_aux_scripts import clustering_objects_with_wav2vec as obj

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = (BASE_DIR / "templates").resolve()

STATIC_DIR = (BASE_DIR / "static").resolve() 
STATIC_UPLOAD_DIR = (BASE_DIR / "static_uploads").resolve()
UPLOAD_DIR = (BASE_DIR / "user_data").resolve()

#UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "user_data")
#STATIC_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static_uploads")
os.makedirs(STATIC_UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/static_uploads", StaticFiles(directory=str(STATIC_UPLOAD_DIR)), name="static_uploads")
app.mount("/user_data", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

#! Added this to disable browser cache
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
app.add_middleware(NoCacheMiddleware)

# Initialize Jinja2Templates and point it to the templates directory
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Initial audio files
initial_audio_files = [
    # "melodic_synth_evolving.mp3",
    # "Monty_Python_bright_side_of_life.mp3",
    # "scary_door.mp3",
    # "tzitziki.mp3",
    # "water_drop.mp3"
]

# Preprocess initial audio files
initial_embeddings = []
initial_labels = []

for audio_file in initial_audio_files:
    file_path = os.path.join(STATIC_UPLOAD_DIR, audio_file)
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1)
    initial_embeddings.append(features.squeeze().numpy())
    initial_labels.append(audio_file)

initial_file_ids = initial_labels  # Use filenames as IDs

initial_embeddings = np.array(initial_embeddings)


def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ['1', 'true', 'yes', 'on']
    return False

def compute_audio_hash(audio_bytes):
    """
    Compute a SHA256 hash of the audio file's content to uniquely identify it.
    """
    return hashlib.sha256(audio_bytes).hexdigest()

def create_unique_folder():
    """
    Create a unique folder name based on timestamp and random string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = str(uuid.uuid4())[:8]  # Use the first 8 characters of a UUID
    return f"{timestamp}_{random_string}"

@app.head("/")
async def read_root_head():
    return Response(status_code=200)

from jinja2 import TemplateNotFound

@app.get("/")
async def index(request: Request):
    try:
        return templates.TemplateResponse(request, "index.html", {"request": request})
    except TemplateNotFound as e:
        print("🚨 Template not found:", e)
        return HTMLResponse(f"<h1>Template {e.name} not found in {TEMPLATES_DIR}</h1>", status_code=500)


# Serve the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

def ensure_proper_shape(segment):
    """
    Ensure the segment has the proper shape for Mosqito processing.
    """
    if len(segment.shape) == 1:  # If mono, add an axis to simulate stereo
        segment = np.expand_dims(segment, axis=0)
    return segment

import numpy as np
import librosa

def extract_hybrid_centroid_with_base_window(
    y, sr, n_fft=2048, hop_length=512,
    top_n_peaks=5, base_padding_hz=100, alpha=0.5,
    snr_threshold=1.2, canvas_range=(40.0, 8000.0)
) -> np.ndarray:
    """
    Computes a hybrid localized spectral centroid per frame,
    blending base frequency region and global top-N peaks.

    Returns:
        np.ndarray: Hybrid centroid per frame (Hz), clipped to canvas_range.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    num_frames = magnitude.shape[1]

    y_values_per_frame = []
    prev_base_freq = None

    for i in range(num_frames):
        frame = magnitude[:, i]
        if np.all(frame == 0):
            y_values_per_frame.append(canvas_range[0])
            continue

        peak_indices = np.argpartition(frame, -top_n_peaks)[-top_n_peaks:]
        peak_mags = frame[peak_indices]
        peak_freqs = freqs[peak_indices]

        sorted_idx = np.argsort(peak_freqs)
        base_freq = peak_freqs[sorted_idx[0]]
        base_mag = peak_mags[sorted_idx[0]]

        # --- Base region: around base_freq ± padding ---
        min_freq = base_freq - base_padding_hz
        max_freq = base_freq + base_padding_hz
        base_mask = (freqs >= min_freq) & (freqs <= max_freq)

        base_region = frame[base_mask]
        base_freqs = freqs[base_mask]
        if np.sum(base_region) > 0:
            base_centroid = np.sum(base_freqs * base_region) / (np.sum(base_region) + 1e-9)
        else:
            base_centroid = 0.0

        # --- Global centroid from top-N peaks, weighted ---
        weights = peak_mags ** 1.5
        global_centroid = np.sum(peak_freqs * weights) / (np.sum(weights) + 1e-9)

        # --- Adjust alpha based on base peak reliability ---
        if base_mag < np.mean(frame) * snr_threshold:
            adjusted_alpha = 0.2  # don’t trust base peak much
        elif prev_base_freq and abs(base_freq - prev_base_freq) > 50:
            adjusted_alpha = 0.3  # base peak not stable
        else:
            adjusted_alpha = alpha

        hybrid_centroid = adjusted_alpha * base_centroid + (1 - adjusted_alpha) * global_centroid
        hybrid_centroid = np.clip(hybrid_centroid, *canvas_range)

        y_values_per_frame.append(hybrid_centroid)
        prev_base_freq = base_freq

    return np.array(y_values_per_frame)

def extract_localized_centroid_from_peaks(y, sr, n_fft=2048, hop_length=512, top_n_peaks=3, pad_hz=400):
    """
    Compute a localized spectral centroid per frame by focusing on the region defined
    by the top-N spectral peaks, padded by ±pad_hz.

    Returns:
        np.ndarray: Array of localized spectral centroids per frame.
    """
    # STFT magnitude and frequency bins
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    num_bins, num_frames = magnitude.shape

    localized_centroids = []

    for frame_idx in range(num_frames):
        frame = magnitude[:, frame_idx]

        if np.all(frame == 0):
            localized_centroids.append(0.0)
            continue

        # Find top-N peak indices
        peak_indices = np.argpartition(frame, -top_n_peaks)[-top_n_peaks:]
        peak_freqs = freqs[peak_indices]
        min_freq = np.min(peak_freqs) - pad_hz
        max_freq = np.max(peak_freqs) + pad_hz

        # Constrain range within valid frequency limits
        min_freq = max(min_freq, freqs[0])
        max_freq = min(max_freq, freqs[-1])

        # Get mask for bins within the range
        mask = (freqs >= min_freq) & (freqs <= max_freq)

        if np.sum(mask) < 5:
            # If too few bins, fallback to full-band centroid
            centroid = np.sum(freqs * frame) / (np.sum(frame) + 1e-9)
        else:
            frame_region = frame[mask]
            freqs_region = freqs[mask]
            centroid = np.sum(freqs_region * frame_region) / (np.sum(frame_region) + 1e-9)

        localized_centroids.append(centroid)

    return np.array(localized_centroids)

def smooth_time_series(feature_array, window_length=11, polyorder=2):
    """
    Applies Savitzky-Golay smoothing to a 1D time series.

    Args:
        feature_array (np.ndarray): 1D array of feature values over time.
        window_length (int): Size of the moving window (must be odd and <= len(feature_array)).
        polyorder (int): Order of the polynomial fit.

    Returns:
        np.ndarray: Smoothed feature array.
    """
    # Ensure the window length is valid
    if len(feature_array) < window_length:
        window_length = len(feature_array) if len(feature_array) % 2 == 1 else len(feature_array) - 1
        if window_length < 3:
            return feature_array  # Not enough points to smooth

    return savgol_filter(feature_array, window_length=window_length, polyorder=polyorder)

from scipy.ndimage import gaussian_filter1d
import scipy.signal


def extract_harmonic_mapping_descriptor(
    y, sr, n_fft=2048, hop_length=512, top_n_peaks=5, harmonic_tolerance=0.2
):
    """
    Returns a Hz-based y-axis value per frame that reflects harmonic structure.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sample rate.
        n_fft (int): FFT size.
        hop_length (int): Hop size.
        top_n_peaks (int): Number of prominent peaks per frame.
        harmonic_tolerance (float): Tolerance when matching peaks to harmonic multiples.

    Returns:
        np.ndarray: y-axis values in Hz per frame (clipped to 40–8000 Hz).
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    num_frames = magnitude.shape[1]

    output = []

    for i in range(num_frames):
        frame = magnitude[:, i]
        if np.all(frame == 0):
            output.append(40.0)
            continue

        # Get top-N peaks
        peak_indices = np.argpartition(frame, -top_n_peaks)[-top_n_peaks:]
        peak_freqs = freqs[peak_indices]
        peak_mags = frame[peak_indices]

        # Sort by magnitude
        sorted_idx = np.argsort(-peak_mags)
        peak_freqs = peak_freqs[sorted_idx]
        peak_mags = peak_mags[sorted_idx]

        base_freq = peak_freqs[0]
        if base_freq < 40.0:
            output.append(40.0)
            continue

        harmonic_orders = []
        harmonic_weights = []

        for f, mag in zip(peak_freqs[1:], peak_mags[1:]):
            ratio = f / base_freq
            rounded = np.round(ratio)
            if np.abs(ratio - rounded) < harmonic_tolerance and rounded > 0:
                harmonic_orders.append(rounded)
                harmonic_weights.append(mag)

        if harmonic_orders:
            harmonic_descriptor = np.average(harmonic_orders, weights=harmonic_weights)
            perceptual_y = base_freq * harmonic_descriptor
        else:
            # Fallback to full spectral centroid
            perceptual_y = np.sum(freqs * frame) / (np.sum(frame) + 1e-9)

        # Clip to canvas-supported Hz range
        perceptual_y = np.clip(perceptual_y, 40.0, 8000.0)
        output.append(perceptual_y)

    return np.array(output)

#! Disabled old features
def extract_librosa_features(y,sr,n_fft,hop_length):

    #! Fullness: Alluri & Toiviainen
    f_low = 50
    f_high = 200
    f_bins = librosa.fft_frequencies(sr = sr,n_fft = n_fft)
    mask = (f_low <= f_bins) & (f_bins <= f_high)
    stft = librosa.stft(y,n_fft = n_fft,hop_length = hop_length,center = False)
    mag = np.abs(stft)
    norm = [1,2][1] # 0: 1-norm | 1: 2-norm
    fullness = np.sum(np.abs(np.diff(mag[mask,:],axis = 1))**norm,axis = 0)**(1/norm)
    fullness = np.insert(fullness,0,0) 

    # stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # magnitude = np.abs(stft)
    # stft_db = librosa.amplitude_to_db(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # spectral_flux = np.mean(np.diff(stft_db ** 2, axis=0), axis=0)
    # f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop_length)
    # f0 = np.nan_to_num(f0, nan=0.0)

    # # Spectral Peak Frequency
    # frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # peak_indices = np.argmax(magnitude, axis=0)  # one index per frame
    # spectral_peak = frequencies[peak_indices]    # frequency in Hz per frame

    # # Peak magnitude per frame
    # peak_magnitudes = magnitude[peak_indices, np.arange(magnitude.shape[1])]
    # mean_frame_magnitudes = np.mean(magnitude, axis=0) + 1e-9
    # # Prevent division by zero and log(0)
    # peak_magnitudes_safe = np.maximum(peak_magnitudes, 1e-9)
    # mean_frame_magnitudes_safe = np.maximum(mean_frame_magnitudes, 1e-9)
    # # Prominence in dB
    # ratio = peak_magnitudes_safe / mean_frame_magnitudes_safe
    # spectral_peak_prominence_db = 20 * np.log10(ratio)
    

    # # Normalize prominence to [0, 1] (linear mapping: 6–18 dB → 0–1)
    # def normalize_prominence(prom_db, min_db=6.0, max_db=18.0):
    #     p = (prom_db - min_db) / (max_db - min_db)
    #     return np.clip(p, 0.0, 1.0)

    # spectral_peak_prominence_norm = normalize_prominence(spectral_peak_prominence_db)
    # # Multi-peak centroid
    # # multipeak_centroid = extract_harmonic_mapping_descriptor(y, sr)
    # multipeak_centroid = extract_hybrid_centroid_with_base_window(y, sr, n_fft=n_fft, hop_length=hop_length, top_n_peaks=5, base_padding_hz=120, alpha=0.7)
    # print("f0 is ", f0)
    # voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    return {
        # 'zcr': librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0],
        'spectral_centroid': librosa.feature.spectral_centroid(y = y,sr = sr,n_fft = n_fft,hop_length = hop_length,center = False)[0], #! Added center = False
        # 'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        # 'rms': librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0],
        # 'spectral_flatness': librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0],
        # 'spectral_flux': spectral_flux,
        # 'f0_librosa': f0,
        # 'voiced_prob': voiced_prob,
        # 'spectral_peak': spectral_peak,
        # 'spectral_peak_prominence_db': spectral_peak_prominence_db,
        # 'spectral_peak_prominence_norm': spectral_peak_prominence_norm,
        # 'multipeak_centroid': multipeak_centroid,
        'fullness': fullness
    }

def extract_aubio_features(y, sr, n_fft, hop_length):
    hop_size = hop_length
    win_size = n_fft
    aubio_pitch_detector = aubio.pitch("yin", win_size, hop_size, sr)
    aubio_pitch_detector.set_unit("Hz")
    aubio_pitch_detector.set_silence(-40)
    fmin, fmax = 50, 500

    aubio_f0 = []
    for i in range(0, len(y), hop_size):
        frame = y[i:i + win_size].astype(np.float32)
        if len(frame) < win_size:
            frame = np.pad(frame, (0, win_size - len(frame)), mode='constant')
        pitch = aubio_pitch_detector(frame)[0]
        aubio_f0.append(pitch if fmin <= pitch <= fmax else 0.0)

    aubio_f0_smoothed = median_filter(aubio_f0, size=3)
    timestamps = np.arange(len(aubio_f0_smoothed)) * (hop_size / sr)
    return {'aubio_f0': aubio_f0_smoothed, 'aubio_timestamps': timestamps}

#! Disabled old features
def extract_crepe_features(y, sr, timestamps):
    time_crepe, frequency_crepe, confidence_crepe, _ = crepe.predict(y, sr, viterbi=True, model_capacity="tiny")
    crepe_f0 = np.interp(timestamps, time_crepe, frequency_crepe)
    confidence = np.interp(timestamps, time_crepe, confidence_crepe)
    return {
        'crepe_f0': crepe_f0,
        'crepe_confidence': confidence,
        # 'crepe_time': time_crepe
        }

#! Disabled old features
def extract_mosqito_features(y,sr,timestamps,n_fft,noverlap):
    sharpness_values,sharpness_time_values = sharpness_din_perseg(signal = y,fs = sr,nperseg = n_fft,noverlap = noverlap,field_type = 'free')
    loudness_values,N_spec,Bark_axis,loudness_time_values = loudness_zwst_perseg(signal = y,fs = sr,nperseg = n_fft,noverlap = noverlap,field_type = 'free')

    
    # bark_numerator = sum(i * j for i, j in zip(N_spec[:, 0].tolist(), Bark_axis.tolist()))
    # bark_denominator = sum(Bark_axis)
    # result = bark_numerator / bark_denominator if bark_denominator != 0 else 0
    # bark2freq = 600 * math.sinh(result / 6)

    interpolated = {
        'loudness': np.interp(timestamps, loudness_time_values, loudness_values),
        'sharpness': np.interp(timestamps, sharpness_time_values, sharpness_values)
    }

    return {**interpolated,
            # 'bark2freq': bark2freq
            }

#! Disabled old features
def extract_sonic_annotator_csvs(output_csv_dir, timestamps):
    result = {
        'yin_periodicity': np.zeros_like(timestamps),
        # 'f0_candidates': np.zeros_like(timestamps)
    }
    for file in os.listdir(output_csv_dir):
        file_path = os.path.join(output_csv_dir, file)
        if "periodicity" in file and file.endswith(".csv"):
            df = pd.read_csv(file_path)
            periodicity = np.interp(timestamps, df.iloc[:, 0], df.iloc[:, 1])
            periodicity = np.maximum(periodicity,0) #! Handling negative values option 1
            # periodicity = np.abs(periodicity) #! Handling negative values option 2
            result['yin_periodicity'] = periodicity
        # elif "candidates" in file and file.endswith(".csv"):
        #     timestamps_column, frequencies_column = [], []
        #     with open(file_path, "r") as f:
        #         for line in f:
        #             parts = line.strip().split(",")
        #             if len(parts) > 0:
        #                 try:
        #                     timestamp = float(parts[0])
        #                     timestamps_column.append(timestamp)
        #                     freqs = [float(freq) for freq in parts[1:] if freq.strip() != ""]
        #                     frequencies_column.append(freqs[0] if freqs else 0.0)
        #                 except ValueError:
        #                     continue
        #     result['f0_candidates'] = np.interp(timestamps, timestamps_column, frequencies_column)
    return result

#! Replacing 'run_parallel_extraction' for better performance | Disabled old features
def run_serial_extraction(wav_path,n_fft = 2048,hop_length = 1024,normalize_audio = False):

    y,sr = librosa.load(wav_path,sr = None,mono = True)
    if normalize_audio:
        y = y / np.max(np.abs(y))
    # frames_needed = int(np.ceil(len(y)/hop_length))
    # target_length = frames_needed*hop_length + n_fft
    #! This change ensures correct number of timestamps for librosa and other features
    frames_needed = int(np.ceil((len(y) - n_fft)/hop_length)) + 1
    target_length = hop_length*(frames_needed - 1) + n_fft
    y = librosa.util.fix_length(y,size = target_length)
    timestamps = librosa.frames_to_time(range(frames_needed),sr = sr,hop_length = hop_length)

    # Run Sonic Annotator
    folder_path = os.path.dirname(wav_path)
    sac.run_sonic_annotator(wav_path,folder_path)

    # Extract Features
    results = {
        'librosa': extract_librosa_features(y,sr,n_fft,hop_length),
        # 'aubio': extract_aubio_features(y,sr,n_fft,hop_length),
        'crepe': extract_crepe_features(y,sr,timestamps),
        'mosqito': extract_mosqito_features(y,sr,timestamps,n_fft,n_fft - hop_length),
        'sonic': extract_sonic_annotator_csvs(folder_path,timestamps)
    }

    # Extract MIR Features
    event = matlab_manager.Event()
    container = matlab_manager.dict()
    job = (wav_path,timestamps,container,event)
    matlab_queue.put(job)
    event.wait()
    mir_results = container['results']

    #! Calculating derived features here
    def perceivedPitchF0OrSC(periodicity,crepeF0,sc,threshold = 0.5,gamma = 1):
        # toggle = np.where(periodicity > threshold,1,0)
        # return crepeF0*toggle + sc*(1 - toggle)*0.6
        P = (threshold - periodicity)/(threshold)
        P = np.clip(P,0,1)
        P = P**gamma
        return crepeF0*(1 - P) + sc*P*0.4

    spectral_centroid = results['librosa']['spectral_centroid']
    weighted_spectral_centroid = mir_results['weighted_spectral_centroid']
    crepe_f0 = results['crepe']['crepe_f0']
    loudness = results['mosqito']['loudness']
    sharpness = results['mosqito']['sharpness']
    mir_mps_roughness = mir_results['MPS_roughness']
    mir_sharpness_zwicker = mir_results['sharpness_Zwicker']
    mir_roughness_vassilakis = mir_results['roughness_vassilakis']
    yin_periodicity = np.where(results['sonic']['yin_periodicity'] > 0.3,results['sonic']['yin_periodicity'],0)
    raw_periodicity = results['sonic']['yin_periodicity']
    crepe_confidence = results['crepe']['crepe_confidence']
    fullness = results['librosa']['fullness']
    
    perceived_pitch_f0_or_SC_weighted = perceivedPitchF0OrSC(raw_periodicity,crepe_f0,weighted_spectral_centroid)
    loudness_periodicity = loudness*(1 - yin_periodicity)
    loudness_pitchConf = loudness*(1 - crepe_confidence)

    # Merge all results into one flat dict
    features = {
        'timestamp': timestamps,
        'spectral_centroid': spectral_centroid,
        'weighted_spectral_centroid': weighted_spectral_centroid,
        'crepe_f0': crepe_f0,
        'perceived_pitch_f0_or_SC_weighted': perceived_pitch_f0_or_SC_weighted,
        'loudness': loudness,
        'loudness_periodicity': loudness_periodicity,
        'loudness_pitchConf': loudness_pitchConf,
        'sharpness': sharpness,
        'mir_mps_roughness': mir_mps_roughness,
        'mir_sharpness_zwicker': mir_sharpness_zwicker,
        'mir_roughness_vassilakis': mir_roughness_vassilakis,
        'yin_periodicity': yin_periodicity,
        'crepe_confidence': crepe_confidence,
        'fullness': fullness,
        'raw_periodicity': raw_periodicity,
        # **results['librosa'],
        # 'aubio_f0': np.interp(timestamps, results['aubio']['aubio_timestamps'], results['aubio']['aubio_f0']),
        # 'f0_candidates': results['sonic']['f0_candidates'],
        # 'mir_roughness_zwicker': mir_results['roughness_Zwicker'],
        # 'mir_roughness_sethares': mir_results['roughness_sethares'],
        # 'bark2freq': results['mosqito']['bark2freq'],
    }

    return features

#! Deprecated 'run_parallel_extraction'
# def run_parallel_extraction(y, sr, input_wav_path, output_csv_dir, n_fft, hop_length):
#     frames_needed = int(np.ceil(len(y) / hop_length))
#     target_length = frames_needed * hop_length + n_fft
#     y = librosa.util.fix_length(y, size=target_length)
#     timestamps = librosa.frames_to_time(range(frames_needed), sr=sr, hop_length=hop_length)

#     # Run Sonic Annotator (blocking)
#     sac.run_sonic_annotator(input_wav_path, output_csv_dir)

#     # Extract all features in parallel EXCEPT mir
#     with ProcessPoolExecutor() as executor:
#         futures = {
#             'librosa': executor.submit(extract_librosa_features, y, sr, n_fft, hop_length),
#             'aubio': executor.submit(extract_aubio_features, y, sr, n_fft, hop_length),
#             'crepe': executor.submit(extract_crepe_features, y, sr, timestamps),
#             'mosqito': executor.submit(extract_mosqito_features, y, sr, timestamps, n_fft),
#             'sonic': executor.submit(extract_sonic_annotator_csvs, output_csv_dir, timestamps)
#         }
#         results = {k: f.result() for k, f in futures.items()}

#     # Run MIR feature extraction synchronously (outside parallel execution)
#     mir_results = matlextract.extract_mir_features_for_file_interpolated(
#         input_wav_path,
#         '/usr/local/MATLAB/R2024b/mirtoolbox-main/MIRToolbox',  # MIRToolbox path
#         '/mnt/ssd1/kvelenis/soundsketcher/soundsketcher_aux_scripts',  # MATLAB script path
#         timestamps
#     )

#     # Merge all results into one flat dict
#     features = {
#         'timestamp': timestamps,
#         **results['librosa'],
#         'aubio_f0': np.interp(timestamps, results['aubio']['aubio_timestamps'], results['aubio']['aubio_f0']),
#         'crepe_f0': results['crepe']['crepe_f0'],
#         'crepe_confidence': results['crepe']['crepe_confidence'],
#         'yin_periodicity': results['sonic']['yin_periodicity'],
#         'f0_candidates': results['sonic']['f0_candidates'],
#         'loudness': results['mosqito']['loudness'],
#         'sharpness': results['mosqito']['sharpness'],
#         'bark2freq': results['mosqito']['bark2freq'],

#         # MIR interpolated features
#         'mir_mps_roughness': mir_results['MPS_roughness'],
#         'mir_roughness_zwicker': mir_results['roughness_Zwicker'],
#         'mir_sharpness_zwicker': mir_results['sharpness_Zwicker'],
#         'mir_roughness_vassilakis': mir_results['roughness_vassilakis'],
#         'mir_roughness_sethares': mir_results['roughness_sethares'],
#         'weighted_spectral_centroid': mir_results['weighted_spectral_centroid']
#     }

#     return features


def convert_to_serializable(data):
    """
    Recursively converts NumPy types in the data structure to Python native types.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy arrays to lists
    elif isinstance(data, np.generic):
        return data.item()  # Convert NumPy scalars to Python scalars
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(element) for element in data]
    else:
        return data  # Leave Python-native types unchanged

@app.get("/objectifier-upload-page", response_class=HTMLResponse)
async def upload_page(request: Request):
    """
    Serves the HTML form page for uploading an audio file.
    """
    return templates.TemplateResponse("objectifier-upload-page.html", {"request": request})

@app.post("/objectifier-upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    audio_path = f"uploaded_audio/{file.filename}"

    # Save the uploaded file
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Redirect to objectifier page for this audio
    return RedirectResponse(url=f"/objectifier/{file.filename}", status_code=303)

@app.get("/objectifier/{audio_file}", response_class=HTMLResponse)
async def get_plotly_page(request: Request, audio_file: str):
    """
    Run the objectifier on the audio file and render the Plotly visualization page.
    """
    audio_path = f"uploaded_audio/{audio_file}"  # or wherever you store your uploaded files

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        return HTMLResponse(
            content=f"<h1>Error: Audio file {audio_file} not found.</h1>", status_code=404
        )


    json_output_path = f"/mnt/ssd1/kvelenis/soundsketcher/clustering_destination_plots/{Path(audio_file).stem}_plotly_data.json"

    # Run objectifier and save its returned JSON to file
    json_data = obj.objectifier(audio_path)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # Save JSON to file
    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Now check if the output JSON was created
    json_file_path = f"clustering_destination_plots/{Path(audio_file).stem}_plotly_data.json"
    if not os.path.exists(json_file_path):
        return HTMLResponse(
            content=f"<h1>Error: Plotly data was not generated for {audio_file}</h1>", status_code=500
        )

    return templates.TemplateResponse("plotly_visualization.html", {
        "request": request,
        "audio_file": audio_file
    })

#THIS IS THE FUNCTIONAL STATIC SCRIPT

# @app.get("/objectifier/{audio_file}", response_class=HTMLResponse)
# async def get_plotly_page(request: Request, audio_file: str):
#     """
#     Render the Plotly visualization page for the given audio file.
#     """

#     #obj.objectifier(audio_path)
#     json_file_path = f"clustering_destination_plots/{audio_file}_plotly_data.json"
#     if not os.path.exists(json_file_path):
#         return HTMLResponse(
#             content=f"<h1>Error: Plotly data not found for {audio_file}</h1>", status_code=404
#         )
    
#     return templates.TemplateResponse("plotly_visualization.html", {"request": request, "audio_file": audio_file})

@app.get("/plotly-data/{audio_file}")
async def get_plotly_data(audio_file: str):
    """
    Serve the JSON data for Plotly visualization.
    """
    print(audio_file)
    json_file_path = f"/mnt/ssd1/kvelenis/soundsketcher/clustering_destination_plots/{Path(audio_file).stem}_plotly_data.json"
    try:
        with open(json_file_path, "r") as f:
            plot_data = json.load(f)
        return plot_data
    except FileNotFoundError:
        return {"error": "Plotly data not found"}


@app.get("/check_file_exists")
async def check_file_exists(audio_hash: str = Query(...)):
    """
    Checks if cached features and objectifier data exist for a given audio hash.
    """
    folder_path = os.path.join(UPLOAD_DIR, audio_hash)
    feature_path = os.path.join(folder_path, "features.json")
    objectifier_path = os.path.join(folder_path, "objectifier.json")

    return {
        "features_exists": os.path.exists(feature_path),
        "objectifier_exists": os.path.exists(objectifier_path)
    }

#! Updated for better performance (doesn't load whole JSON)
@app.get("/list_cached_files")
async def list_cached_files():
    """
    Returns a list of all cached audio files with their hashes, filenames, and accessible audio URLs.
    """
    cached_files = []
    pattern = re.compile(r'"audio_file"\s*:\s*"([^"]*)"')
    recording_pattern = re.compile(r"^recording_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z\.[a-zA-Z0-9]+$")

    for audio_hash in os.listdir(UPLOAD_DIR):
        
        folder_path = os.path.join(UPLOAD_DIR,audio_hash)
        features_path = os.path.join(folder_path,"features.json")
        objectifier_path = os.path.join(folder_path,"objectifier.json")

        if os.path.isfile(features_path):

            filename = "unknown.wav"

            # Search filename in objectifier.json
            if os.path.isfile(objectifier_path):
                try:
                    with open(objectifier_path,"r") as file:
                        for line in file:
                            match = pattern.search(line)
                            if match:
                                filename = match.group(1)
                                filename = codecs.decode(filename,'unicode_escape')
                                break
                except Exception as e:
                    print(f"Error reading objectifier.json: {e}")

            # Search filename in features.json 
            if filename == "unknown.wav":
                try:
                    with open(features_path,"r") as file:
                        for line in file:
                            match = pattern.search(line)
                            if match:
                                filename = match.group(1)
                                filename = codecs.decode(filename,'unicode_escape')
                                break
                except Exception as e:
                    print(f"Error reading features.json: {e}")

            # Search filename in folder
            if filename == "unknown.wav":
                tmp_string_length = 0
                for file in os.listdir(folder_path):
                    if(file.endswith((".wav",".mp3",".ogg")) and len(file) > tmp_string_length):
                        filename = file

            if recording_pattern.match(filename):
                continue

            # Check that the audio file exists
            audio_path = os.path.join(folder_path,filename)
            if os.path.isfile(audio_path):
                cached_files.append(
                {
                    "filename": filename,
                    "hash": audio_hash,
                    "audio_url": f"/user_data/{audio_hash}/{filename}"
                })
            else:
                print(audio_path,": No matching file")
        else:
            print(audio_hash,": No cached data")

    # Sort Files
    cached_files.sort(key = lambda x: x["filename"].lower())

    return JSONResponse(content = {"cached_files" : cached_files})

#! Previous version 'list_cached_files'
# @app.get("/list_cached_files")
# async def list_cached_files():
#     """
#     Returns a list of all cached audio files with their hashes, filenames, and accessible audio URLs.
#     """
#     cached_files = []

#     for audio_hash in os.listdir(UPLOAD_DIR):
#         folder_path = os.path.join(UPLOAD_DIR, audio_hash)
#         features_path = os.path.join(folder_path, "features.json")
#         objectifier_path = os.path.join(folder_path, "objectifier.json")

#         if os.path.isfile(features_path):
#             filename = "unknown.wav"

#             # Prefer audio_file from objectifier.json
#             if os.path.isfile(objectifier_path):
#                 try:
#                     with open(objectifier_path, "r") as f:
#                         obj_data = json.load(f)
#                         filename = obj_data.get("audio_file", filename)
#                 except Exception as e:
#                     print(f"Error reading objectifier.json: {e}")

#             # Fallback to features.json
#             if filename == "unknown.wav":
#                 try:
#                     with open(features_path, "r") as f:
#                         features_data = json.load(f)
#                         filename = features_data.get("Song", {}).get("filename", filename)
#                 except Exception as e:
#                     print(f"Error reading features.json: {e}")

#             # Check that the audio file exists
#             audio_path = os.path.join(folder_path, filename)
#             if os.path.isfile(audio_path):
#                 cached_files.append({
#                     "hash": audio_hash,
#                     "filename": filename,
#                     "audio_url": f"/user_data/{audio_hash}/{filename}"
#                 })

#     return JSONResponse(content={"cached_files": cached_files})

from fastapi import Form

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(vars(obj))
    elif is_nan_or_inf(obj): #! Added this, maybe change return value ?
        return 0
    else:
        return obj
    
#! New function to handle nan/inf values
def is_nan_or_inf(x):
    try:
        x = float(x)
        return np.isnan(x) or np.isinf(x)
    except (ValueError,TypeError):
        return False
    
# @app.post("/upload_wavs")
# async def upload_wavs(
#     wav_files: Optional[List[UploadFile]] = File(None),
#     filename: Optional[str] = Form(None),
#     hash: Optional[str] = Form(None),
#     run_objectifier: bool = Query(True),
#     reuse_cached: bool = Query(False)
# ):
#     all_features = []
#     print("reuse_cached: ", reuse_cached, "hash :", hash, "filename: ", filename)
#     if reuse_cached and hash and filename:
#         # Lookup cache directly
#         folder_path = os.path.join(UPLOAD_DIR, hash)
#         cached_feature_path = os.path.join(folder_path, "features.json")
#         cached_objectifier_path = os.path.join(folder_path, "objectifier.json")

#         if not os.path.exists(cached_feature_path):
#             return JSONResponse(status_code=404, content={"error": "Cached features not found."})

#         with open(cached_feature_path, "r") as f:
#             features_json = json.load(f)

#         file_result = {
#             "filename": filename,
#             "features": features_json["Song"]["features_per_timestamp"]
#         }

#         if run_objectifier and os.path.exists(cached_objectifier_path):
#             with open(cached_objectifier_path, "r") as f:
#                 objectifier_data = json.load(f)
#             if "clusters" in objectifier_data:
#                 file_result["clusters"] = objectifier_data["clusters"]

#         all_features.append(file_result)
#         return JSONResponse(content={
#             "files_processed": 1,
#             "features": all_features
#         })

#     # Fallback to normal processing
#     if not wav_files:
#         return JSONResponse(status_code=400, content={"error": "No audio files provided."})
#     all_features = []

#     for wav_file in wav_files:
#         raw_bytes = await wav_file.read()
#         audio_hash = compute_audio_hash(raw_bytes)

#         folder_path = os.path.join(UPLOAD_DIR, audio_hash)
#         os.makedirs(folder_path, exist_ok=True)

#         input_wav_path = os.path.join(folder_path, wav_file.filename)
#         with open(input_wav_path, "wb") as buffer:
#             buffer.write(raw_bytes)

#         # Check if cached features exist
#         cached_feature_path = os.path.join(folder_path, "features.json")
#         cached_objectifier_path = os.path.join(folder_path, "objectifier.json")

#         if reuse_cached and os.path.exists(cached_feature_path):
#             with open(cached_feature_path, "r") as f:
#                 features_json = json.load(f)
#         else:
#             # Extract features fresh
#             y, sr = librosa.load(input_wav_path, sr=None)
#             n_fft = 2048
#             hop_length = 2048
#             features = run_parallel_extraction(y, sr, input_wav_path, folder_path, n_fft, hop_length)
#             features_json = jsc.creator_librosa(features)
#             features_json = convert_to_serializable(features_json)

#             with open(cached_feature_path, "w") as f:
#                 json.dump(features_json, f, indent=2)

#         file_result = {
#             "filename": wav_file.filename,
#             "features": features_json["Song"]["features_per_timestamp"]
#         }

#         # Run objectifier
#         if run_objectifier:
#             if reuse_cached and os.path.exists(cached_objectifier_path):
#                 with open(cached_objectifier_path, "r") as f:
#                     objectifier_data = json.load(f)
#             else:
#                 objectifier_data = obj.objectifier(input_wav_path)
#                 with open(cached_objectifier_path, "w") as f:
#                     json.dump(objectifier_data, f, indent=2, default=convert_numpy)


#             if "clusters" in objectifier_data:
#                 file_result["clusters"] = objectifier_data["clusters"]

#         all_features.append(file_result)

#     return JSONResponse(content=sanitize_for_json({
#         "files_processed": len(wav_files),
#         "features": all_features
#     }))

#! Created to recalculate features
@app.post("/recalculate_features")
async def recalculate_features(
    filenames: List[str] = Form(None),
    hashes: List[str] = Form(None),
    n_fft: Optional[int] = Form(2048),
    overlap: Optional[float] = Form(0.5),
    normalize_audio: Optional[bool] = Form(False),
    apply_filter: Optional[bool] = Form(False),
    save_json: Optional[bool] = Form(False),
    run_objectifier: Optional[bool] = Form(False)
):

    data = []
    audio_urls = []
    raw_bytes = None
    loop = asyncio.get_running_loop()
    for hash,filename in zip(hashes,filenames):
        # result = process_audio_file(filename,hash,raw_bytes,n_fft,overlap,normalize_audio,run_objectifier,save_json)
        result = await loop.run_in_executor(executor,process_audio_file,filename,hash,raw_bytes,n_fft,overlap,normalize_audio,apply_filter,save_json,run_objectifier)
        data.append({"features": result.get("features"),"clusters": result.get("clusters")})
        audio_urls.append(f"/user_data/{hash}/{filename}")
    
    return JSONResponse(content = sanitize_for_json(
    {
        "files_processed": len(audio_urls),
        "filename": filenames,
        "hash": hashes,
        "audio_url": audio_urls,
        "data": data
    }))

#! Updated for parallel request handling
@app.post("/upload_wavs")
async def upload_wavs(
    audio_files: List[UploadFile] = File(None),
    filenames: Optional[List[str]] = Form(None),
    hashes: Optional[List[str]] = Form(None),
    reuse_cached: Optional[bool] = Query(False),
    n_fft: Optional[int] = Form(2048),
    overlap: Optional[float] = Form(0.5),
    normalize_audio: Optional[bool] = Form(False),
    apply_filter: Optional[bool] = Form(False),
    save_json: Optional[bool] = Form(False),
    run_objectifier: Optional[bool] = Form(False)
):
    
    num_of_files = len(audio_files)
    data = [None] * num_of_files
    audio_urls = [None] * num_of_files

    if reuse_cached and hashes and filenames:
        for index,(hash,filename) in enumerate(zip(hashes,filenames)):
            result = process_cached_file(hash)
            if result:
                data[index] = {"features": result.get("features"),"clusters": result.get("clusters")}
                audio_urls[index] = f"/user_data/{hash}/{filename}",

    loop = asyncio.get_running_loop()
    for index,audio_file in enumerate(audio_files):
        if data[index] is None:
            raw_bytes = await audio_file.read()
            filename = audio_file.filename
            hash = compute_audio_hash(raw_bytes)
            # result = process_audio_file(filename,hash,raw_bytes,n_fft,overlap,normalize_audio,run_objectifier,save_json)
            result = await loop.run_in_executor(executor,process_audio_file,filename,hash,raw_bytes,n_fft,overlap,normalize_audio,apply_filter,save_json,run_objectifier)
            data[index] = {"features": result.get("features"),"clusters": result.get("clusters")}
            audio_urls[index] = f"/user_data/{hash}/{filename}"
    
    return JSONResponse(content = sanitize_for_json(
    {
        "files_processed": num_of_files,
        "filename": filenames,
        "hash": hashes,
        "audio_url": audio_urls,
        "data": data
    }))

#! Previous version 'upload wavs'
# @app.post("/upload_wavs")
# async def upload_wavs(
#     audio_files: Optional[List[UploadFile]] = File(None),
#     filename: Optional[str] = Form(None),
#     hash: Optional[str] = Form(None),
#     run_objectifier: bool = Query("false"),
#     reuse_cached: bool = Query(False)
# ):
#     run_objectifier = str_to_bool(run_objectifier)
#     all_features = []
#     print("reuse_cached: ", reuse_cached, "hash :", hash, "filename: ", filename)

#     if reuse_cached and hash and filename:
#         folder_path = os.path.join(UPLOAD_DIR, hash)
#         cached_feature_path = os.path.join(folder_path, "features.json")
#         cached_objectifier_path = os.path.join(folder_path, "objectifier.json")

#         if not os.path.exists(cached_feature_path):
#             return JSONResponse(status_code=404, content={"error": "Cached features not found."})

#         with open(cached_feature_path, "r") as f:
#             features_json = json.load(f)

#         file_result = {
#             "filename": filename,
#             "features": features_json["Song"]["features_per_timestamp"]
#         }

#         if run_objectifier and os.path.exists(cached_objectifier_path):
#             with open(cached_objectifier_path, "r") as f:
#                 objectifier_data = json.load(f)
#             if "clusters" in objectifier_data:
#                 file_result["clusters"] = objectifier_data["clusters"]

#         all_features.append(file_result)
#         return JSONResponse(content={
#             "files_processed": 1,
#             "features": all_features
#         })

#     if not audio_files:
#         return JSONResponse(status_code=400, content={"error": "No audio files provided."})

#     for audio_file in audio_files:
#         raw_bytes = await audio_file.read()
#         audio_hash = compute_audio_hash(raw_bytes)
#         folder_path = os.path.join(UPLOAD_DIR, audio_hash)
#         os.makedirs(folder_path, exist_ok=True)

#         # Convert to WAV using pydub
#         audio = AudioSegment.from_file(BytesIO(raw_bytes))
#         wav_path = os.path.join(folder_path, f"{audio_file.filename}.converted.wav")
#         audio.export(wav_path, format="wav")

#         # Caching paths
#         cached_feature_path = os.path.join(folder_path, "features.json")
#         cached_objectifier_path = os.path.join(folder_path, "objectifier.json")

#         # Extract features
#         if reuse_cached and os.path.exists(cached_feature_path):
#             with open(cached_feature_path, "r") as f:
#                 features_json = json.load(f)
#         else:
#             y, sr = librosa.load(wav_path, sr=None)
#             n_fft = 2048
#             hop_length = 2048
#             features = run_parallel_extraction(y, sr, wav_path, folder_path, n_fft, hop_length)
#             features_json = jsc.creator_librosa(features)
#             features_json = convert_to_serializable(features_json)

#             with open(cached_feature_path, "w") as f:
#                 json.dump(features_json, f, indent=2)

#         file_result = {
#             "filename": audio_file.filename,
#             "features": features_json["Song"]["features_per_timestamp"]
#         }

#         # Run objectifier
#         if run_objectifier:
#             if reuse_cached and os.path.exists(cached_objectifier_path):
#                 with open(cached_objectifier_path, "r") as f:
#                     objectifier_data = json.load(f)
#             else:
#                 objectifier_data = obj.objectifier(wav_path)
#                 with open(cached_objectifier_path, "w") as f:
#                     json.dump(objectifier_data, f, indent=2, default=convert_numpy)

#             if "clusters" in objectifier_data:
#                 file_result["clusters"] = objectifier_data["clusters"]

#         all_features.append(file_result)

#     return JSONResponse(content=sanitize_for_json({
#         "files_processed": len(audio_files),
#         "features": all_features
#     }))

# THIS IS THE CODE THAT USES UNIQUE FOLDER CREATION

# @app.post("/upload_wavs")
# async def upload_wavs(
#     wav_files: List[UploadFile] = File(...),
#     run_objectifier: bool = Query(True)  # Optional query flag
# ):
#     unique_folder = create_unique_folder()
#     folder_path = os.path.join(UPLOAD_DIR, unique_folder)
#     os.makedirs(folder_path)

#     input_wav_dir = os.path.join(folder_path, "input_wavs")
#     output_csv_dir = os.path.join(folder_path, "output_csv")
#     objectifier_json_dir = os.path.join(folder_path, "json_objectifier")
#     os.makedirs(input_wav_dir)
#     os.makedirs(output_csv_dir)
#     os.makedirs(objectifier_json_dir)

#     all_features = []

#     for wav_file in wav_files:
#         input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
#         with open(input_wav_path, "wb") as buffer:
#             buffer.write(await wav_file.read())

#         y, sr = librosa.load(input_wav_path, sr=None)
#         n_fft = 2048
#         hop_length = 2048

#         features = run_parallel_extraction(y, sr, input_wav_path, output_csv_dir, n_fft, hop_length)
#         print("FEATURES", features)
#         json_data = jsc.creator_librosa(features)
#         json_data = convert_to_serializable(json_data)

#         file_result = {
#             "filename": wav_file.filename,
#             "features": json_data["Song"]["features_per_timestamp"]
#         }


#         # Objectifier
#         if run_objectifier:
#             objectifier_json = f"{objectifier_json_dir}/{Path(wav_file.filename).stem}_plotly_data.json"
#             # Run objectifier and save its returned JSON to file
#             json_data = obj.objectifier(input_wav_path)

#             # Ensure output folder exists
#             os.makedirs(os.path.dirname(objectifier_json_dir), exist_ok=True)

#             # Save JSON to file
#             with open(objectifier_json, "w") as f:
#                 json.dump(json_data, f, indent=2)

#             # Include only the 'clusters' key in the response
#             if "clusters" in json_data:
#                 file_result["clusters"] = json_data["clusters"]
        
#         all_features.append(file_result)

#     return JSONResponse(content={
#         "files_processed": len(wav_files),
#         "features": all_features
#     })

#! Created for calls by backend (upload_wavs -> reuse_cache)
def process_cached_file(hash):

    result = {}
    folder_path = os.path.join(UPLOAD_DIR,hash)
    cached_feature_path = os.path.join(folder_path,"features.json")
    cached_objectifier_path = os.path.join(folder_path,"objectifier.json")

    if os.path.exists(cached_feature_path):
        with open(cached_feature_path,"r") as file:
            features_json = json.load(file)
            result["features"] = features_json["Song"]["features_per_timestamp"]

        if os.path.exists(cached_objectifier_path):
            with open(cached_objectifier_path,"r") as file:
                objectifier_data = json.load(file)
                if "clusters" in objectifier_data:
                    result["clusters"] = objectifier_data["clusters"]    

    return result

#! Updated to wrap 'process_cached_file' for calls by frontend (examples)
@app.post("/load_cached_audio")
def load_cached_audio(
    filename: str = Form(...),
    hash: str = Form(...)
):
    result = process_cached_file(hash)
    if result:
        data = [{"features": result.get("features"),"clusters": result.get("clusters")}]
        return JSONResponse(content = sanitize_for_json(
        {
            "files_processed": 1,
            "filename": [filename],
            "hash": [hash],
            "audio_url": [f"/user_data/{hash}/{filename}"],
            "data": data
        }))
    else:
        return JSONResponse(status_code = 404,content = {"Error": "cached features not found."})

#! Previous version 'load_cached_audio'
# @app.post("/load_cached_audio")
# async def load_cached_audio(
#     filename: str = Form(...),
#     hash: str = Form(...),
#     run_objectifier: bool = Query("false")
# ):
#     run_objectifier = str_to_bool(run_objectifier)
#     folder_path = os.path.join(UPLOAD_DIR, hash)
#     cached_feature_path = os.path.join(folder_path, "features.json")
#     cached_objectifier_path = os.path.join(folder_path, "objectifier.json")

#     if not os.path.exists(cached_feature_path):
#         return JSONResponse(status_code=404, content={"error": "Cached features not found."})

#     with open(cached_feature_path, "r") as f:
#         features_json = json.load(f)

#     file_result = {
#         "filename": filename,
#         "features": features_json["Song"]["features_per_timestamp"]
#     }

#     if run_objectifier and os.path.exists(cached_objectifier_path):
#         with open(cached_objectifier_path, "r") as f:
#             objectifier_data = json.load(f)
#         if "clusters" in objectifier_data:
#             file_result["clusters"] = objectifier_data["clusters"]

#     return JSONResponse(content={
#         "files_processed": 1,
#         "features": [file_result],
#         "audio_url": f"/user_data/{hash}/{filename}",
#         "filename": filename
#     })

#! Created to be called by the executor
def process_audio_file(filename,hash,raw_bytes = None,n_fft = 2048,overlap = 0.5,normalize_audio = False,apply_filter = False,save_json = False,run_objectifier = False):

    result = {}
    folder_path = os.path.join(UPLOAD_DIR,hash)
    os.makedirs(folder_path,exist_ok=True)
    wav_filename = filename.rsplit('.', 1)[0] + ".wav" #! Added this here
    wav_path = os.path.join(folder_path,wav_filename)

    if raw_bytes is not None:
        audio = AudioSegment.from_file(BytesIO(raw_bytes))
        audio.export(wav_path,format = "wav")
        
    # Extract Features
    hop_length = n_fft - round(n_fft*overlap)
    features = run_serial_extraction(wav_path,n_fft,hop_length,normalize_audio)
    # ff.plot_features(features,overlap,os.path.join(os.path.dirname(os.path.abspath(__file__)),"plots")) #! Debugging
    if(apply_filter):
        filter_length = ff.calculate_optimal_length(overlap)
        features = ff.filter_features(features,filter_length)
    features_json = jsc.creator_librosa(features)
    features_json = convert_to_serializable(features_json)
    result["features"] = features_json["Song"]["features_per_timestamp"]
    
    if save_json:
        features_json["Song"]["audio_file"] = wav_filename
        features_json["Song"] = {key: features_json["Song"][key] for key in reversed(features_json["Song"])}
        cached_feature_path = os.path.join(folder_path,"features.json")
        with open(cached_feature_path,"w") as file:
            json.dump(features_json,file,indent = 2)

    # Run Objectifier
    if run_objectifier:

        objectifier_data = obj.objectifier(wav_path)
        if "clusters" in objectifier_data:
            result["clusters"] = objectifier_data["clusters"]

        if save_json:
            objectifier_path = os.path.join(folder_path,"objectifier.json")
            with open(objectifier_path,"w") as file:
                json.dump(objectifier_data,file,indent = 2,default = convert_numpy)

    return result

# Route to serve the audio_features.json file from static/sound directory
@app.get("/get_audio_features")
async def get_audio_features():
    try:
        # Replace this with your logic to read and return audio_features.json data
        audio_features_data = {"example": "data"}
        return audio_features_data
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.post("/analyze_wav", response_class=JSONResponse)
async def analyze_wav(request: Request, wav_file: UploadFile = File(...), candidate_labels: str = Form(...)):
    unique_folder = create_unique_folder()
    folder_path = os.path.join(UPLOAD_DIR, unique_folder)
    os.makedirs(folder_path)

    input_wav_dir = os.path.join(folder_path, "input_wav")
    os.makedirs(input_wav_dir)

    input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
    with open(input_wav_path, "wb") as buffer:
        buffer.write(await wav_file.read())

    # Convert candidate labels to a list
    candidate_labels_list = [label.strip() for label in candidate_labels.split(",")]

    # Load audio file and convert to mono
    audio, sr = librosa.load(input_wav_path, sr=None, mono=True)

    # Initialize the zero-shot audio classification pipeline
    audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general")

    # Define window parameters
    window_size = 2 * sr  # 2 second window
    overlap = 1 * sr  # 1 second overlap
    step = window_size - overlap

    # Initialize a dictionary to store scores
    scores_dict = {label: [] for label in candidate_labels_list}
    time_points = []

    # Iterate through the audio with overlapping windows
    for start in range(0, len(audio) - window_size + 1, step):
        window = audio[start:start + window_size]
        output = audio_classifier(window, candidate_labels=candidate_labels_list)
        
        for result in output:
            scores_dict[result['label']].append(result['score'])
        
        time_points.append(start / sr)

    # Prepare the plot data for Plotly
    plot_data = {}
    for label in scores_dict:
        plot_data[label] = {
            "time_points": time_points,
            "scores": scores_dict[label]
        }

    # Return the plot data as JSON
    return JSONResponse(content=plot_data)

@app.get("/wav2vec", response_class=HTMLResponse)
async def wav2vec(request: Request):
    return templates.TemplateResponse("wav2vec.html", {"request": request})

@app.post("/upload_wav2vec")
async def upload_wav2vec(file: UploadFile = File(...)):
    file_path = os.path.join(STATIC_UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1)
    new_embedding = features.squeeze().numpy()
    

    # Prepare embeddings data for JSON output
    embedding_json = {
        "file_name": file.filename,
        "embedding": new_embedding.tolist(),
        "sampling_rate": sr,
    }

    # Save embeddings to a JSON file
    json_file_path = os.path.splitext(file_path)[0] + "_embedding.json"
    with open(json_file_path, "w") as json_file:
        json.dump(embedding_json, json_file, indent=4)

    combined_embeddings = np.vstack([initial_embeddings, new_embedding])
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(combined_embeddings)

    plot_data = {
        "x": embeddings_2d[:, 0].tolist(),
        "y": embeddings_2d[:, 1].tolist(),
        "text": initial_labels + [file.filename],
        "file_ids": initial_file_ids + [file.filename],
        "new_audio_url": f"/static_uploads/{file.filename}"
    }

    return JSONResponse(content=plot_data)

@app.get("/all_audios")
async def get_all_audios():
    audio_files = os.listdir(STATIC_UPLOAD_DIR)
    embeddings = []
    labels = []

    for audio_file in audio_files:
        file_path = os.path.join(STATIC_UPLOAD_DIR, audio_file)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            features = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(features.squeeze().numpy())
        labels.append(audio_file)

    if embeddings:
        embeddings = np.array(embeddings)  # Convert to NumPy array
        tsne = TSNE(n_components=2, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = np.array([])

    plot_data = {
        "x": embeddings_2d[:, 0].tolist() if embeddings.size else [],
        "y": embeddings_2d[:, 1].tolist() if embeddings.size else [],
        "text": labels,
        "file_ids": labels
    }

    return JSONResponse(content=plot_data)

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    audio_path = os.path.join(STATIC_UPLOAD_DIR, file_id)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(audio_path, media_type='audio/mpeg')


@app.get("/clip-clap", response_class=HTMLResponse)
async def analyze(request: Request):
    return templates.TemplateResponse("clip-clap.html", {"request": request})

@app.post("/clip-clap-wav", response_class=JSONResponse)
async def analyze_wav(request: Request, wav_file: UploadFile = File(...), candidate_labels: str = Form(...), candidate_images: List[UploadFile] = File(...)):
    unique_folder = create_unique_folder()
    folder_path = os.path.join(UPLOAD_DIR, unique_folder)
    os.makedirs(folder_path)

    input_wav_dir = os.path.join(folder_path, "input_wav")
    os.makedirs(input_wav_dir)

    input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
    with open(input_wav_path, "wb") as buffer:
        buffer.write(await wav_file.read())

    candidate_labels_list = [label.strip() for label in candidate_labels.split(",")]

    # Save uploaded images
    image_paths = []
    for image_file in candidate_images:
        image_path = os.path.join(folder_path, image_file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(await image_file.read())
        image_paths.append(image_path)

    audio, sr = librosa.load(input_wav_path, sr=None, mono=True)
    audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-fused")

    window_size = 2 * sr
    overlap = 1 * sr
    step = window_size - overlap

    scores_dict = {label: [] for label in candidate_labels_list}
    time_points = []

    for start in range(0, len(audio) - window_size + 1, step):
        window = audio[start:start + window_size]
        output = audio_classifier(window, candidate_labels=candidate_labels_list)
        
        for result in output:
            scores_dict[result['label']].append(result['score'])
        
        time_points.append(start / sr)

    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

    # Preprocess images and text
    images = [preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cpu") for image_path in image_paths]
    text_inputs = clip.tokenize(candidate_labels_list).to("cpu")

    # Encode images and text
    with torch.no_grad():
        image_features = torch.cat([clip_model.encode_image(image) for image in images])
        text_features = clip_model.encode_text(text_inputs)

    # Calculate similarity
    similarities = torch.matmul(image_features, text_features.T)

    best_images = {}
    for i, label in enumerate(candidate_labels_list):
        best_image_idx = similarities[:, i].argmax().item()
        best_images[label] = image_paths[best_image_idx]
        print(f"Label: {label}, Best Image: {image_paths[best_image_idx]}, Score: {similarities[best_image_idx, i].item()}")

    plot_data = {
        "audio_scores": {label: {"time_points": time_points, "scores": scores_dict[label]} for label in scores_dict},
        "image_urls": best_images
    }

    return JSONResponse(content=plot_data)

@app.get("/high_level_features", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("high_level_features.html", {"request": request})




@app.post("/high_level_features_request", response_class=HTMLResponse)
async def high_level_features(wav_file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await wav_file.read())
    
    # Load the audio file
    y, sr = librosa.load(temp_file_path, sr=None)
    
    # Define window length and hop length (0.25 seconds window size)
    window_length = int(sr * 0.25)  # 0.25 seconds in samples
    hop_length = window_length      # no overlap
    
    # Frame the audio into chunks
    frames = librosa.util.frame(y, frame_length=window_length, hop_length=hop_length)

    # Prepare response data structure
    feature_data = {
        'brightness': {'time_points': [], 'values': []},
        # 'hardness': {'time_points': [], 'values': []},
        # 'depth': {'time_points': [], 'values': []},
        'roughness': {'time_points': [], 'values': []},
        # 'warmth': {'time_points': [], 'values': []},
        'sharpness': {'time_points': [], 'values': []},
        'booming': {'time_points': [], 'values': []}
    }
    
    # Process each frame and calculate timbral features
    for i, frame in enumerate(frames.T):  # iterate over time windows
        # Skip frames that are all zeros (silent) to avoid errors
        if np.max(np.abs(frame)) == 0:
            continue
        
        temp_frame_file = f"temp_frame_{i}.wav"
        sf.write(temp_frame_file, frame, sr)  # write frame to a temporary file using soundfile
        
        try:
            # Extract timbral features using timbral models
            brightness = timbral_models.timbral_brightness(temp_frame_file)
            # hardness = timbral_models.timbral_hardness(temp_frame_file)
            # depth = timbral_models.timbral_depth(temp_frame_file)
            roughness = timbral_models.timbral_roughness(temp_frame_file)
            # warmth = timbral_models.timbral_warmth(temp_frame_file)
            sharpness = timbral_models.timbral_sharpness(temp_frame_file)
            booming = timbral_models.timbral_booming(temp_frame_file)
            
          

            # Calculate time frame (start time in seconds)
            start_time = i * hop_length / sr
            
            # Store feature data in the structure
            feature_data['brightness']['time_points'].append(start_time)
            feature_data['brightness']['values'].append(brightness)
            
            # feature_data['hardness']['time_points'].append(start_time)
            # feature_data['hardness']['values'].append(hardness)
            
            # feature_data['depth']['time_points'].append(start_time)
            # feature_data['depth']['values'].append(depth)
            
            feature_data['roughness']['time_points'].append(start_time)
            feature_data['roughness']['values'].append(roughness)
            
            # feature_data['warmth']['time_points'].append(start_time)
            # feature_data['warmth']['values'].append(warmth)
            
            feature_data['sharpness']['time_points'].append(start_time)
            feature_data['sharpness']['values'].append(sharpness)
            
            feature_data['booming']['time_points'].append(start_time)
            feature_data['booming']['values'].append(booming)
        
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
        
        # Clean up the temporary frame file
        os.remove(temp_frame_file)

    # Remove the original temp file
    os.remove(temp_file_path)

    # Return the JSON response
    return JSONResponse(content=feature_data)


# MOSQITO VERSION I //START//

# # Helper function to extract Mosqito features
# def extract_features(audio_file_path):
#     # Load the audio file
#     audio_data, sr = librosa.load(audio_file_path, sr=None)
    
#     # Convert audio_data to the expected format for Mosqito (list of values)
#     signal = {
#         'data': np.array(audio_data),
#         'fs': sr
#     }

#     # Extract features
#     features = {}

    # # Loudness
    # loudness = mosqito.loudness_zwicker(signal, field_type="free")
    # features['loudness'] = loudness['values']

    # # Sharpness
    # sharpness = mosqito.sharpness_din(signal, field_type="free")
    # features['sharpness'] = sharpness['values']

    # # Roughness
    # roughness = mosqito.roughness_dw(signal)
    # features['roughness'] = roughness['values']

    # # Tonality
    # tonality = mosqito.tonality(signal)
    # features['tonality'] = tonality['values']

#     # Speech Intelligibility
#     intelligibility = mosqito.sii(signal)
#     features['speech_intelligibility'] = intelligibility['sii']

#     return features

# @app.get("/high_level_mosqito", response_class=HTMLResponse)
# async def analyze(request: Request):
#     # Render the analyze.html template
#     return templates.TemplateResponse("high_level_mosqito.html", {"request": request})

# @app.post("/high_level_mosqito")
# async def high_level_mosqito(wav_file: UploadFile = File(...)):
#     # Save the uploaded file temporarily
#     temp_file_path = f"temp_{wav_file.filename}"
#     with open(temp_file_path, "wb") as temp_file:
#         temp_file.write(await wav_file.read())

#     # Extract features using Mosqito
#     try:
#         features = extract_features(temp_file_path)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
#     # Delete the temp file
#     import os
#     os.remove(temp_file_path)

#     return JSONResponse(content=features)
# MOSQITO VERSION I //END//

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Helper function to extract loudness using loudness_zwst_perseg
def extract_loudness_per_band(audio_data, sr, bands, nperseg=4096):
    # Dictionary to store loudness for each band
    loudness_per_band = {}

    # Iterate through each frequency band
    for low, high in bands:
        # Apply bandpass filter
        band_filtered = bandpass_filter(audio_data, low, high, sr)
        
        # Compute loudness using Mosqito
        loudness, _, _, time_axis = mosqito.loudness_zwst_perseg(band_filtered, fs=sr, nperseg=nperseg, noverlap=None, field_type="free")
        
        # Store results
        loudness_per_band[f"{low}-{high}Hz"] = {"loudness": loudness.tolist(), "time_axis": time_axis.tolist()}
    
    return loudness_per_band

@app.get("/high_level_mosqito_2", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("high_level_mosqito_2.html", {"request": request})

@app.post("/analyze_audio")
async def analyze_audio(wav_file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{wav_file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await wav_file.read())

    # Load the audio file
    audio_data, sr = librosa.load(temp_file_path, sr=None)

    # Define frequency bands
    bands = [
        (20, 100), (100, 200), (200, 500), (500, 1000),
        (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)
    ]

    # Extract loudness for each frequency band
    loudness_per_band = extract_loudness_per_band(audio_data, sr, bands)

    # Return the result as JSON
    return JSONResponse(content=loudness_per_band)

import random
from pydantic import BaseModel


# --- New response model for shape questionnaire ---
class ShapeResponseData(BaseModel):
    session_id: str
    sound: str
    response: dict  # expects shapeValue + timestamp

@app.post("/save_shape_response")
async def save_shape_response(data: ShapeResponseData):
    os.makedirs("responses_shape", exist_ok=True)
    session_id = data.session_id or "anonymous"
    filepath = f"responses_shape/{session_id}.json"

    # Load existing responses
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(data.dict())

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok"}

@app.get("/sound_image_shape", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("sound_image_shape.html", {"request": request})

# --- New response model for shape questionnaire ---
class TextureResponseData(BaseModel):
    session_id: str
    sound: str
    response: dict  # expects shapeValue + timestamp

@app.post("/save_texture_response")
async def save_shape_response(data: TextureResponseData):
    os.makedirs("responses_texture", exist_ok=True)
    session_id = data.session_id or "anonymous"
    filepath = f"responses_texture/{session_id}.json"

    # Load existing responses
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(data.dict())

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok"}


@app.get("/sound_image_texture", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("sound_image_texture.html", {"request": request})

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR = "static/indefinite_pitch"

def get_all_sounds():
    return [f for f in os.listdir(SOUND_DIR) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds")
async def get_sounds():
    all_sounds = get_all_sounds()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR_TRAINING = "static/indefinite_pitch/training"

def get_all_sounds_training():
    return [f for f in os.listdir(SOUND_DIR_TRAINING) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds_training")
async def get_sounds_training():
    all_sounds = get_all_sounds_training()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR_SHAPE = "static/image_shape"

def get_all_sounds_shape():
    return [f for f in os.listdir(SOUND_DIR_SHAPE) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds_shape")
async def get_sounds_shape():
    all_sounds = get_all_sounds_shape()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR_SHAPE_TRAINING = "static/image_shape/training"

def get_all_sounds_shape_training():
    return [f for f in os.listdir(SOUND_DIR_SHAPE_TRAINING) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds_shape_training")
async def get_sounds_shape_training():
    all_sounds = get_all_sounds_shape_training()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR_TEXTURE = "static/image_texture"

def get_all_sounds_texture():
    return [f for f in os.listdir(SOUND_DIR_TEXTURE) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds_texture")
async def get_sounds_texture():
    all_sounds = get_all_sounds_texture()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)

# ---- Dynamic Sound Stimuli Listing ----
SOUND_DIR_TEXTURE_TRAINING = "static/image_texture/training"

def get_all_sounds_texture_training():
    return [f for f in os.listdir(SOUND_DIR_TEXTURE_TRAINING) if f.lower().endswith((".wav", ".mp3", ".ogg"))]

@app.get("/get_sounds_texture_training")
async def get_sounds_texture_training():
    all_sounds = get_all_sounds_texture_training()
    random_order = random.sample(all_sounds, len(all_sounds))
    return JSONResponse(content=random_order)


@app.get("/indefinite_pitch", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("indefinite_pitch.html", {"request": request})

@app.get("/evaluation", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("soundsketcher_evaluation.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def analyze(request: Request):
    # Render the analyze.html template
    return templates.TemplateResponse("test.html", {"request": request})


class ResponseData(BaseModel):
    session_id: str
    sound: str
    training: Optional[bool] = None
    response: dict  # you can structure this depending on what you collect (slider values etc.)


@app.post("/save_responses_indefinite_pitch")
async def save_response(data: ResponseData):
    os.makedirs("responses_indefinite_pitch", exist_ok=True)
    session_id = data.dict().get("session_id", "anonymous")
    print("session_id", session_id)
    filepath = f"responses_indefinite_pitch/{session_id}.json"

    # Load existing responses for that session
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    # Append new response
    existing.append(data.dict())

    # Save back
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok"}

# Create folder if it doesn't exist
os.makedirs("responses_indefinite_pitch/user_info", exist_ok=True)

class UserInfo(BaseModel):
    session_id: str
    gender: str
    age: int
    music_experience: str
    years_experience: int          # required
    training_type: str             # required
    plays_instrument: bool
    instrument_name: str = ""
    #equipment: str
    knows_pitch: bool
    hearing_condition: str
    absolute_pitch: bool
    # ✅ NEW: consent fields
    consent: bool
    consent_version: Optional[str] = None
    consent_timestamp: Optional[str] = None

@app.post("/save_indefinite_pitch_user_info")
async def save_user_info(info: UserInfo):
    directory = "responses_indefinite_pitch/user_info"
    print("🔎 Current working dir:", os.getcwd())
    print("🔎 Exists?", os.path.exists(directory), "->", os.path.abspath(directory))
    
    filepath = os.path.join(directory, f"{info.session_id}.json")
    print("🔎 Writing to:", filepath)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info.dict(), f, indent=2, ensure_ascii=False)
    return {"status": "ok", "message": "User info saved successfully."}

# --- ADD THESE IMPORTS (top of file) ---
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field
import os, json

# --- CONFIG: where logs will live ---
LOG_ROOT = "responses_indefinite_pitch/logs"
os.makedirs(LOG_ROOT, exist_ok=True)

# --- Pydantic models that accept flexible payloads ---
class LogRecord(BaseModel):
    # Required fields coming from the client logger
    t: datetime                      # ISO timestamp
    event: str                       # e.g., "sine_input", "question_summary", ...
    session_id: str
    question_id: int = 0

    # Optional context fields (client provides these; keep flexible)
    training: Optional[bool] = None
    sound: Optional[str] = None
    sound_index: Optional[int] = None

    # Allow any extra fields (freq_hz, vol_db, durations, gains, etc.)
    class Config:
        extra = "allow"

class LogBatch(BaseModel):
    records: List[LogRecord] = Field(default_factory=list)

# REPLACE your helpers with these

def _append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

# --- NEW ENDPOINT: receive a batch of interaction logs ---
@app.post("/log_slider_events")
async def log_slider_events(batch: LogBatch):
    """
    Accepts: { "records": [ {t, event, session_id, question_id, ...}, ... ] }
    Stores:
      - NDJSON stream per (session_id, question_id): logs/<session>/q###.ndjson
      - If 'event' == 'question_summary', also writes q###_summary.json
    Returns: counts by (session_id, question_id)
    """
    if not batch.records:
        return {"status": "ok", "received": 0, "by_question": {}}

    by_question_counts: Dict[str, int] = {}

    for rec in batch.records:
        # normalize & paths
        sid = rec.session_id or "anonymous"
        qid = int(getattr(rec, "question_id", 0) or 0)

        session_dir = os.path.join(LOG_ROOT, sid)
        os.makedirs(session_dir, exist_ok=True)

        ndjson_path = os.path.join(session_dir, f"q{qid:03d}.ndjson")
        rec_dict = rec.dict()  # includes extra fields

        # append to NDJSON stream
        _append_ndjson(ndjson_path, rec_dict)

        # optional: store a clean per-question summary
        if rec.event == "question_summary":
            # include some context next to stats
            summary_payload = {
                "session_id": sid,
                "question_id": qid,
                "training": rec_dict.get("training"),
                "sound": rec_dict.get("sound"),
                "sound_index": rec_dict.get("sound_index"),
                "stats": rec_dict.get("stats", {}),   # whatever you placed there
                "received_at": datetime.utcnow().isoformat() + "Z"
            }
            _write_json(os.path.join(session_dir, f"q{qid:03d}_summary.json"),
                        summary_payload)

        key = f"{sid}#q{qid}"
        by_question_counts[key] = by_question_counts.get(key, 0) + 1

    return {
        "status": "ok",
        "received": len(batch.records),
        "by_question": by_question_counts
    }

# Create folder if it doesn't exist
os.makedirs("responses_shape/user_info", exist_ok=True)

@app.post("/save_image_shape_user_info")
async def save_image_shape_user_info(info: UserInfo):
    print("📝 Received user info:", info.dict())  # debug
    filepath = f"responses_shape/user_info/{info.session_id}.json"
    with open(filepath, "w") as f:
        json.dump(info.dict(), f, indent=2, ensure_ascii=False)
    return {"status": "ok", "message": "User info saved successfully."}

# Create folder if it doesn't exist
os.makedirs("responses_texture/user_info", exist_ok=True)

@app.post("/save_image_texture_user_info")
async def save_image_texture_user_info(info: UserInfo):
    print("📝 Received user info:", info.dict())  # debug
    filepath = f"responses_texture/user_info/{info.session_id}.json"
    with open(filepath, "w") as f:
        json.dump(info.dict(), f, indent=2, ensure_ascii=False)
    return {"status": "ok", "message": "User info saved successfully."}



@app.get("/indefinite_pitch_results", response_class=HTMLResponse)
async def show_results(request: Request):
    return templates.TemplateResponse("indefinite_pitch_results.html", {"request": request})

@app.get("/sound_image_shape_results", response_class=HTMLResponse)
async def show_results(request: Request):
    return templates.TemplateResponse("sound_image_shape_results.html", {"request": request})

@app.get("/sound_image_texture_results", response_class=HTMLResponse)
async def show_results(request: Request):
    return templates.TemplateResponse("sound_image_texture_results.html", {"request": request})

import glob

@app.get("/get_all_responses_indefinite_pitch")
async def get_all_responses():
    files = glob.glob("responses_indefinite_pitch/*.json")
    all_data = []

    for filepath in files:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

    return JSONResponse(content=all_data)

@app.get("/get_all_responses_shape")
async def get_all_responses():
    files = glob.glob("responses_shape/*.json")
    all_data = []

    for filepath in files:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

    return JSONResponse(content=all_data)

@app.get("/get_all_responses_texture")
async def get_all_responses():
    files = glob.glob("responses_texture/*.json")
    all_data = []

    for filepath in files:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

    return JSONResponse(content=all_data)

#! Created to extract features with MATLAB on a separate thread | Disabled old features
def matlab_worker(queue):
    
    # Start MATLAB
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('/usr/local/MATLAB/R2024b/mirtoolbox-main/MIRToolbox'))
    eng.addpath(eng.genpath('/mnt/ssd1/kvelenis/soundsketcher/soundsketcher_aux_scripts'))
    eng.cd('/mnt/ssd1/kvelenis/soundsketcher/soundsketcher_aux_scripts')
    eng.eval("clear all; rehash;",nargout=0)
    print("MATLAB worker started")
    # eng = None

    # Process requests serially
    while True:
        audio_path,timestamps,container,event = queue.get()
        try:
            results = matlextract.extract_mir_features_for_file_interpolated(eng,audio_path,timestamps)
            container['results'] = results
        except Exception as error:
            print(error)
            container['results'] = {
                'MPS_roughness': np.zeros(len(timestamps)),
                # 'roughness_Zwicker': np.zeros(len(timestamps)),
                'sharpness_Zwicker': np.zeros(len(timestamps)),
                'roughness_vassilakis': np.zeros(len(timestamps)),
                # 'roughness_sethares': np.zeros(len(timestamps)),
                'weighted_spectral_centroid': np.zeros(len(timestamps))
            }
        finally:
            event.set()
            
#! Created to allow handling of parallel requests
def create_executor(workers = -1):

    def aux():
        return None
    
    if workers == -1:
        workers = psutil.cpu_count(logical = False) # Server has 24 physical cores / 32 logical cores

    executor = ProcessPoolExecutor(max_workers = workers)
    executor.submit(aux)

    return executor

#! Updated to create MATLAB thread & process executor [DON'T CHANGE ORDER OF CODE LINES]
if __name__ == "__main__":
    matlab_queue = Queue()
    matlab_manager = Manager()
    executor = create_executor(8) # 4 -> temporary argument, remove for deployment
    Thread(target = matlab_worker,args = [matlab_queue],daemon = True).start()
    uvicorn.run(app,host = "0.0.0.0",port = 5002)

# , ssl_keyfile='/home/kvelenis/Documents/SSL_certificates/server.key', ssl_certfile='/home/kvelenis/Documents/SSL_certificates/server.crt'``