from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
from soundsketcher_aux_scripts import sonic_annotator_call as sac
from soundsketcher_aux_scripts import clustering_objects as clobj
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
from scipy.signal import butter, lfilter

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

from mosqito.sq_metrics import sharpness_din_perseg, loudness_zwst_perseg  # Correct imports for features

from soundsketcher_aux_scripts import clustering_objects_with_wav2vec as obj

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "user_data")
STATIC_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_UPLOAD_DIR, exist_ok=True)
app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static_uploads", StaticFiles(directory=STATIC_UPLOAD_DIR), name="static_uploads")
app.mount("/user_data", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# Initialize Jinja2Templates and point it to the templates directory
templates = Jinja2Templates(directory="templates")

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

@app.get("/")
async def index(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})

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

def run_feature_extraction(input_wav_path, output_csv_dir, n_fft=2048, hop_length=2048):



    


    # with open(input_wav_path, "wb") as buffer:
    #     buffer.write(await wav_file.read())
    # input_wav_path = '../input_wav/synth_sound_2.wav'
    # output_csv_dir = '../output_csv'
    audio_features = ["vamp:pyin:yin:periodicity"]
    # audio_feature = 'vamp:bbc-vamp-plugins:bbc-rhythm:onset'
    # for i in range(len(audio_features)):
    sac.run_sonic_annotator("../" + input_wav_path, "../" + output_csv_dir) 
    print(input_wav_path)   
    # sac.run_sonic_annotator("../" + input_wav_path, "../" + output_csv_dir)



    # features_librosa = run_feature_extraction(input_wav_path, output_csv_dir, n_fft, hop_length)
    # json_data = jsc.creator_librosa(features_librosa)

    




    # Load the audio file
    y, sr = librosa.load(input_wav_path, sr=None)

    # Validate audio length
    if len(y) < n_fft:
        raise ValueError("Audio signal is too short for the specified frame_length and hop_length.")

    # Ensure the signal length is an integral multiple of hop_length
    frames_needed = int(np.ceil(len(y) / hop_length))
    target_length = frames_needed * hop_length + n_fft
    y = librosa.util.fix_length(y, size=target_length)

    # Extract features (per window)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_flux = np.mean(np.diff(librosa.amplitude_to_db(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2, axis=0), axis=0)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)

    # F0 and Voiced Probability extraction using Librosa
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)

    # F0 extraction using Aubio
    hop_size = hop_length
    win_size = n_fft
    aubio_pitch_detector = aubio.pitch("yin", win_size, hop_size, sr)
    aubio_pitch_detector.set_unit("Hz")
    aubio_pitch_detector.set_silence(-40)

    fmin = 50
    fmax = 500

    aubio_f0 = []
    for i in range(0, len(y), hop_size):
        frame = y[i:i + win_size].astype(np.float32)
        if len(frame) < win_size:
            frame = np.pad(frame, (0, win_size - len(frame)), mode='constant')
        pitch = aubio_pitch_detector(frame)[0]
        aubio_f0.append(pitch if fmin <= pitch <= fmax else 0.0)

    aubio_f0_smoothed = median_filter(aubio_f0, size=3)
    aubio_timestamps = np.arange(len(aubio_f0_smoothed)) * (hop_size / sr)

    # CREPE Pitch Extraction
    time_crepe, frequency_crepe, confidence_crepe, _ = crepe.predict(y, sr, viterbi=True, model_capacity="tiny")
    crepe_f0 = np.interp(librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length), time_crepe, frequency_crepe)
    # Get timestamps for Librosa features
    timestamps = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)


    # Find the CSV file containing "periodicity" in its name
    periodicity_csv = None
    for file in os.listdir(output_csv_dir):
        if "periodicity" in file and file.endswith(".csv"):
            periodicity_csv = os.path.join(output_csv_dir, file)
            break

    if periodicity_csv:
        # Load the periodicity CSV file
        periodicity_df = pd.read_csv(periodicity_csv)
        
        # Assuming the CSV has two columns: 'time' and 'value'
        time_values = periodicity_df.iloc[:, 0].values  # First column: timeseries
        periodicity_values = periodicity_df.iloc[:, 1].values  # Second column: values
        
        # Interpolate using numpy.interp
        interpolated_periodicity_values = np.interp(timestamps, time_values, periodicity_values)
        
        # # Combine timestamps with interpolated periodicity values
        # result_df = pd.DataFrame({
        #     'librosa_time': timestamps,
        #     'interpolated_periodicity': interpolated_periodicity_values
        # })
    else:
        print("No file containing 'periodicity' found in the specified directory.")

    
    # Find the CSV file containing "candidates" in its name
    candidates_csv = None
    for file in os.listdir(output_csv_dir):
        if "candidates" in file and file.endswith(".csv"):
            candidates_csv = os.path.join(output_csv_dir, file)
            break

    if not candidates_csv:
        print("No file containing 'candidates' found in the specified directory.")
        return None

    timestamps_column = []
    frequencies_column = []

    # Open and process the CSV file line-by-line
    with open(candidates_csv, "r") as file:
        for line in file:
            # Split the line by commas and strip whitespace
            parts = line.strip().split(",")
            if len(parts) > 0:
                try:
                    # First column is the timestamp
                    timestamp = float(parts[0])
                    timestamps_column.append(timestamp)

                    # Rest of the columns are frequencies
                    frequencies = [
                        float(freq) for freq in parts[1:] if freq.strip() != ""
                    ]
                    if frequencies:
                        # Keep only the first (most powerful) frequency
                        frequencies_column.append(frequencies[0])
                    else:
                        # Silence case: no frequencies
                        frequencies_column.append(0)
                except ValueError:
                    print(f"Skipping malformed line: {line}")
                    continue

    # Create a DataFrame from parsed data
    processed_df = pd.DataFrame({
        "timestamp": timestamps_column,
        "frequency": frequencies_column
    })

    # Interpolate frequencies to match the input timestamps
    interpolated_candidate_frequencies = np.interp(timestamps, processed_df["timestamp"], processed_df["frequency"])





    # Mosqito Features
    mosqito_features = {
        'loudness': [],
        'sharpness': [],
    }
    mosqito_timestamps = []

    sharpness_values, sharpness_time_values = sharpness_din_perseg(signal=y, fs=sr,nperseg=n_fft, noverlap=None, field_type='free')
    loudness_values, N_spec, Bark_axis, loudness_time_values = loudness_zwst_perseg(signal=y, fs=sr,nperseg=n_fft, noverlap=None, field_type='free')
    # print(N_spec[:, 0].tolist(), len(N_spec))
    # print(Bark_axis.tolist(), len(Bark_axis))
    # Define the file path
    output_file = "output.txt"
    # Calculate the numerator: (a*z + b*x + c*d)
    bark_numerator = sum(i * j for i, j in zip(N_spec[:, 0].tolist(), Bark_axis.tolist()))
    
    # Calculate the denominator: (z + x + d)
    bark_denominator = sum(Bark_axis)
    
    # Compute the result
    result = bark_numerator / bark_denominator if bark_denominator != 0 else None
    bark2freq = 600 * math.sinh(result / 6)
    
    # Open the file in write mode and save the data
    with open(output_file, "w") as file:
        file.write(f"N_spec: {N_spec[:, 0]}\n")
        file.write(f"Length of N_spec: {len(N_spec[:, 0])}\n\n")
        file.write(f"Length of N_time: {N_spec[:, 1]}\n\n")
        file.write(f"Bark_axis: {Bark_axis}\n")
        file.write(f"Length of Bark_axis: {len(Bark_axis)}\n")
        file.write(f"bark_numerator: {bark_numerator}\n")
        file.write(f"bark_denominator: {bark_denominator}\n")
        file.write(f"result: {result}\n")
        file.write(f"bark2freq: {bark2freq}\n")


    print(f"Data saved to {output_file}")
    mosqito_features['loudness'].extend(loudness_values)
    mosqito_features['sharpness'].extend(sharpness_values)

    # Interpolate Mosqito features to match Librosa timestamps
    interpolated_features = {}
    for key, time_values in zip(['loudness', 'sharpness'], [loudness_time_values, sharpness_time_values]):
        interpolated_features[key] = np.interp(timestamps, time_values, mosqito_features[key])

    # Create a dictionary for all features
    features = {
        'timestamp': timestamps,
        'zcr': zcr[0],
        'spectral_centroid': spectral_centroid[0],
        'spectral_bandwidth': spectral_bandwidth[0],
        'rms': rms[0],
        'spectral_flux': spectral_flux,
        'spectral_flatness': spectral_flatness[0],
        'f0_librosa': f0,
        'voiced_prob': voiced_prob,
        'aubio_f0': np.interp(timestamps, aubio_timestamps, aubio_f0),
        'crepe_f0': crepe_f0,
        'crepe_confidence': np.interp(timestamps, time_crepe, confidence_crepe),
        'yin_periodicity': interpolated_periodicity_values,
        'f0_candidates': interpolated_candidate_frequencies,
        'loudness': interpolated_features['loudness'],
        'sharpness': interpolated_features['sharpness']
        
    }

    # Convert to DataFrame
    df = pd.DataFrame(features)

    # Save each feature to a separate CSV (optional for debugging)
    for feature_name in features.keys():
        if feature_name != 'timestamp':
            feature_df = df[['timestamp', feature_name]]
            feature_df.to_csv(os.path.join(output_csv_dir, f"{feature_name}.csv"), index=False)

    print(f"Features extracted and saved to {output_csv_dir}")
    return features



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

@app.get("/objectifier/{audio_file}", response_class=HTMLResponse)
async def get_plotly_page(request: Request, audio_file: str):
    """
    Render the Plotly visualization page for the given audio file.
    """
    # obj.objectifier(audio_pa)
    json_file_path = f"clustering_destination_plots/{audio_file}_plotly_data.json"
    if not os.path.exists(json_file_path):
        return HTMLResponse(
            content=f"<h1>Error: Plotly data not found for {audio_file}</h1>", status_code=404
        )
    
    return templates.TemplateResponse("plotly_visualization.html", {"request": request, "audio_file": audio_file})

@app.get("/plotly-data/{audio_file}")
async def get_plotly_data(audio_file: str):
    """
    Serve the JSON data for Plotly visualization.
    """
    json_file_path = f"clustering_destination_plots/{audio_file}_plotly_data.json"
    try:
        with open(json_file_path, "r") as f:
            plot_data = json.load(f)
        return plot_data
    except FileNotFoundError:
        return {"error": "Plotly data not found"}

# # POST Endpoint: Upload and Process Audio
# @app.post("/objectifier_wav", response_class=JSONResponse)
# async def objectifier_endpoint(wav_file: UploadFile = File(...)):
#     """
#     Upload a WAV file, process it with obj.objectifier, and return the results.
#     """
#     try:
#         # Step 1: Save the uploaded file
#         unique_folder = create_unique_folder()
#         folder_path = os.path.join(UPLOAD_DIR, unique_folder)
#         os.makedirs(folder_path, exist_ok=True)

#         input_wav_path = os.path.join(folder_path, wav_file.filename)
#         with open(input_wav_path, "wb") as buffer:
#             buffer.write(await wav_file.read())

#         # Step 2: Process the WAV file with obj.objectifier
#         output = obj.objectifier(input_wav_path)

#         # Step 3: Directly return the processed data as a response
#         return JSONResponse(content=output)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # GET Endpoint: Render Visualization Page
# @app.get("/objectifier", response_class=HTMLResponse)
# async def get_plotly_page(request: Request):
#     """
#     Render the Plotly visualization page for the given audio file.
#     """
#     return templates.TemplateResponse("plotly_visualization.html", {"request": request})

# @app.post("/upload_wav")
# async def upload_wav(wav_file: UploadFile = File(...)):
#     unique_folder = create_unique_folder()
#     folder_path = os.path.join(UPLOAD_DIR, unique_folder)
#     os.makedirs(folder_path)

#     input_wav_dir = os.path.join(folder_path, "input_wav")
#     output_csv_dir = os.path.join(folder_path, "output_csv")
#     os.makedirs(input_wav_dir)
#     os.makedirs(output_csv_dir)

#     input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
#     with open(input_wav_path, "wb") as buffer:
#         buffer.write(await wav_file.read())
#     # input_wav_path = '../input_wav/synth_sound_2.wav'
#     # output_csv_dir = '../output_csv'
#     audio_features = ["vamp:vamp-example-plugins:zerocrossing:zerocrossings", "vamp:vamp-example-plugins:spectralcentroid:linearcentroid", "vamp:vamp-example-plugins:amplitudefollower:amplitude"]#, "vamp:bbc-vamp-plugins:bbc-rhythm:onset"]
#     audio_feature = 'vamp:bbc-vamp-plugins:bbc-rhythm:onset'
#     # for i in range(len(audio_features)):
#     #     sac.run_sonic_annotator(audio_features[i], "../" + input_wav_path, "../" + output_csv_dir) 
#     print(input_wav_path)   
#     # sac.run_sonic_annotator("../" + input_wav_path, "../" + output_csv_dir)

#     n_fft=2048
#     hop_length=2048

#     features_librosa = run_feature_extraction(input_wav_path, output_csv_dir, n_fft, hop_length)
#     json_data = jsc.creator_librosa(features_librosa)
    
#     # The same but with Librosa
#     # At the end of the creator function
#     json_data = convert_to_serializable(json_data)

#     features = json_data["Song"]["features_per_timestamp"]

#     # Call objectifier
#     clobj.cluster_objectification(features, input_wav_path, n_fft, hop_length)
#     # Call objectifier
#     # clobj.cluster_objectification_with_dtw(features, input_wav_path)


#     # Prepare JSON response
#     response_data = {
#         "filename": wav_file.filename,
#         "content_type": wav_file.content_type,
#         "json_data": json_data  # Assuming this is the data you want to return
#     }
#     os.chdir("/media/datadisk/velenisrepos/soundsketcher")

#     # Store the plot data in a session or temporary storage
#     # request.session["plot_data"] = plot_data    

#     # Return JSON response
#     return JSONResponse(content=response_data)
@app.post("/upload_wavs")
async def upload_wavs(wav_files: List[UploadFile] = File(...)):  # Use typing.List
    # Create a unique folder for this session
    unique_folder = create_unique_folder()
    folder_path = os.path.join(UPLOAD_DIR, unique_folder)
    os.makedirs(folder_path)

    input_wav_dir = os.path.join(folder_path, "input_wavs")
    output_csv_dir = os.path.join(folder_path, "output_csv")
    os.makedirs(input_wav_dir)
    os.makedirs(output_csv_dir)

    # Container for results
    all_features = []

    for wav_file in wav_files:
        input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
        with open(input_wav_path, "wb") as buffer:
            buffer.write(await wav_file.read())

        # Extract features using your custom method
        n_fft = 2048
        hop_length = 2048
        features_librosa = run_feature_extraction(input_wav_path, output_csv_dir, n_fft, hop_length)

        # Convert features to JSON format
        json_data = jsc.creator_librosa(features_librosa)
        json_data = convert_to_serializable(json_data)

        # Collect features for plotting
        all_features.append({
            "filename": wav_file.filename,
            "features": json_data["Song"]["features_per_timestamp"]
        })

    # Return the features for all files
    response_data = {
        "files_processed": len(wav_files),
        "features": all_features
    }
    return JSONResponse(content=response_data)

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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
