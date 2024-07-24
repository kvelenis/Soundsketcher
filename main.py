from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from sklearn.manifold import TSNE
import numpy as np
import os
import pandas as pd
import uuid
from datetime import datetime
from typing import List
import time
import sonic_annotator_call as sac
import json_creator as jsc
import torch
import librosa
from transformers import pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PIL import Image
import clip  # Import the CLIP library



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
    "melodic_synth_evolving.mp3",
    "Monty_Python_bright_side_of_life.mp3",
    "scary_door.mp3",
    "tzitziki.mp3",
    "water_drop.mp3"
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

@app.get("/")
async def index(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})

# Serve the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.post("/upload_wav")
async def upload_wav(wav_file: UploadFile = File(...)):
    unique_folder = create_unique_folder()
    folder_path = os.path.join(UPLOAD_DIR, unique_folder)
    os.makedirs(folder_path)

    input_wav_dir = os.path.join(folder_path, "input_wav")
    output_csv_dir = os.path.join(folder_path, "output_csv")
    os.makedirs(input_wav_dir)
    os.makedirs(output_csv_dir)

    input_wav_path = os.path.join(input_wav_dir, wav_file.filename)
    with open(input_wav_path, "wb") as buffer:
        buffer.write(await wav_file.read())
    # input_wav_path = '../input_wav/synth_sound_2.wav'
    # output_csv_dir = '../output_csv'
    audio_features = ["vamp:vamp-example-plugins:zerocrossing:zerocrossings", "vamp:vamp-example-plugins:spectralcentroid:linearcentroid", "vamp:vamp-example-plugins:amplitudefollower:amplitude", "vamp:bbc-vamp-plugins:bbc-rhythm:onset"]
    audio_feature = 'vamp:bbc-vamp-plugins:bbc-rhythm:onset'
    # for i in range(len(audio_features)):
    #     sac.run_sonic_annotator(audio_features[i], "../" + input_wav_path, "../" + output_csv_dir)    
    sac.run_sonic_annotator("../" + input_wav_path, "../" + output_csv_dir)
    
    json_data = jsc.creator(output_csv_dir)
    
    # Prepare JSON response
    response_data = {
        "filename": wav_file.filename,
        "content_type": wav_file.content_type,
        "json_data": json_data  # Assuming this is the data you want to return
    }
    os.chdir("/media/datadisk/velenisrepos/soundsketcher")

    # Store the plot data in a session or temporary storage
    # request.session["plot_data"] = plot_data    

    # Return JSON response
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
