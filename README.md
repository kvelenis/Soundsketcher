# Soundsketcher
An online app that creates graphic music scores with audio input
# Soundsketcher Project

Soundsketcher is a research-driven tool designed to explore the intersection of sound and visual art. It translates audio features into meaningful visual representations, focusing on perceptual relevance, automation, and customization. The project incorporates cognitive science principles and cross-modal mappings to generate intuitive and expressive visualizations of sound events.

## Features
- **Audio-to-Visual Mapping**: Converts sound features into visual sketches using spectral and perceptual characteristics.
- **Structural Sound Segmentation**: Uses Wav2Vec 2.0 and CLAP embeddings to define meaningful sound objects.
- **Semantic Sound Identification**: Employs zero-shot classification for sound labeling with future enhancements planned for timbre-focused categorization.
- **Creative Extensions**: Introduces randomized mappings, polygon-based representations, and sound-driven synthesis.

## Current Progress and Challenges
- **Feature Mapping Enhancements**: Improving the integration of multiple features for perceptually rich visualizations.
- **Refined Sound Segmentation**: Transitioning from frame-based analysis to object-based segmentation for better cohesion.
- **Enhanced Semantic Identification**: Addressing limitations in CLAP embeddings by fine-tuning with a descriptor dataset.
- **Stream Separation**: Exploring AI-driven models (e.g., Demucs, Open-Unmix) to isolate overlapping sound sources.

By refining feature extraction, segmentation, and classification, Soundsketcher aims to provide a more expressive, perceptually relevant, and artistically meaningful visualization system.

# Project Setup Guide

## Installation

Follow these steps to set up and run the project.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Project
Start the FastAPI server using:
```bash
uvicorn main:app --reload
```
Replace `main` with the actual script where your FastAPI `app` instance is defined.

### 5. Access the API
Once running, you can access the API at:
- **Interactive API docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc API docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Notes
- Ensure that `soundsketcher_aux_scripts` exists in your project structure.
- If you are missing system dependencies, install them separately (e.g., `ffmpeg` for audio processing).
- If using `clip-by-openai`, install the required models following OpenAI's CLIP documentation.

---