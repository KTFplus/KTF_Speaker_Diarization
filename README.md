# Speaker Diarization and Transcription Module (KTF Plus)

## Overview

This repository contains the Colab notebook for the speaker diarization and transcription module developed for the KTF project.  
It combines ECAPA-TDNN-based speaker embeddings with Whisper-based ASR to perform accurate speaker diarization and generate high-quality transcriptions.

---

## Features

- ECAPA-TDNN-based speaker embedding extraction
- Automatic speaker count estimation via eigen-gap analysis
- Speaker diarization using spectral clustering
- Fast ASR with Whisper
- Accurate sentence correction using fine-tuned Whisper model
- RESTful API implementation using FastAPI
- Optimized for GPU execution on Google Colab

---

## How to Use

1. Open the notebook in Google Colab (GPU runtime recommended)
2. Install the required packages and run the provided cell to start the server
3. The FastAPI server will start via ngrok and expose the `/api/analyze-audio` endpoint

---

## Tech Stack

- Python
- PyTorch
- torchaudio
- SpeechBrain (ECAPA-TDNN)
- Faster-Whisper
- HuggingFace Transformers (fine-tuned Whisper)
- scikit-learn, SciPy
- FastAPI
- pyngrok

---

## Processing Flow and Implementation

1. **Audio Preprocessing**
    - Load the input audio file using `torchaudio`
    - Convert to mono if multi-channel
    - Resample to 16 kHz (required for ECAPA and Whisper models)

2. **Speaker Embedding & Clustering**
    - Split audio into sliding windows (`segment_duration = 2.3 seconds`)
    - Extract ECAPA-TDNN embeddings per segment
    - Compute pairwise cosine distances → build affinity matrix
    - Compute Laplacian eigenvalues → determine optimal number of clusters (via eigen-gap)
    - Perform Spectral Clustering → assign speaker labels

3. **Whisper-based Transcription & Speaker Mapping**
    - Perform full transcription using Faster-Whisper model
    - Map transcription segments to speaker segments based on overlap
    - Compose speaker-attributed subtitle list

4. **Fine-tuned Whisper-based Correction**
    - Run transcription again using fine-tuned Whisper model (`urewui/ktf`)
    - Compare base Whisper and fine-tuned Whisper outputs using similarity matching
    - Replace and refine sentences → finalize speaker-attributed transcript

5. **Result Output**
    - The output is returned in JSON format:
        - `transcript`: full sentence-level transcript of the entire audio (string)
        - `speakers`: list of speaker-attributed segments in the format `[ { speaker: "A", text: "..." }, ... ]`
