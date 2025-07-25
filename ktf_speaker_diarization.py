# -*- coding: utf-8 -*-
"""KTF_Speaker_Diarization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kZsD_aH7Zw0v_jckIpdgvM57eJL6kVQH
"""

!pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -q speechbrain
!pip install -q scikit-learn
!pip install -q matplotlib seaborn
!pip install numpy==1.26.4
!pip install faster-whisper
!pip install transformers==4.30.2
!pip install tokenizers==0.13.3
!pip install -q fastapi pyngrok uvicorn

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import torch
import torchaudio
from pyngrok import ngrok
import nest_asyncio
import uvicorn

def run_full_pipeline(audio_path):
    import torchaudio
    import torch
    import numpy as np
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.metrics.pairwise import pairwise_distances
    from scipy.sparse import csgraph
    from scipy.linalg import eigh
    from sklearn.cluster import SpectralClustering
    from transformers import pipeline as hf_pipeline
    from faster_whisper import WhisperModel
    from difflib import SequenceMatcher
    import re

    # 1. 오디오 로드 및 리샘플링
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
        audio_path = "resampled.wav"
        torchaudio.save(audio_path, waveform, sr)

    # 2. 화자 분리
    def is_silent(segment, threshold=0.005):
        rms = torch.sqrt(torch.mean(segment ** 2))
        return rms.item() < threshold

    segment_duration = 2.3
    segment_samples = int(segment_duration * sr)
    stride_samples = int(0.4 * sr)

    segments, frame_times = [], []
    for i in range(0, waveform.shape[1] - segment_samples + 1, stride_samples):
        seg = waveform[:, i:i + segment_samples]
        if not is_silent(seg):
            segments.append(seg)
            frame_times.append(i / sr)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_ecapa"
    )
    embeddings = []
    for seg in segments:
        try:
            emb = classifier.encode_batch(seg).squeeze(0).detach().numpy()
            embeddings.append(emb)
        except:
            continue

    embedding_matrix = np.vstack(embeddings)
    valid_indices = ~np.isnan(embedding_matrix).any(axis=1)
    embedding_matrix = embedding_matrix[valid_indices]
    frame_times = [frame_times[i] for i, v in enumerate(valid_indices) if v]

    dist_matrix = pairwise_distances(embedding_matrix, metric='cosine')
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    affinity = 1 - np.clip(dist_matrix, 0.0, 1.0)

    laplacian = csgraph.laplacian(affinity, normed=True)
    laplacian = np.nan_to_num(laplacian, nan=0.0, posinf=0.0, neginf=0.0)

    max_index = min(9, len(embedding_matrix) - 1)
    eigenvals, _ = eigh(laplacian, subset_by_index=[0, max_index])
    gaps = np.diff(eigenvals)
    best_k = np.argmax(gaps) + 1

    clustering = SpectralClustering(n_clusters=best_k, affinity='precomputed')
    labels = clustering.fit_predict(affinity)

    merged = []
    current_speaker = labels[0]
    start_time = frame_times[0]
    for i in range(1, len(labels)):
        end_time = frame_times[i] + segment_duration
        if labels[i] != current_speaker:
            merged.append((current_speaker, start_time, end_time))
            current_speaker = labels[i]
            start_time = frame_times[i]
    merged.append((current_speaker, start_time, frame_times[-1] + segment_duration))

    # 3. Whisper 자막 + 화자 매핑
    fw_model = WhisperModel("medium", device="cuda", compute_type="float16")
    fwh_segments, _ = fw_model.transcribe(audio_path, beam_size=5)
    fwh_segments = list(fwh_segments)

    def overlap_time(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in fwh_segments:
        best_overlap = 0.0
        best_speaker = "?"
        for speaker, s_start, s_end in merged:
            overlap = overlap_time(seg.start, seg.end, s_start, s_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        seg.speaker = best_speaker

    merged_segments = []
    prev_speaker = None
    buffer_text = ""
    for seg in fwh_segments:
        speaker = seg.speaker
        text = seg.text.strip()
        if speaker == prev_speaker:
            buffer_text += " " + text
        else:
            if prev_speaker is not None:
                merged_segments.append((prev_speaker, buffer_text.strip()))
            buffer_text = text
            prev_speaker = speaker
    if buffer_text:
        merged_segments.append((prev_speaker, buffer_text.strip()))

    # 4. 파인튜닝된 Whisper로 문장 덮어쓰기
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model="urewui/ktf",
        device=0,  # 0번 GPU
        chunk_length_s=15,
        use_auth_token="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",
        framework="pt"
    )
    ft_result = asr_pipeline(audio_path)
    if "text" not in ft_result:
        raise ValueError("파인튜닝 모델에서 텍스트 결과 없음")

    ft_text = ft_result["text"].strip()
    ft_sentences = [m.group().strip() for m in re.finditer(r"[^.?!]+[.?!]", ft_text)]

    def find_best_combo_match(whisper_text, ft_sentences, max_combo=3):
        best_score = 0.0
        best_text = whisper_text
        used_range = set()
        for i in range(len(ft_sentences)):
            for j in range(i + 1, min(len(ft_sentences), i + max_combo) + 1):
                if any(k in used_range for k in range(i, j)):
                    continue
                combo = " ".join(ft_sentences[i:j])
                score = SequenceMatcher(None, whisper_text, combo).ratio()
                if score > best_score:
                    best_score = score
                    best_text = combo
                    best_range = set(range(i, j))
        if best_score > 0.0:
            used_range.update(best_range)
        return best_text

    updated_segments = []
    for speaker, old_text in merged_segments:
        new_text = find_best_combo_match(old_text, ft_sentences)
        updated_segments.append((speaker, new_text))

    def speaker_id_to_letter(speaker_id):
        try:
            return chr(ord("A") + int(speaker_id))
        except:
            return str(speaker_id)

    # 전체 스크립트 생성
    full_transcript = " ".join([text for _, text in updated_segments])

    # speaker별 JSON 변환
    speaker_json = [
        {"speaker": speaker_id_to_letter(s), "text": t} for s, t in updated_segments
    ]

    return full_transcript, speaker_json

app = FastAPI()

@app.post("/api/analyze-audio")
async def analyze_audio(audio: UploadFile = File(...), userId: str = Form(...)):
    save_path = f"./{audio.filename}"

    # 1. 파일 저장
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"파일 저장 실패: {str(e)}"})

    # 2. 화자 분리 + 자막 분석 실행
    try:
        transcript, speaker_segments = run_full_pipeline(save_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"오디오 처리 실패: {str(e)}"})

    # 3. JSON 포맷으로 결과 생성
    result = {
        "transcript": transcript,
        "speakers": speaker_segments
    }

    return JSONResponse(content=result)

# ngrok 연결
ngrok.set_auth_token("2ycQxO9vUYLivsVwyXYiyDBw59q_6xNH6MmgSaJKMMZ3VM2zx")
public_url = ngrok.connect(8000, domain="tops-beetle-vocal.ngrok-free.app")
print(f"API 주소: {public_url}")

# Colab 환경에 맞게 uvicorn 실행 준비
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)