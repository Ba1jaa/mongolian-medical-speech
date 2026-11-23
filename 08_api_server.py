"""
Step 8 - One REST API for both models, so the health information system can call them.

  export ASR_MODEL=outputs/whisper-small-mn/final      # or openai/whisper-small
  export TTS_MODEL=facebook/mms-tts-khk
  uvicorn 08_api_server:app --host 0.0.0.0 --port 8000

Test:
  curl -F "file=@test.wav" http://localhost:8000/stt
  curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
       -d '{"text":"Өдөрт 2 удаа 5 мг"}' --output out.wav
"""
import io
import os

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoTokenizer, VitsModel, pipeline

from mn_text import normalize_text

ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-small")
TTS_MODEL = os.getenv("TTS_MODEL", "facebook/mms-tts-khk")

app = FastAPI(title="Mongolian Speech API")

asr = pipeline("automatic-speech-recognition", model=ASR_MODEL,
               device=0 if torch.cuda.is_available() else -1, chunk_length_s=30,
               generate_kwargs={"language": "mongolian", "task": "transcribe"})
tts_model = VitsModel.from_pretrained(TTS_MODEL).eval()
tts_tok = AutoTokenizer.from_pretrained(TTS_MODEL)


class TTSRequest(BaseModel):
    text: str


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    data = await file.read()
    audio, _ = librosa.load(io.BytesIO(data), sr=16000, mono=True)
    return {"text": asr({"raw": audio, "sampling_rate": 16000})["text"]}


@app.post("/tts")
def tts(req: TTSRequest):
    text = normalize_text(req.text)
    inputs = tts_tok(text, return_tensors="pt")
    with torch.no_grad():
        wav = tts_model(**inputs).waveform[0].numpy()
    buf = io.BytesIO()
    sf.write(buf, wav, tts_model.config.sampling_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok"}
