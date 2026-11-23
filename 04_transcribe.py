"""
Transcribe an audio file with your fine-tuned model.

  python 04_transcribe.py --model outputs/whisper-small-mn/final --audio test.wav
Long recordings are handled automatically (30 s chunks).
"""
import argparse

import torch
from transformers import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--audio", required=True)
args = ap.parse_args()

pipe = pipeline("automatic-speech-recognition", model=args.model,
                device=0 if torch.cuda.is_available() else -1,
                chunk_length_s=30,
                generate_kwargs={"language": "mongolian", "task": "transcribe"})
print(pipe(args.audio)["text"])
