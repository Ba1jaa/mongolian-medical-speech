"""
Step 5 - Baseline Mongolian TTS with Meta MMS (Khalkha Mongolian, VITS).

  python 05_tts_baseline.py --text "Өвчтөнд өдөрт 2 удаа 5 мг өгнө" --out hello.wav
"""
import argparse

import scipy.io.wavfile as wavfile
import torch
from transformers import AutoTokenizer, VitsModel

from mn_text import normalize_text

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True)
ap.add_argument("--out", default="tts_out.wav")
ap.add_argument("--model", default="facebook/mms-tts-khk")
args = ap.parse_args()

model = VitsModel.from_pretrained(args.model).eval()
tok = AutoTokenizer.from_pretrained(args.model)

text = normalize_text(args.text)
print("Normalized text:", text)
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0]
wavfile.write(args.out, rate=model.config.sampling_rate, data=wav.numpy())
print("Saved", args.out)
