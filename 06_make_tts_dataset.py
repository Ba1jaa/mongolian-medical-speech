"""
Step 6 - Convert your studio recordings into an LJSpeech-style dataset for TTS training.

Input CSV (UTF-8, header required):  path,text
Output:
  <out>/wavs/xxx.wav   (22.05 kHz mono, trimmed, normalized volume)
  <out>/metadata.csv   (id|raw text|normalized text)

  python 06_make_tts_dataset.py --csv my_voice/metadata.csv --wav_dir my_voice/raw --out data/tts
"""
import argparse
import os

import librosa
import pandas as pd
import soundfile as sf

from mn_text import normalize_text

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--wav_dir", default=".")
ap.add_argument("--out", default="data/tts")
ap.add_argument("--sr", type=int, default=22050)
args = ap.parse_args()

os.makedirs(f"{args.out}/wavs", exist_ok=True)
df = pd.read_csv(args.csv)
rows, total = [], 0.0
for i, r in df.iterrows():
    y, _ = librosa.load(os.path.join(args.wav_dir, r["path"]), sr=args.sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    peak = max(abs(y).max(), 1e-6)
    y = y / peak * 0.95
    dur = len(y) / args.sr
    if dur < 1.0 or dur > 15.0:
        continue
    uid = f"mn_{i:05d}"
    sf.write(f"{args.out}/wavs/{uid}.wav", y, args.sr)
    norm = normalize_text(str(r["text"]))
    rows.append(f"{uid}|{r['text']}|{norm}")
    total += dur

with open(f"{args.out}/metadata.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print(f"{len(rows)} clips, {total / 3600:.2f} hours -> {args.out}")
