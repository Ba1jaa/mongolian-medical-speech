"""Make REAL Mongolian test recordings with Meta MMS-TTS (facebook/mms-tts-khk).

Reads planned sentences aloud with a synthetic voice and saves them, named as in
recording_plan.csv, into test_wavs/. Use them to test the whole pipeline
(check_data -> 01_prepare_data -> 02_eval_asr -> 04_transcribe) before real speakers record.

Synthetic speech is ONLY for testing. Do not use it in training or in medical_test results.

Run where the model can download (Colab/Kaggle or your machine):
    pip install transformers torch scipy
    python tools/make_tts_test_wavs.py --n 20
    python check_data.py --wav_dir test_wavs
"""
import argparse, csv, os, sys
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import VitsModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="recording_plan.csv")
ap.add_argument("--out", default="test_wavs")
ap.add_argument("--n", type=int, default=20, help="files per speaker")
ap.add_argument("--speakers", default="spk01,spk02")
a = ap.parse_args()

# Use your text normalizer if it is reachable (numbers/units -> words), otherwise raw text.
sys.path += [".", ".."]
normalize = None
try:
    import mn_text
    normalize = next(getattr(mn_text, n) for n in ("normalize", "normalize_text", "normalize_mn")
                     if hasattr(mn_text, n))
    print(f"Using mn_text.{normalize.__name__} for numbers and units")
except Exception:
    print("mn_text.py not found: numbers and units will be read as-is (may sound wrong)")

tok = AutoTokenizer.from_pretrained("facebook/mms-tts-khk")
model = VitsModel.from_pretrained("facebook/mms-tts-khk").eval()
sr_in = model.config.sampling_rate
os.makedirs(a.out, exist_ok=True)
plan = list(csv.DictReader(open(a.plan, encoding="utf-8")))

made = 0
for si, spk in enumerate(a.speakers.split(",")):
    torch.manual_seed(si)                 # different random variation per "speaker"
    model.speaking_rate = [1.0, 0.9, 1.1, 0.95][si % 4]
    for r in [r for r in plan if r["speaker"] == spk][: a.n]:
        text = normalize(r["text"]) if normalize else r["text"]
        with torch.no_grad():
            x = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        x = resample_poly(x, 16000 // np.gcd(sr_in, 16000), sr_in // np.gcd(sr_in, 16000))
        x = np.concatenate([np.zeros(8000), x, np.zeros(8000)])          # 0.5 s silence each side
        x = 0.5 * x / (np.abs(x).max() + 1e-9)
        wavfile.write(os.path.join(a.out, r["path"]), 16000, (x * 32767).astype(np.int16))
        made += 1
        print(r["path"], text)
print(f"Wrote {made} synthetic files to {a.out}/")
