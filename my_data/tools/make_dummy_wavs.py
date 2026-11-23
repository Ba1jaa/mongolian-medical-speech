"""Make DUMMY recordings to test the pipeline (not real speech, never use for training).

Writes speech-like audio (voiced 'syllables' with pauses) named exactly as in recording_plan.csv,
into a separate folder (default test_wavs/) so it never mixes with real recordings.
A few files are made bad on purpose so you can see check_data.py catch them.

    python tools/make_dummy_wavs.py --n 30
    python check_data.py --wav_dir test_wavs
"""
import argparse, csv, os
import numpy as np
from scipy.io import wavfile

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="recording_plan.csv")
ap.add_argument("--out", default="test_wavs")
ap.add_argument("--n", type=int, default=30, help="files per speaker")
ap.add_argument("--speakers", default="spk01,spk02")
a = ap.parse_args()

SR = 16000
rng = np.random.default_rng(0)
os.makedirs(a.out, exist_ok=True)
plan = list(csv.DictReader(open(a.plan, encoding="utf-8")))


def syllable(f0, dur):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))            # slight pitch movement
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 8))            # harmonics like a voice
    x += 0.05 * rng.standard_normal(len(t))
    return x * np.hanning(len(t))


def fake_speech(text, f0):
    parts = [np.zeros(int(0.5 * SR))]
    for word in text.split():
        for _ in range(max(1, len(word) // 3)):
            parts.append(syllable(f0 * rng.uniform(0.9, 1.15), rng.uniform(0.12, 0.22)))
        parts.append(np.zeros(int(rng.uniform(0.05, 0.15) * SR)))
    parts.append(np.zeros(int(0.5 * SR)))
    x = np.concatenate(parts)
    return 0.3 * x / np.abs(x).max()


made = 0
for si, spk in enumerate(a.speakers.split(",")):
    f0 = [120, 210, 150, 190][si % 4]                       # different "voice" per speaker
    rows = [r for r in plan if r["speaker"] == spk][: a.n]
    for i, r in enumerate(rows):
        x = fake_speech(r["text"], f0)
        if si == 0 and i == 3:   x = x[: int(0.5 * SR)]                       # too short
        if si == 0 and i == 7:   x = x * 0.001                                # silent
        if si == 0 and i == 11:  x = np.clip(x * 8, -1, 1)                    # clipped
        if si == 0 and i == 15:  continue                                     # missing
        wavfile.write(os.path.join(a.out, r["path"]), SR, (x * 32767).astype(np.int16))
        made += 1
print(f"Wrote {made} dummy files to {a.out}/ (4 problems planted in {a.speakers.split(',')[0]}: "
      f"#4 too short, #8 silent, #12 clipped, #16 missing)")
