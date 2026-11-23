"""Quality check for Task 3 (step 3.5) and metadata.csv builder.

Checks every planned file in wavs/: exists, readable, 1-15 s long, not silent, not clipped.
Writes:
    metadata.csv    path,text   (only files that passed; this is what 01_prepare_data.py reads)
    qc_report.csv   one row per planned file with its status
and prints hours and file counts per speaker for your report.

    python check_data.py
    python check_data.py --listen 0.05    # also list a random 5% of files to listen to
Files in wavs/ that are not in the plan are reported too (add them to the plan or delete them).
"""
import argparse, csv, os, random
from collections import defaultdict
import numpy as np

try:
    import soundfile as sf

    def load(p):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
except ImportError:  # fallback: WAV only
    from scipy.io import wavfile

    def load(p):
        sr, x = wavfile.read(p)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        else:
            x = x.astype(np.float32)
        return (x.mean(axis=1) if x.ndim > 1 else x), sr

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="recording_plan.csv")
ap.add_argument("--wav_dir", default="wavs")
ap.add_argument("--min_sec", type=float, default=1.0)
ap.add_argument("--max_sec", type=float, default=15.0)
ap.add_argument("--listen", type=float, default=0.0, help="fraction of OK files to list for listening")
a = ap.parse_args()

plan = list(csv.DictReader(open(a.plan, encoding="utf-8")))
planned = {r["path"] for r in plan}
report, ok_rows = [], []
hours, counts, issues = defaultdict(float), defaultdict(int), defaultdict(int)

for r in plan:
    p = os.path.join(a.wav_dir, r["path"])
    status, dur = "ok", 0.0
    if not os.path.exists(p):
        status = "missing"
    else:
        try:
            x, sr = load(p)
            dur = len(x) / sr
            peak = float(np.abs(x).max()) if len(x) else 0.0
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            if dur < a.min_sec:
                status = "too_short"
            elif dur > a.max_sec:
                status = "too_long"
            elif rms < 0.003:
                status = "silent_or_very_quiet"
            elif peak >= 0.999:
                status = "clipped"
        except Exception as e:
            status = f"unreadable: {e.__class__.__name__}"
    report.append([r["path"], r["speaker"], status, f"{dur:.2f}", r["text"]])
    if status == "ok":
        ok_rows.append([r["path"], r["text"]])
        hours[r["speaker"]] += dur / 3600
        counts[r["speaker"]] += 1
    elif status != "missing":
        issues[r["speaker"]] += 1

with open("metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "text"]); w.writerows(ok_rows)
with open("qc_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "speaker", "status", "seconds", "text"]); w.writerows(report)

speakers = sorted({r["speaker"] for r in plan})
planned_per = defaultdict(int)
for r in plan:
    planned_per[r["speaker"]] += 1
print(f"{'speaker':8} {'ok':>5} {'planned':>8} {'problems':>9} {'hours':>6}")
for s in speakers:
    print(f"{s:8} {counts[s]:5d} {planned_per[s]:8d} {issues[s]:9d} {hours[s]:6.2f}")
print(f"TOTAL    {len(ok_rows):5d} {len(plan):8d} {sum(issues.values()):9d} {sum(hours.values()):6.2f}")
print(f"Speakers with at least one recording: {sum(1 for s in speakers if counts[s])}")

extra = sorted(set(os.listdir(a.wav_dir)) - planned) if os.path.isdir(a.wav_dir) else []
if extra:
    print(f"\n{len(extra)} files in {a.wav_dir}/ are not in the plan, e.g. {extra[:5]}")
print("\nWrote metadata.csv (passed files only) and qc_report.csv (all statuses).")
if a.listen and ok_rows:
    k = max(1, int(len(ok_rows) * a.listen))
    print(f"\nListen to these {k} files and check the audio matches the text:")
    for p, t in random.sample(ok_rows, k):
        print(f"  {p}\t{t}")
