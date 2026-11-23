"""Assign sentences to speakers and write the recording plan.

Every speaker reads the shared sentences (shared=1 in sentences.csv).
The other sentences are split so each one is read by READS_PER_SENTENCE speakers.

Usage:
    python make_plan.py --speakers 10 --reads 2
Outputs:
    recording_plan.csv          path,text,speaker,sentence_id,category
    speaker_sheets/spkXX.txt    printable list for each speaker (fallback for phone recording)
"""
import argparse, csv, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", default="sentences.csv")
ap.add_argument("--speakers", type=int, default=10)
ap.add_argument("--reads", type=int, default=2, help="how many speakers read each non-shared sentence")
ap.add_argument("--out", default="recording_plan.csv")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.sentences, encoding="utf-8")))
shared = [r for r in rows if r["shared"] == "1"]
rest = [r for r in rows if r["shared"] != "1"]
random.Random(a.seed).shuffle(rest)

spk = [f"spk{i:02d}" for i in range(1, a.speakers + 1)]
plan = {s: list(shared) for s in spk}
# round-robin: sentence k goes to speakers k, k+1, ... (mod N)
for k, r in enumerate(rest):
    for j in range(a.reads):
        plan[spk[(k + j) % a.speakers]].append(r)

os.makedirs("speaker_sheets", exist_ok=True)
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "text", "speaker", "sentence_id", "category"])
    for s in spk:
        lines = []
        for n, r in enumerate(plan[s], 1):
            path = f"{s}_{n:04d}.wav"
            w.writerow([path, r["text"], s, r["id"], r["category"]])
            lines.append(f"{path}\t{r['text']}")
        with open(f"speaker_sheets/{s}.txt", "w", encoding="utf-8") as g:
            g.write(f"Speaker {s}: {len(lines)} sentences. Save each recording with the file name shown.\n\n")
            g.write("\n".join(lines) + "\n")

total = sum(len(v) for v in plan.values())
print(f"{a.speakers} speakers, {total} recordings planned "
      f"(~{total * 5 / 3600:.1f} h at 5 s each), {len(plan[spk[0]])} per speaker")
