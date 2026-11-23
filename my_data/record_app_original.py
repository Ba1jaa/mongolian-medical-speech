"""Browser recording app for Task 3.

Shows each planned sentence for the chosen speaker, records it from the microphone,
and saves it as wavs/<file>.wav (16 kHz mono). Progress is saved, so you can stop and continue.

    pip install gradio soundfile numpy scipy
    python record_app.py            # opens http://127.0.0.1:7860
    python record_app.py --share    # temporary public link, so speakers can record from their own phone or laptop

After recording, run check_data.py to build metadata.csv.
"""
import argparse, csv, os
import numpy as np
import soundfile as sf
import gradio as gr
from scipy.signal import resample_poly

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="recording_plan.csv")
ap.add_argument("--wav_dir", default="wavs")
ap.add_argument("--share", action="store_true")
a = ap.parse_args()
os.makedirs(a.wav_dir, exist_ok=True)

PLAN = list(csv.DictReader(open(a.plan, encoding="utf-8")))
SPEAKERS = sorted({r["speaker"] for r in PLAN})
SR = 16000


def items(spk):
    return [r for r in PLAN if r["speaker"] == spk]


def done(path):
    return os.path.exists(os.path.join(a.wav_dir, path))


def first_todo(spk):
    its = items(spk)
    for i, r in enumerate(its):
        if not done(r["path"]):
            return i
    return len(its) - 1


def view(spk, i):
    its = items(spk)
    i = max(0, min(i, len(its) - 1))
    r = its[i]
    n_done = sum(done(x["path"]) for x in its)
    status = "✅ already recorded (recording again overwrites it)" if done(r["path"]) else "⏺ not recorded yet"
    head = f"**{spk}** · sentence {i + 1} of {len(its)} · recorded {n_done}/{len(its)} · `{r['path']}` · {status}"
    return i, head, f"<div style='font-size:28px;line-height:1.4;padding:16px 0'>{r['text']}</div>", None


def save(spk, i, audio):
    if audio is None:
        gr.Warning("Record the sentence first.")
        return view(spk, i)
    sr, x = audio
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.0:          # int16 range from the browser
        x = x / 32768.0
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    dur = len(x) / SR
    if dur < 1.0:
        gr.Warning(f"Too short ({dur:.1f} s). Please record again.")
        return view(spk, i)
    if np.abs(x).max() >= 0.99:
        gr.Warning("The recording is clipped (too loud). Move a bit further from the mic and record again.")
    sf.write(os.path.join(a.wav_dir, items(spk)[i]["path"]), x, SR, subtype="PCM_16")
    return view(spk, i + 1)


with gr.Blocks(title="Medical speech recording") as demo:
    gr.Markdown("## Mongolian medical speech recording\nChoose your speaker ID, read the sentence naturally, "
                "then press **Save and next**. Leave about half a second of silence before and after.")
    idx = gr.State(0)
    spk = gr.Dropdown(SPEAKERS, value=SPEAKERS[0], label="Speaker ID")
    head = gr.Markdown()
    text = gr.HTML()
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Recording")
    with gr.Row():
        prev_b = gr.Button("◀ Previous")
        skip_b = gr.Button("Skip ▶")
        save_b = gr.Button("Save and next", variant="primary")
    outs = [idx, head, text, mic]
    spk.change(lambda s: view(s, first_todo(s)), spk, outs)
    prev_b.click(lambda s, i: view(s, i - 1), [spk, idx], outs)
    skip_b.click(lambda s, i: view(s, i + 1), [spk, idx], outs)
    save_b.click(save, [spk, idx, mic], outs)
    demo.load(lambda s: view(s, first_todo(s)), spk, outs)

demo.launch(share=a.share)
