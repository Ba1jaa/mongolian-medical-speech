"""Browser recording app for Task 3.

1. Pick your speaker ID and press Record: the sentence appears.
2. Press Stop when you finish, then press play to listen to yourself.
3. Good?  Press "Save and next": saved as wavs/<planned file name>.wav (e.g. wavs/spk01_0001.wav, 16 kHz mono).
   Not good?  Press "Delete and record again".
Going back with "Previous" lets you listen to (or delete) a sentence you already saved.
Progress is saved, so you can stop and continue later.

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
BOX = "font-size:30px;line-height:1.4;padding:24px;border-radius:12px;text-align:center;min-height:120px"


def items(spk):
    return [r for r in PLAN if r["speaker"] == spk]


def wav_path(spk, i):
    return os.path.join(a.wav_dir, items(spk)[i]["path"])


def done(path):
    return os.path.exists(os.path.join(a.wav_dir, path))


def first_todo(spk):
    its = items(spk)
    for i, r in enumerate(its):
        if not done(r["path"]):
            return i
    return len(its) - 1


def header(spk, i, note=""):
    its = items(spk)
    r = its[i]
    n_done = sum(done(x["path"]) for x in its)
    status = ("✅ already saved: press play to listen, or delete and record again"
              if done(r["path"]) else "⏺ not recorded yet")
    h = (f"**{spk}** · sentence {i + 1} of {len(its)} · recorded {n_done}/{len(its)}  \n"
         f"File name: **`{r['path']}`** · {status}")
    return (note + "\n\n" + h) if note else h


def hidden_text():
    return (f"<div style='{BOX};border:2px dashed #999;color:#888'>"
            "🎙 Press <b>Record</b> to see the sentence</div>")


def shown_text(spk, i, mode="rec"):
    t = items(spk)[i]["text"]
    if mode == "rec":
        return f"<div style='{BOX};border:2px solid #e5484d'>🔴 {t}</div>"
    return (f"<div style='{BOX};border:2px solid #3b82f6'>{t}"
            "<div style='font-size:16px;color:#888;margin-top:8px'>▶ Listen to your recording, then save or delete it</div></div>")


def load_saved(spk, i):
    p = wav_path(spk, i)
    if not os.path.exists(p):
        return None
    x, sr = sf.read(p, dtype="int16")
    return (sr, x)


def view(spk, i, note=""):
    i = max(0, min(i, len(items(spk)) - 1))
    saved = load_saved(spk, i)
    text = shown_text(spk, i, "review") if saved is not None else hidden_text()
    return i, header(spk, i, note), text, saved


def start(spk, i):
    return shown_text(spk, i, "rec")


def stopped(spk, i):
    return shown_text(spk, i, "review")


def save(spk, i, audio):
    if audio is None:
        gr.Warning("Nothing to save. Press Record first.")
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
        gr.Warning(f"Too short ({dur:.1f} s). Delete it and record again.")
        return gr.update(), gr.update(), gr.update(), gr.update()
    if np.abs(x).max() >= 0.99:
        gr.Warning("The recording is clipped (too loud). Move a bit further from the mic; use ◀ Previous to redo it.")
    name = items(spk)[i]["path"]
    sf.write(wav_path(spk, i), x, SR, subtype="PCM_16")
    gr.Info(f"Saved {name} ({dur:.1f} s)")
    j = min(i + 1, len(items(spk)) - 1)
    nxt = view(spk, j, note=f"💾 Saved **`{name}`** ({dur:.1f} s)")
    if j != i:                          # next sentence: start fresh, text hidden until Record
        nxt = (nxt[0], nxt[1], hidden_text() if nxt[3] is None else nxt[2], nxt[3])
    return nxt


def delete(spk, i):
    p = wav_path(spk, i)
    name = items(spk)[i]["path"]
    if os.path.exists(p):
        os.remove(p)
        note = f"🗑 Deleted **`{name}`**. Press Record to record it again."
    else:
        note = "🗑 Recording discarded. Press Record to record it again."
    return i, header(spk, i, note), hidden_text(), None


def saved_list(spk):
    rows = []
    for n, r in enumerate(items(spk), 1):
        p = os.path.join(a.wav_dir, r["path"])
        if os.path.exists(p):
            try:
                secs = round(sf.info(p).duration, 1)
            except Exception:
                secs = None
            rows.append([n, r["path"], secs, r["text"]])
    total = sum(x[2] or 0 for x in rows)
    label = f"📂 Saved recordings for {spk}: {len(rows)} of {len(items(spk))} files · {total / 60:.1f} min"
    return rows, gr.update(label=label)


def open_row(spk, evt: gr.SelectData):
    row = getattr(evt, "row_value", None)
    if row:
        return view(spk, int(row[0]) - 1)
    return view(spk, saved_list(spk)[0][evt.index[0]][0] - 1)


with gr.Blocks(title="Medical speech recording") as demo:
    gr.Markdown("## Mongolian medical speech recording\n"
                "1. Choose your speaker ID.  2. Press **Record**: the sentence appears. Read it, then press **Stop**.  \n"
                "3. Press ▶ to **listen**. Good → **Save and next**. Not good → **Delete and record again**.  \n"
                "Leave about half a second of silence before and after.")
    idx = gr.State(0)
    spk = gr.Dropdown(SPEAKERS, value=SPEAKERS[0], label="Speaker ID")
    head = gr.Markdown()
    text = gr.HTML()
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Recording")
    with gr.Row():
        del_b = gr.Button("🗑 Delete and record again", variant="stop")
        save_b = gr.Button("✅ Save and next", variant="primary")
    with gr.Row():
        prev_b = gr.Button("◀ Previous")
        skip_b = gr.Button("Skip ▶")
    with gr.Accordion("📂 Saved recordings", open=True) as acc:
        gr.Markdown("Click a row to open that sentence: you can listen to it, or delete it and record again.")
        table = gr.Dataframe(headers=["#", "File", "Seconds", "Sentence"], datatype=["number", "str", "number", "str"],
                             interactive=False, wrap=True, max_height=400)
    outs = [idx, head, text, mic]
    tbl = [table, acc]
    spk.change(lambda s: view(s, first_todo(s)), spk, outs).then(saved_list, spk, tbl)
    prev_b.click(lambda s, i: view(s, i - 1), [spk, idx], outs)
    skip_b.click(lambda s, i: view(s, i + 1), [spk, idx], outs)
    mic.start_recording(start, [spk, idx], text)
    mic.stop_recording(stopped, [spk, idx], text)
    save_b.click(save, [spk, idx, mic], outs).then(saved_list, spk, tbl)
    del_b.click(delete, [spk, idx], outs).then(saved_list, spk, tbl)
    table.select(open_row, spk, outs)
    demo.load(lambda s: view(s, first_todo(s)), spk, outs).then(saved_list, spk, tbl)

demo.launch(share=a.share)
