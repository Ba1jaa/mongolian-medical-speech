# Mongolian Medical Speech Recognition (ASR) and Speech Synthesis (TTS)

Speech-to-text and text-to-speech for Mongolian medical systems. The project fine-tunes OpenAI Whisper on public Mongolian speech plus medical sentences I recorded myself, evaluates it on a held-out medical test set, and serves it as a REST API.

```
My medical recordings (my_data/wavs)  +  Common Voice  +  FLEURS
   -> text normalization (mn_text.py)
   -> baseline evaluation (WER / CER)
   -> fine-tune Whisper                  -> evaluate on medical_test
   -> Mongolian TTS (Meta MMS)
   -> FastAPI server (/stt, /tts)
```

**Project plan:** objectives, work packages, milestones, KPIs and risks are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Contents**
1. [What has been done](#1-what-has-been-done)
2. [Project structure](#2-project-structure)
3. [Setup](#3-setup)
4. [How to record data](#4-how-to-record-data)
5. [How to run the code](#5-how-to-run-the-code)
6. [How my recordings are used](#6-how-my-recordings-are-used)
7. [Next steps](#7-next-steps)
8. [Privacy](#8-privacy)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What has been done

| Part | Status |
|---|---|
| Pipeline code (data prep, evaluation, Whisper fine-tune, transcription, TTS, API) | Done (`01_` to `08_` scripts) |
| Mongolian text normalizer: numbers, units (мг, мл…), punctuation | Done (`mn_text.py`) |
| Medical sentence set: **503 sentences** in 7 categories (symptoms, doses, instructions, …), 20 read by every speaker | Done (`my_data/sentences.csv`) |
| Recording plan: 10 speakers × ~117 sentences = **1166 files (~1.6 h)** | Done (`my_data/recording_plan.csv`, `speaker_sheets/`) |
| Bilingual consent form | Done (`my_data/consent_form.pdf`) |
| Browser recording app and quality-check script | Done (`record_app.py`, `check_data.py`) |
| **Recordings, speaker spk01: 117 / 117 files, all passed the quality check, ~7.6 min** | Done (`my_data/wavs/`) |
| Recordings, speakers spk02–spk10 | Not recorded yet (marked `missing` in `qc_report.csv`) |
| Colab notebook that runs the whole ASR pipeline on a free GPU | Done (`notebooks/medical_speech_colab.ipynb`) |
| Whisper fine-tune and results | *Fill in after the training run (see the results table in [section 7](#7-next-steps))* |

---

## 2. Project structure

| File | What it does |
|---|---|
| `requirements.txt` | Python packages |
| `mn_text.py` | Mongolian text normalization, shared by ASR and TTS |
| `01_prepare_data.py` | Downloads FLEURS + Common Voice, adds my recordings, cleans text, saves the dataset |
| `02_eval_asr.py` | Computes WER and CER for any ASR model on any split |
| `03_finetune_whisper.py` | Fine-tunes Whisper on Mongolian |
| `04_transcribe.py` | Transcribes one audio file |
| `05_tts_baseline.py` | Mongolian TTS with Meta MMS (`facebook/mms-tts-khk`) |
| `06_make_tts_dataset.py` | Turns single-speaker recordings into a TTS dataset (optional) |
| `07_train_tts_vits.py` | Trains a custom VITS voice with Coqui TTS (optional) |
| `08_api_server.py` | REST API: `/stt` and `/tts` |
| `notebooks/medical_speech_colab.ipynb` | Runs everything on Google Colab |
| **`my_data/`** | **My own medical speech data** |
| `my_data/sentences.csv` / `.txt` | The 503 medical sentences |
| `my_data/make_plan.py` | Assigns sentences to speakers, writes `recording_plan.csv` and `speaker_sheets/` |
| `my_data/recording_plan.csv` | File name, text and speaker for every planned recording |
| `my_data/speaker_sheets/spkXX.txt` | Printable sentence list per speaker (for phone recording) |
| `my_data/record_app.py` | Browser recording app (Gradio) |
| `my_data/check_data.py` | Quality check; writes `metadata.csv` and `qc_report.csv` |
| `my_data/metadata.csv` | `path,text` for every recording that passed (what training reads) |
| `my_data/qc_report.csv` | Status of every planned file: `ok`, `missing`, `too_short`, … |
| `my_data/speakers.csv` | Speaker info, no names |
| `my_data/wavs/` | The recordings (`spk01_0001.wav` …), 16 kHz mono. Not in git. |
| `my_data/tools/` | Generates fake test audio into `test_wavs/` to test the pipeline |

---

## 3. Setup

### Option A: Google Colab (recommended, free GPU)
1. Open `notebooks/medical_speech_colab.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. On your Mac, zip the recordings (`my_data/wavs` → `wavs.zip`) and upload the zip to Google Drive at `MyDrive/medical-speech/wavs.zip`.
4. Run the cells from top to bottom. The notebook installs packages, unzips the wavs into `my_data/wavs`, runs the quality check, prepares data, evaluates the baseline, fine-tunes, and evaluates again. Datasets, checkpoints and the final model are saved to Google Drive, so a disconnect does not lose the work.

### Option B: your own machine (needs an NVIDIA GPU with 16 GB+ for training)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login            # token from huggingface.co/settings/tokens
```
On macOS install ffmpeg with `brew install ffmpeg`; on Linux use `sudo apt install ffmpeg`.

### Accept the Common Voice terms
Open https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 while logged in and click **Agree**. Without this Common Voice is skipped; FLEURS still works.

---

## 4. How to record data

All commands in this section run inside `my_data/`:
```bash
cd my_data
pip install gradio soundfile numpy scipy
```

### Step 1: Check the sentences and make the plan
The sentences are in `sentences.csv`. Have a native speaker (ideally a medical professional) check them. Then create the plan:
```bash
python make_plan.py --speakers 10 --reads 2
```
Each speaker gets about 117 sentences; every non-shared sentence is read by 2 speakers. For more audio, use more speakers or `--reads 3`.

### Step 2: Consent
Print `consent_form.pdf`, have each speaker sign it, give them an ID (`spk01`, `spk02`, …) and fill in `speakers.csv` (gender, age group, region, device; no names).

### Step 3: Record
```bash
python record_app.py            # opens http://127.0.0.1:7860
python record_app.py --share    # temporary public link, so a speaker can record on their own phone or laptop
```
In the app:
1. Pick your speaker ID and press **Record**. The sentence appears.
2. Read it, press **Stop**, then press play and listen.
3. Good → **Save and next**. The file is saved as `wavs/spkXX_NNNN.wav` (16 kHz mono).
   Bad → **Delete and record again**.
4. **Previous** goes back to listen to or delete a saved sentence. Progress is saved, so you can stop and continue later.

Tips: quiet room, same microphone, 15–30 cm from the mouth, read the text exactly as written.

**Phone alternative:** record with any voice-recorder app using `speaker_sheets/spkXX.txt`, name each file exactly as in the sheet, convert to WAV and put it in `wavs/`.

### Step 4: Quality check
```bash
python check_data.py --listen 0.05
```
This checks every planned file (exists, 1–15 s, not silent, not clipped) and writes:
- `metadata.csv`: only the files that passed. Training reads this.
- `qc_report.csv`: status for every file.

Re-record anything marked `too_short`, `too_long`, `silent_or_very_quiet` or `clipped`. Listen to the files it lists (5% here) to make sure the audio matches the text. If a speaker said something different, fix the text in `recording_plan.csv` and run the check again.

### Testing without real recordings
```bash
python tools/make_dummy_wavs.py --n 30         # fake audio into test_wavs/
python check_data.py --wav_dir test_wavs
```
Delete `test_wavs/` afterwards and run `python check_data.py` again so `metadata.csv` only lists real recordings.

---

## 5. How to run the code

Run these from the project root (not `my_data/`). On Colab the notebook does steps 1–4 for you.

### Step 1: Test the text normalizer
```bash
python mn_text.py
```
Numbers and units should become Mongolian words, e.g. `5 мг` → `тав миллиграмм`.

### Step 2: Prepare the dataset (public data + my recordings)
```bash
python 01_prepare_data.py --out data/asr --custom_csv my_data/metadata.csv --custom_dir my_data/wavs
```
Creates the splits `train`, `validation`, `test` and `medical_test`.

### Step 3: Baseline (before training)
```bash
python 02_eval_asr.py --model openai/whisper-small --data data/asr --split test
python 02_eval_asr.py --model openai/whisper-small --data data/asr --split medical_test
```
Write down the WER and CER. The fine-tuned model has to beat these.

### Step 4: Fine-tune Whisper
```bash
python 03_finetune_whisper.py --data data/asr --out outputs/whisper-small-mn
```
- Default: 4000 steps, effective batch 16, learning rate 1e-5. Several hours on a T4.
- Progress: `tensorboard --logdir outputs/whisper-small-mn`
- Out of memory: `--batch 4 --accum 4`
- The final model is saved to `outputs/whisper-small-mn/final`.

### Step 5: Evaluate and transcribe
```bash
python 02_eval_asr.py --model outputs/whisper-small-mn/final --data data/asr --split test
python 02_eval_asr.py --model outputs/whisper-small-mn/final --data data/asr --split medical_test
python 04_transcribe.py --model outputs/whisper-small-mn/final --audio my_data/wavs/spk01_0001.wav
```

### Step 6: Text-to-speech
```bash
python 05_tts_baseline.py --text "Өвчтөнд өдөрт 2 удаа 5 мг өгнө" --out hello.wav
```

### Step 7: API server
```bash
export ASR_MODEL=outputs/whisper-small-mn/final
export TTS_MODEL=facebook/mms-tts-khk
uvicorn 08_api_server:app --host 0.0.0.0 --port 8000
```
Test it:
```bash
curl -F "file=@my_data/wavs/spk01_0001.wav" http://localhost:8000/stt
curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
     -d '{"text":"Өдөрт 2 удаа 5 мг"}' --output out.wav
```
Interactive docs: http://localhost:8000/docs

### Optional: custom TTS voice
Needs 1+ hours from a single speaker. Put the recordings in `my_voice/raw/` with `my_voice/metadata.csv` (`path,text`), then:
```bash
python 06_make_tts_dataset.py --csv my_voice/metadata.csv --wav_dir my_voice/raw --out data/tts
pip install coqui-tts             # in a separate virtual environment
python 07_train_tts_vits.py --data data/tts --out outputs/vits-mn
```

---

## 6. How my recordings are used

| Step | Code | What happens to `my_data/wavs/` |
|---|---|---|
| Record | `my_data/record_app.py` | Saves each sentence as `wavs/spkXX_NNNN.wav` |
| Check | `my_data/check_data.py` | Checks every file; passed files go into `metadata.csv` |
| Prepare | `01_prepare_data.py`, `load_custom()` | Shuffles the recordings (seed 42): **90% go into `train`** together with FLEURS and Common Voice, **10% become `medical_test`** |
| Train | `03_finetune_whisper.py` | Trains on the `train` split, which includes my recordings |
| Evaluate | `02_eval_asr.py --split medical_test` | Measures WER/CER on the held-out 10% of my recordings, before and after fine-tuning |

With the current 117 files, about 106 are used for training and 11 for `medical_test`. The recordings are **not** used for TTS: `06_make_tts_dataset.py` reads `my_voice/raw/` instead.

Note: the 10% split is random, so the same speaker appears in both train and test. For a fairer test once more speakers are recorded, hold out whole speakers (for example `spk09`, `spk10`) by editing `load_custom()`.

---

## 7. Next steps

1. Record speakers spk02–spk10 (about 1050 more files, ~1.5 h in total).
2. Run `check_data.py` again, then retrain on Colab.
3. Hold out 1–2 whole speakers for `medical_test`.
4. Fill in the results:

| Model | WER `test` | CER `test` | WER `medical_test` | CER `medical_test` |
|---|---|---|---|---|
| whisper-small (baseline) | | | | |
| whisper-small fine-tuned | | | | |

5. Error analysis: which words fail (drug names, numbers?), then add sentences with those words.
6. TTS listening test (MOS) and an API demo for the final report.

---

## 8. Privacy

- Every speaker signs the consent form; `speakers.csv` has no names.
- `.gitignore` keeps `wavs/*.wav` and signed forms out of git. Never upload the recordings to GitHub or Hugging Face.
- No real patient data is used.
- The system is an aid; a human must check transcribed doses and diagnoses.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `trust_remote_code` or dataset script error | Keep `datasets<4.0` (already in `requirements.txt`) |
| Common Voice 401/403 | `huggingface-cli login` and accept the dataset terms |
| CUDA out of memory | Lower `--batch`, raise `--accum` |
| Record app does not hear the mic | Allow microphone access in the browser; with `--share` use the https link |
| `check_data.py` says `missing` | That speaker has not recorded yet, or the file name does not match `recording_plan.csv` |
| Whisper repeats words / hallucinates | More data; remove silent or very short clips |
| mp3 cannot be read | Install ffmpeg |

The previous, longer guide is kept in `README_old.md`.
