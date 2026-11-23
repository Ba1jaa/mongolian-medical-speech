# Mongolian Speech-to-Text and Text-to-Speech for Medical Systems
Step-by-step guide, from zero to a working API. Everything here uses free tools and pretrained models.

**Pipeline overview**

```
Data (Common Voice + FLEURS + your medical recordings)
   -> text normalization (mn_text.py)
   -> baseline evaluation (WER/CER)
   -> fine-tune Whisper (ASR)             -> evaluate on medical test set
   -> baseline MMS-TTS -> optional custom VITS voice
   -> FastAPI server (/stt, /tts)         -> report
```

## Files in this project

| File | What it does |
|---|---|
| `requirements.txt` | Python packages |
| `mn_text.py` | Mongolian text normalization: numbers, units (мг, мл...), punctuation |
| `01_prepare_data.py` | Downloads and merges datasets, cleans text, saves to disk |
| `02_eval_asr.py` | Computes WER and CER for any ASR model |
| `03_finetune_whisper.py` | Fine-tunes Whisper on Mongolian |
| `04_transcribe.py` | Transcribes an audio file with your model |
| `05_tts_baseline.py` | Mongolian TTS with Meta MMS |
| `06_make_tts_dataset.py` | Converts your studio recordings into a TTS dataset |
| `07_train_tts_vits.py` | Trains a custom VITS voice with Coqui TTS |
| `08_api_server.py` | REST API for both models |

---

## 0. Realistic timeline

Your form says training runs 2026-10-25 to 2026-12-25 (fix the end year typo, it says 2025). With about 5 weeks left from 2026-11-20, a workable plan is:

| Week | Goal |
|---|---|
| 1 (Nov 21-27) | Environment, data download, normalization, baselines (steps 1-4) |
| 2 (Nov 28-Dec 4) | Record medical sentences, first Whisper fine-tune (step 5-6) |
| 3 (Dec 5-11) | Evaluate, improve data, second fine-tune run (step 7) |
| 4 (Dec 12-18) | TTS baseline, text front-end, optional custom voice (steps 8-9) |
| 5 (Dec 19-25) | API, integration demo, final report and certificate form (steps 10-11) |

Do the **baseline first** and the API early, so you always have something working. Custom TTS training is the riskiest part, so treat it as optional.

---

## 1. Setup

### Option A: Google Colab / Kaggle (recommended, free GPU)
1. Colab: Runtime > Change runtime type > **T4 GPU**. Kaggle: Settings > Accelerator > **GPU T4 x2**.
2. Upload this project folder (or clone it from your own GitHub repo).
3. Run:
```bash
pip install -r requirements.txt
huggingface-cli login        # paste a token from huggingface.co/settings/tokens
```
Colab disconnects after some hours, so save checkpoints to Google Drive:
```python
from google.colab import drive; drive.mount('/content/drive')
```
and use `--out /content/drive/MyDrive/whisper-mn` and `--data /content/drive/MyDrive/data_asr`.

### Option B: your own machine or university server
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
sudo apt install ffmpeg                              # needed to read mp3
huggingface-cli login
```
Needs an NVIDIA GPU with 16 GB+ VRAM for whisper-small. CPU-only is too slow for training.

### Accept dataset terms
Open https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 while logged in and click **Agree** (otherwise Common Voice will be skipped; FLEURS still works).

---

## 2. Test the text normalizer
```bash
python mn_text.py
```
Expected: numbers and units become Mongolian words, e.g. `5 мг` -> `тав миллиграмм`.
**Ask a native speaker to review** the number rules and add your own medical abbreviations to `ABBREVIATIONS` in `mn_text.py`. This file is shared by ASR training and TTS, so errors here affect both.

---

## 3. Collect your own medical-domain data (most valuable step)

Public data is general speech. Medical words (drug and disease names, doses) are what your system must get right, so you need your own recordings.

1. **Write 500-2000 sentences** in Mongolian Cyrillic: symptoms, instructions, doses, appointment phrases, statistics terms. Include numbers and units. Get sentences from public health guidelines, textbooks, and your supervisor's domain. No real patient data.
2. **Record** them with 5-20 different speakers (colleagues, friends) on phone or laptop mic in a quiet room. Get written consent from each speaker.
   - WAV or FLAC preferred, mono, any sample rate (scripts resample).
   - One sentence per file, 2-15 seconds.
3. **Create `my_data/metadata.csv`**:
```csv
path,text
rec_0001.wav,Өвчтөнд өдөрт 2 удаа 5 мг эм өгнө
rec_0002.wav,Цусны даралт 120 хүртэл буурсан
```
   Put wavs in `my_data/wavs/`. Write text exactly as spoken; numbers are fine because they get normalized.

Aim for 3-10 hours of medical-domain audio for a visible improvement. Even 1 hour helps evaluation.

---

## 4. Prepare the dataset
```bash
# public data only
python 01_prepare_data.py --out data/asr

# public + your own data (recommended)
python 01_prepare_data.py --out data/asr --custom_csv my_data/metadata.csv --custom_dir my_data/wavs
```
This produces splits `train`, `validation`, `test`, and `medical_test` (10 percent of your own recordings, held out). It takes a while because it decodes all audio.

---

## 5. Baselines (do this BEFORE training)
```bash
python 02_eval_asr.py --model openai/whisper-small  --data data/asr --split test
python 02_eval_asr.py --model facebook/mms-1b-all   --data data/asr --split test
python 02_eval_asr.py --model openai/whisper-small  --data data/asr --split medical_test
```
Write the numbers down. The point of the project is to show your fine-tuned model beats these, especially on `medical_test`.
Typical result: Whisper-small has a very high WER on Mongolian before fine-tuning; MMS is much better. If MMS beats Whisper after your fine-tune, that is a valid finding. Report it honestly.

---

## 6. Fine-tune Whisper (ASR)
```bash
python 03_finetune_whisper.py --data data/asr --out outputs/whisper-small-mn
```
- Default: 4000 steps, effective batch 16, learning rate 1e-5. On a T4 this takes several hours.
- Watch progress: `tensorboard --logdir outputs/whisper-small-mn`.
- Crash or Colab disconnect: set `trainer.train(resume_from_checkpoint=True)` in the script and rerun.
- Out-of-memory: `--batch 4 --accum 4`, or reduce `MAX_SEC` in `01_prepare_data.py`.
- Better GPU: `--model openai/whisper-medium --batch 4 --accum 4`.
- If the loss goes down but WER stays high, you probably have text/audio mismatches in your data. Listen to a few samples.

Result: `outputs/whisper-small-mn/final`.

---

## 7. Evaluate and improve
```bash
python 02_eval_asr.py --model outputs/whisper-small-mn/final --data data/asr --split test
python 02_eval_asr.py --model outputs/whisper-small-mn/final --data data/asr --split medical_test
python 04_transcribe.py --model outputs/whisper-small-mn/final --audio some_recording.wav
```
Ways to improve, in order of impact:
1. **More medical-domain data** and more speakers.
2. **Data augmentation**: speed perturbation (0.9x/1.1x), background noise, to your own recordings.
3. **More steps** (8000-10000) if validation WER is still falling.
4. **Bigger model** (whisper-medium) if you have the GPU.
5. **Error analysis**: print the worst 50 utterances, see which words fail (drug names? numbers?), and add targeted sentences with those words.
6. Alternative: fine-tune MMS/wav2vec2 with a CTC head (lighter, and often strong for low-resource languages). Compare both in your report.

---

## 8. TTS baseline (MMS)
```bash
python 05_tts_baseline.py --text "Өвчтөнд өдөрт 2 удаа 5 мг өгнө" --out hello.wav
```
This uses `facebook/mms-tts-khk` (Khalkha Mongolian). Listen to it and collect problem words. Common issues are numbers, abbreviations, and foreign drug names, which is why the text front-end (`mn_text.py`) matters. Improve it until your 50 test sentences sound right.

---

## 9. Custom TTS voice (optional, advanced)

### 9a. Record the voice
- One speaker, quiet room, same mic and distance for everything, reading from text.
- 5-20 hours for training from scratch; 1-3 hours may be enough when fine-tuning a pretrained model (9b).
- Create `my_voice/metadata.csv` (`path,text`) and run:
```bash
python 06_make_tts_dataset.py --csv my_voice/metadata.csv --wav_dir my_voice/raw --out data/tts
```

### 9b. Choose a training route
**Route 1: fine-tune MMS-TTS (best for small data).** Use the community repo `ylacombe/finetune-hf-vits` (Hugging Face VITS fine-tuning). Follow its README: convert the checkpoint with its script, then run its training script on your dataset. Check the repo for current instructions since they change.

**Route 2: train VITS from scratch with Coqui (needs ~10+ hours of audio):**
```bash
# in a separate virtual environment
pip install coqui-tts
python 07_train_tts_vits.py --data data/tts --out outputs/vits-mn
```
Test during training: the script generates samples for `test_sentences` and logs audio to TensorBoard.
Synthesize afterwards:
```bash
tts --text "сайн байна уу" --model_path outputs/vits-mn/<run>/best_model.pth \
    --config_path outputs/vits-mn/<run>/config.json --out_path out.wav
```
Evaluation: listening test (ask 10 people to rate 1-5 for naturalness, the MOS), plus intelligibility: run your fine-tuned ASR on the TTS output and compute WER against the input text.

---

## 10. Serve both models as an API
```bash
export ASR_MODEL=outputs/whisper-small-mn/final
export TTS_MODEL=facebook/mms-tts-khk
uvicorn 08_api_server:app --host 0.0.0.0 --port 8000
```
Test:
```bash
curl -F "file=@test.wav" http://localhost:8000/stt
curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" \
     -d '{"text":"Өдөрт 2 удаа 5 мг"}' --output out.wav
```
Interactive docs: http://localhost:8000/docs. A health information system can call these endpoints from any language. For production, add authentication and HTTPS, and never log patient audio.

---

## 11. Report and deliverables
Your supervisor and the Institute of Informatics will want evidence. Include:
1. Problem and scope (ASR and TTS for Mongolian medical systems).
2. Data: sources, hours, speaker count, consent, anonymization.
3. Method: models, hyperparameters, text normalization rules.
4. Results table: baseline vs fine-tuned, WER/CER on `test` and `medical_test`; TTS MOS and ASR-on-TTS WER.
5. Error analysis with examples, limitations, future work.
6. Demo: the API plus a short video or a simple Gradio page.
7. Upload the filled training form to the MAG Praktikum system as your form says.

## 12. Privacy and ethics checklist
- Written consent from every speaker; explain how recordings are used.
- No real patient recordings or names in training data without formal approval from the Center for Health Development.
- Keep raw recordings private (not in public GitHub or Hugging Face repos).
- Say clearly that the system is an aid and not a replacement for clinical judgment; a human should verify transcriptions of doses and diagnoses.

## 13. Troubleshooting
| Problem | Fix |
|---|---|
| `trust_remote_code` or dataset script error | Keep `datasets<4.0` (already in requirements) |
| Common Voice 401/403 | Log in with `huggingface-cli login` and accept the dataset terms |
| CUDA out of memory | Lower `--batch`, raise `--accum`, use whisper-small |
| Whisper outputs another language or English | Check `language="mongolian"` is set (it is in the scripts) |
| Repeated words or hallucinations | More fine-tuning data; filter silent or very short clips |
| Numbers read badly in TTS | Fix rules in `mn_text.py` and have a native speaker check |
| mp3 cannot be read | Install ffmpeg |
