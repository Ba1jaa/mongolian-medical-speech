# my_data: Task 3 kit (medical-domain speech data)

Everything you need to collect your own medical recordings. The only thing missing is the audio itself.

| File | What it is |
|---|---|
| `sentences.txt` / `sentences.csv` | 503 medical sentences in 7 categories; `shared=1` marks the 20 sentences every speaker reads |
| `make_sentences.py` | Rebuilds the sentence files (edit the lists here) |
| `make_plan.py` | Assigns sentences to speakers and writes `recording_plan.csv` and `speaker_sheets/` |
| `recording_plan.csv` | Planned file name, text and speaker for every recording (10 speakers × ~117 = 1166 files, ~1.6 h) |
| `speaker_sheets/spkXX.txt` | Printable list for each speaker, if they record on a phone instead of the app |
| `record_app.py` | Browser app: shows each sentence, records it and saves `wavs/spkXX_NNNN.wav` at 16 kHz |
| `check_data.py` | Quality check; writes `metadata.csv` (passed files only) and `qc_report.csv` |
| `speakers.csv` | Speaker list to fill in (no names) |
| `consent_form.md` / `.pdf` | Bilingual consent form; one signed copy per speaker |
| `wavs/` | Your recordings go here |

## Steps

1. **Review the sentences.** Have a native speaker (ideally a medical professional) check `sentences.csv`, then run `python make_sentences.py` again if you changed anything. Run all sentences through `mn_text.py` and add any missing units or abbreviations.
2. **Plan the recordings.** Defaults: 10 speakers, each non-shared sentence read by 2 speakers.
   ```bash
   python make_plan.py --speakers 10 --reads 2
   ```
   For more audio, use more speakers or `--reads 3`. 20 speakers with `--reads 3` gives about 4 hours.
3. **Get consent.** Print `consent_form.pdf`, have each speaker sign it, give them a speaker ID and fill in `speakers.csv`.
4. **Record.**
   ```bash
   pip install gradio soundfile numpy scipy
   python record_app.py            # same computer
   python record_app.py --share    # temporary link, so speakers can record on their own device
   ```
   Speakers pick their ID; the app continues where they stopped. Phone alternative: record with any voice-recorder app, name each file as in `speaker_sheets/spkXX.txt`, convert to WAV and put it in `wavs/`.
5. **Check quality.**
   ```bash
   python check_data.py --listen 0.05
   ```
   Re-record anything in `qc_report.csv` that is `too_short`, `silent_or_very_quiet`, `clipped` or `too_long`. Listen to the files it lists and check the audio matches the text. If a speaker said something different, change the text in `recording_plan.csv` and run the check again.
6. **Prepare the dataset** (step 4 of the main guide):
   ```bash
   python 01_prepare_data.py --out data/asr --custom_csv my_data/metadata.csv --custom_dir my_data/wavs
   ```

## Test speakers (step 3.6)
For a fairer `medical_test`, keep 1-2 speakers (for example `spk09`, `spk10`) out of training. `01_prepare_data.py` holds out a random 10% instead, so if you want speaker-based testing, edit it to put those speakers' files in `medical_test`. Say which approach you used in the report.

## Privacy
`.gitignore` keeps the wav files and signed forms out of git. Do not upload `wavs/` or the signed consent forms to GitHub or Hugging Face.

## Test the pipeline first (`tools/`)
Both scripts write to `test_wavs/`, never to `wavs/`, so test files cannot mix with real recordings.

- **Dummy audio (works anywhere, no model):** speech-like sounds, not real words. Tests file handling and the quality check. 4 problems are planted on purpose.
  ```bash
  python tools/make_dummy_wavs.py --n 30
  python check_data.py --wav_dir test_wavs      # should flag spk01_0004, _0008, _0012, and _0016 as missing
  ```
- **Synthetic Mongolian speech (needs internet for the model):** real words read by Meta MMS-TTS. Tests the full chain including ASR.
  ```bash
  pip install transformers torch scipy
  python tools/make_tts_test_wavs.py --n 20
  python check_data.py --wav_dir test_wavs
  python ../01_prepare_data.py --out data/asr_test --custom_csv metadata.csv --custom_dir test_wavs
  ```

When you're done testing, delete `test_wavs/` and run `python check_data.py` again so `metadata.csv` only lists real recordings. Never use the test files for training or for reported results.
