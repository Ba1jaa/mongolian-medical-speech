# Project Plan: Mongolian Medical Speech Recognition and Synthesis

> How to set up and run the code: see [README.md](README.md).

## 1. Project overview

| Item | Description |
|---|---|
| **Project name** | Mongolian Medical Speech: ASR + TTS |
| **Goal** | Build a speech-to-text and text-to-speech system for Mongolian medical use, served as a REST API |
| **Base models** | OpenAI Whisper (ASR), Meta MMS-TTS `facebook/mms-tts-khk` (TTS) |
| **Data** | FLEURS, Common Voice, and the project's own medical recordings (10 speakers) |
| **Compute** | Google Colab (free T4 GPU) |
| **Current status** | Pipeline code done. Speaker 1 of 10 recorded (117 of 1166 files). No training results yet. |

## 2. Objectives

1. **ASR:** a fine-tuned Whisper model with lower WER/CER than the baseline on a medical test set.
2. **TTS:** clear Mongolian speech that reads medical text correctly, including numbers, doses and units.
3. **Dataset:** about 1.6 hours of checked, consented Mongolian medical speech.
4. **System:** a working API (`/stt`, `/tts`) and a demo.
5. **Report:** results, analysis and limitations.

## 3. Scope

**In scope**
- Mongolian (Khalkha Cyrillic) medical sentences: symptoms, doses, instructions
- Fine-tuning Whisper, using MMS-TTS as it is
- Text normalization (numbers, units, punctuation)
- REST API and demo interface

**Out of scope**
- Real patient data or clinical deployment
- Real-time streaming recognition
- Custom TTS voice (optional, stretch goal)
- Mobile app

## 4. Work breakdown structure (WBS)

### WP1: Project setup
| ID | Task | Output |
|---|---|---|
| 1.1 | Merge duplicate folders into one main folder | Clean project structure |
| 1.2 | Fix the folder name `my_data ` (trailing space) | Working paths |
| 1.3 | Set up the Colab environment and Hugging Face login | Working notebook |
| 1.4 | Accept the Common Voice terms | Dataset access |

### WP2: Data collection
| ID | Task | Output |
|---|---|---|
| 2.1 | Medical professional reviews the sentences | Approved `sentences.csv` |
| 2.2 | Collect signed consent forms | Signed forms, `speakers.csv` |
| 2.3 | Record spk01 | ✅ Done (117 files) |
| 2.4 | Record spk02–spk10 | ~1050 files |
| 2.5 | Quality check and re-recording | `metadata.csv`, `qc_report.csv` |
| 2.6 | Listen to a 5% sample to confirm audio matches text | Fixed transcripts |

### WP3: Data preparation
| ID | Task | Output |
|---|---|---|
| 3.1 | Test the text normalizer | Normalizer works |
| 3.2 | Merge FLEURS, Common Voice and own data | `data/asr` dataset |
| 3.3 | Hold out whole speakers for testing (spk09, spk10) | Fair `medical_test` split |

### WP4: ASR model
| ID | Task | Output |
|---|---|---|
| 4.1 | Baseline evaluation of whisper-small | Baseline WER/CER |
| 4.2 | First fine-tune (spk01 data) | Pipeline test model |
| 4.3 | Full fine-tune (all speakers) | Main model |
| 4.4 | Try whisper-medium (if GPU allows) | Comparison |
| 4.5 | Error analysis (drug names, numbers, units) | Error report |
| 4.6 | Add targeted sentences and retrain | Improved model |

### WP5: TTS
| ID | Task | Output |
|---|---|---|
| 5.1 | MMS-TTS baseline with the normalizer | Audio samples |
| 5.2 | Listening test (MOS, 5–10 listeners, 20 sentences) | MOS score |
| 5.3 | Custom VITS voice (optional) | Custom voice model |

### WP6: System integration
| ID | Task | Output |
|---|---|---|
| 6.1 | API with baseline models | Working `/stt`, `/tts` |
| 6.2 | Switch in the fine-tuned ASR model | Final API |
| 6.3 | Gradio demo page | Demo |
| 6.4 | Test with sample medical audio | Test results |

### WP7: Documentation and report
| ID | Task | Output |
|---|---|---|
| 7.1 | Fill in the results table | README results |
| 7.2 | Write the final report | Report |
| 7.3 | Prepare the presentation and demo | Slides, demo video |

## 5. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | **Setup complete** | One clean project folder, Colab runs |
| M2 | **Baseline ready** | Baseline WER/CER recorded, API runs with stock models |
| M3 | **Pipeline proven** | First fine-tune finishes end to end |
| M4 | **Data collected** | 10 speakers recorded and checked (~1.6 h) |
| M5 | **Main model trained** | Fine-tuned model beats the baseline on a held-out speaker test |
| M6 | **TTS evaluated** | MOS score recorded |
| M7 | **System delivered** | Final API and demo work |
| M8 | **Project closed** | Report and presentation done |

**Order of work:** M1 → M2 → M3 → (M4 runs alongside) → M5 → M6 → M7 → M8

## 6. Deliverables

| Deliverable | Format |
|---|---|
| Medical speech dataset (private) | WAV + `metadata.csv` |
| Fine-tuned Whisper model | Model files on Google Drive |
| Text normalizer | `mn_text.py` |
| REST API | `08_api_server.py` |
| Demo | Gradio app |
| Evaluation results | WER/CER table, MOS score |
| Final report | PDF/DOCX |

## 7. Success criteria (KPIs)

| Metric | Target |
|---|---|
| WER on `medical_test` | Clearly lower than the baseline |
| CER on `medical_test` | Clearly lower than the baseline |
| Numbers and doses recognized correctly | Measured in the error analysis |
| TTS MOS | ≥ 3.5 out of 5 |
| Recorded data | ≥ 1.5 hours, 10 speakers, all with consent |
| API | Both endpoints respond correctly |

### Results (fill in after training)

| Model | WER `test` | CER `test` | WER `medical_test` | CER `medical_test` |
|---|---|---|---|---|
| whisper-small (baseline) | | | | |
| whisper-small fine-tuned | | | | |

## 8. Risks and mitigation

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Not enough speakers recorded | High | Medium | Use `--share` remote recording, phone recording sheets |
| Colab disconnects or runs out of GPU | Medium | High | Save checkpoints to Drive, use smaller batch, Kaggle as a backup |
| Small test set gives noisy results | Medium | High | Hold out whole speakers, report CER too |
| Common Voice access fails | Low | Low | FLEURS still works |
| Whisper hallucinates on short clips | Medium | Medium | Remove clips that are too short or silent |
| Custom TTS takes too long | Low | High | Keep it optional, use MMS as the main TTS |
| Privacy leak of recordings | High | Low | No names, `.gitignore`, never upload to public repos |

## 9. Resources

| Resource | Purpose |
|---|---|
| Google Colab T4 GPU | Training and evaluation |
| Google Drive | Storing data, checkpoints and models |
| Hugging Face account | Datasets and pretrained models |
| 10 volunteer speakers | Recordings |
| Medical professional | Reviewing sentences |
| 5–10 listeners | TTS MOS test |

## 10. Ethics and privacy

- Every speaker signs a consent form, and no names are stored.
- No real patient data is used.
- Recordings are never uploaded to public platforms.
- The system is only an aid: a person must check every transcribed dose and diagnosis.
