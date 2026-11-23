"""
Step 2 / 4 - Evaluate any ASR model (baseline or your fine-tuned one) with WER + CER.

Examples:
  python 02_eval_asr.py --model openai/whisper-small --data data/asr --split test
  python 02_eval_asr.py --model facebook/mms-1b-all --data data/asr --split test
  python 02_eval_asr.py --model outputs/whisper-small-mn/final --data data/asr --split medical_test
"""
import argparse

import evaluate
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Wav2Vec2ForCTC, pipeline

from mn_text import normalize_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/asr")
ap.add_argument("--split", default="test")
ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
ap.add_argument("--mms_lang", default="khk")
args = ap.parse_args()

ds = load_from_disk(args.data)[args.split]
if args.max_samples:
    ds = ds.select(range(min(args.max_samples, len(ds))))

device = 0 if torch.cuda.is_available() else -1
refs = [normalize_text(s) for s in ds["sentence"]]
hyps = []

if "mms" in args.model.lower() or "wav2vec" in args.model.lower() or "xls-r" in args.model.lower():
    processor = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model)
    if "mms" in args.model.lower():
        processor.tokenizer.set_target_lang(args.mms_lang)
        model.load_adapter(args.mms_lang)
    model.to("cuda" if device == 0 else "cpu").eval()
    for ex in ds:
        a = ex["audio"]
        inputs = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(model.device)).logits
        ids = torch.argmax(logits, dim=-1)[0]
        hyps.append(normalize_text(processor.decode(ids)))
else:
    pipe = pipeline("automatic-speech-recognition", model=args.model, device=device,
                    generate_kwargs={"language": "mongolian", "task": "transcribe"})
    audios = [{"raw": ex["audio"]["array"], "sampling_rate": ex["audio"]["sampling_rate"]} for ex in ds]
    for out in pipe(audios, batch_size=8):
        hyps.append(normalize_text(out["text"]))

wer = evaluate.load("wer").compute(references=refs, predictions=hyps)
cer = evaluate.load("cer").compute(references=refs, predictions=hyps)
print(f"\nModel: {args.model} | split: {args.split} | n={len(refs)}")
print(f"WER: {100 * wer:.2f}%   CER: {100 * cer:.2f}%")
for r, h in list(zip(refs, hyps))[:5]:
    print("\nREF:", r, "\nHYP:", h)
