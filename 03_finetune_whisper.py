"""
Step 3 - Fine-tune Whisper for Mongolian ASR.

Free Colab/Kaggle T4 (16 GB):   --model openai/whisper-small  (defaults below work)
Bigger GPU (24 GB+):            --model openai/whisper-medium --batch 4 --accum 4

  python 03_finetune_whisper.py --data data/asr --out outputs/whisper-small-mn
"""
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import torch
from datasets import load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor, WhisperForConditionalGeneration,
                          WhisperProcessor, WhisperTokenizer)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/whisper-small")
ap.add_argument("--data", default="data/asr")
ap.add_argument("--out", default="outputs/whisper-small-mn")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
args = ap.parse_args()

fe = WhisperFeatureExtractor.from_pretrained(args.model)
tok = WhisperTokenizer.from_pretrained(args.model, language="mongolian", task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model, language="mongolian", task="transcribe")

ds = load_from_disk(args.data)


def prepare(batch):
    a = batch["audio"]
    batch["input_features"] = fe(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
    batch["labels"] = tok(batch["sentence"]).input_ids
    return batch


ds = ds.map(prepare, remove_columns=["audio", "sentence"])


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]):
        inp = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
        lab = [{"input_ids": f["labels"]} for f in features]
        lab_batch = self.processor.tokenizer.pad(lab, return_tensors="pt")
        labels = lab_batch["input_ids"].masked_fill(lab_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = tok.pad_token_id
    pred_str = tok.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tok.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str)}


model = WhisperForConditionalGeneration.from_pretrained(args.model)
model.generation_config.language = "mongolian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False  # needed with gradient checkpointing

targs = Seq2SeqTrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    warmup_steps=500,
    max_steps=args.steps,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=targs,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=Collator(processor, model.config.decoder_start_token_id),
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()  # to resume after a crash: trainer.train(resume_from_checkpoint=True)
trainer.save_model(f"{args.out}/final")
processor.save_pretrained(f"{args.out}/final")
print("Saved fine-tuned model to", f"{args.out}/final")
