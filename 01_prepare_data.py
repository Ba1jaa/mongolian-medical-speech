"""
Step 1 - Build the ASR dataset (Common Voice + FLEURS + your own recordings).

Usage:
  huggingface-cli login            # needed for Common Voice (accept its terms on the HF page first)
  python 01_prepare_data.py --out data/asr
  python 01_prepare_data.py --out data/asr --custom_csv my_data/metadata.csv --custom_dir my_data/wavs

custom CSV format (UTF-8, header required):
  path,text
  rec_0001.wav,Өвчтөнд өдөрт хоёр удаа таван миллиграмм өгнө
"""
import argparse
import os

import pandas as pd
from datasets import (Audio, Dataset, DatasetDict, concatenate_datasets,
                      load_dataset)

from mn_text import normalize_text

SR = 16000
MIN_SEC, MAX_SEC = 1.0, 30.0


def to_common(ds, text_col):
    ds = ds.select_columns(["audio", text_col])
    if text_col != "sentence":
        ds = ds.rename_column(text_col, "sentence")
    return ds.cast_column("audio", Audio(sampling_rate=SR))


def load_fleurs():
    out = {}
    try:
        for split in ["train", "validation", "test"]:
            ds = load_dataset("google/fleurs", "mn_mn", split=split, trust_remote_code=True)
            out[split] = to_common(ds, "transcription")
        print("FLEURS loaded")
    except Exception as e:
        print("!! FLEURS skipped:", e)
    return out


def load_cv():
    out = {}
    try:
        for split in ["train", "validation", "test"]:
            ds = load_dataset("mozilla-foundation/common_voice_17_0", "mn", split=split,
                              trust_remote_code=True, token=True)
            out[split] = to_common(ds, "sentence")
        print("Common Voice loaded")
    except Exception as e:
        print("!! Common Voice skipped (log in + accept terms?):", e)
    return out


def load_custom(csv_path, wav_dir):
    df = pd.read_csv(csv_path)
    paths = [os.path.join(wav_dir, p) for p in df["path"]]
    ds = Dataset.from_dict({"audio": paths, "sentence": df["text"].astype(str).tolist()})
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = ds.shuffle(seed=42)
    n_test = max(1, int(0.1 * len(ds)))
    return ds.select(range(n_test, len(ds))), ds.select(range(n_test))  # train, medical_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/asr")
    ap.add_argument("--custom_csv")
    ap.add_argument("--custom_dir", default=".")
    args = ap.parse_args()

    sources = [load_fleurs(), load_cv()]
    splits = {}
    for split in ["train", "validation", "test"]:
        parts = [s[split] for s in sources if split in s]
        if parts:
            splits[split] = concatenate_datasets(parts)

    if args.custom_csv:
        tr, med = load_custom(args.custom_csv, args.custom_dir)
        splits["train"] = concatenate_datasets([splits["train"], tr]) if "train" in splits else tr
        splits["medical_test"] = med

    assert "train" in splits, "No training data could be loaded!"
    dd = DatasetDict(splits)

    def norm(batch):
        batch["sentence"] = normalize_text(batch["sentence"])
        return batch

    def ok(ex):
        dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
        return MIN_SEC <= dur <= MAX_SEC and len(ex["sentence"]) > 0

    dd = dd.map(norm).filter(ok)
    for k, v in dd.items():
        print(k, len(v), "utterances")
    dd.save_to_disk(args.out)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
