"""
Step 7 - Train a custom-voice VITS TTS model with Coqui TTS (character-based, Mongolian Cyrillic).

Use a SEPARATE environment:   pip install coqui-tts
Dataset: output of 06_make_tts_dataset.py (LJSpeech format; text taken from the LAST column).

  python 07_train_tts_vits.py --data data/tts --out outputs/vits-mn

Training from scratch needs ~10-20 h of clean single-speaker audio and a GPU for days.
With less data, fine-tune MMS-TTS instead (see README, section 9b).
"""
import argparse

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/tts")
ap.add_argument("--out", default="outputs/vits-mn")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.csv", path=args.data)

audio_config = VitsAudioConfig(sample_rate=22050, win_length=1024, hop_length=256,
                               num_mels=80, mel_fmin=0, mel_fmax=None)

characters = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    characters="абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя",
    punctuations=" ",
    pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_mongolian",
    batch_size=args.batch,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language=None,
    characters=characters,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=args.out,
    datasets=[dataset_config],
    save_step=2000,
    cudnn_benchmark=False,
    test_sentences=[
        "сайн байна уу",
        "өвчтөнд өдөрт хоёр удаа таван миллиграмм өгнө",
        "эмчид хандана уу",
    ],
)

ap_ = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config, eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size)

model = Vits(config, ap_, tokenizer, speaker_manager=None)
trainer = Trainer(TrainerArgs(), config, args.out, model=model,
                  train_samples=train_samples, eval_samples=eval_samples)
trainer.fit()
