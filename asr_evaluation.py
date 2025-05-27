# This code was run in Google Colab(Python 3.10 or else Coqui does not work), using
# Dependencies installation: pip install git+https://github.com/openai/whisper.git datasets torchaudio transformers soundfile evaluate tqdm asteroid stt==1.4.0
# actual running was implemented using %run asr_evaluation.py [flags]

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torchaudio, hashlib
import soundfile as sf
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import asteroid


# Dataset loading
def load_corpus(name: str, split: str):
    if name == "ami":
        ds = load_dataset("ami", "ihm", split=split)
        return ds, "transcript"
    elif name == "librispeech":
        ds = load_dataset("librispeech_asr", "clean", split=split)
        return ds, "text"


# Whisper setup
def load_whisper(model_size: str = "medium"):
    import whisper
    return whisper.load_model(model_size)


def transcribe_whisper(model, audio_path: str) -> str:
    return model.transcribe(audio_path, fp16=False)["text"].strip().lower()


# Wav2vec 2.0 setup
def load_wav2vec(model_id: str = "facebook/wav2vec2-large-960h"):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return processor, model


def transcribe_wav2vec(proc_model: Tuple, audio_path: str) -> str:
    processor, model = proc_model
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].lower().strip()


# Coqui STT setup
def load_stt(model_path: str, scorer_path: str | None = None):
    from stt import Model
    stt_model = Model(model_path)
    if scorer_path and Path(scorer_path).is_file():
        stt_model.enableExternalScorer(scorer_path)
    return stt_model


def transcribe_stt(model, audio_path: str) -> str:
    audio, sr = sf.read(audio_path, always_2d=False)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(audio).unsqueeze(0)).squeeze(0).numpy()
    audio16 = (audio * 32767).astype(np.int16)
    return model.stt(audio16).lower().strip()


# Speaker Separation Algorithm
SEP_MODELS: Dict[str, torch.nn.Module] = {}
def get_separator(kind: str):
        if kind == "none":
            return None
        if kind in SEP_MODELS:
            return SEP_MODELS[kind]
        if kind == "conv-tasnet":
            model = asteroid.models.ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean")
        elif kind == "dprnn":
            model = asteroid.models.DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet_WHAM_sepclean")
        elif kind == "dccrn":
            model = asteroid.models.DCCRNet.from_pretrained("mpariente/DCCRNet_Libri2Mix_enhsingle_16k")

        model.eval()
        SEP_MODELS[kind] = model
        return model


def maybe_separate(path: str, sep_kind: str, cache_dir: str) -> str:
        if sep_kind == "none":
            return path

        # Cache setup
        h = hashlib.sha1((sep_kind + path).encode()).hexdigest()
        out_path = os.path.join(cache_dir, f"{h}.wav")
        if os.path.exists(out_path):
            return out_path

        os.makedirs(cache_dir, exist_ok=True)
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        model = get_separator(sep_kind)
        with torch.no_grad():
            est = model.separate(wav)            
        # pick the loudest speaker
        best = est.pow(2).mean(-1).argmax()
        torchaudio.save(out_path, est[best].unsqueeze(0), 16000)
        return out_path


# Accuracy evaluation
def path_from_sample(sample):
    if "file" in sample:
        return sample["file"]                 # for LibriSpeech dataset
    if "audio" in sample and "path" in sample["audio"]:
        return sample["audio"]["path"]        # for AMI dataset


def evaluate_model(transcribe_fn: Callable[[str], str], dataset, ref_key: str, sep_kind: str, cache) -> float:
    wer_metric = evaluate.load("wer")
    preds, refs = [], []
    for sample in tqdm(dataset, desc="Evaluating", unit="utt"):
        clean_path = maybe_separate(path_from_sample(sample), sep_kind, cache)
        preds.append(transcribe_fn(clean_path))
        refs.append(sample[ref_key].lower())
    return wer_metric.compute(predictions=preds, references=refs)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["whisper"], choices=["whisper", "wav2vec2", "stt"])
    parser.add_argument("--whisper_size", default="base")
    # stuff for Coqui
    parser.add_argument("--stt_model", default="stt-0.9.3-models.pbmm")
    parser.add_argument("--stt_scorer", default="stt-0.9.3-models.scorer")
    # separation algorithms handling
    parser.add_argument("--separation", choices=["none", "conv-tasnet", "dprnn", "dccrn"], default="none")
    parser.add_argument("--separate_cache", default="separated_cache")
    # dataset handling
    parser.add_argument("--corpus", choices=["librispeech", "ami"], default="librispeech")
    parser.add_argument("--dataset_split", default="test[:1%]")
    parser.add_argument("--max_utts", type=int, default=None)

    args, _ = parser.parse_known_args(argv)

    
    print(f"Loading {args.corpus.upper()} subset ({args.dataset_split})")
    ds, ref_key = load_corpus(args.corpus, args.dataset_split)
    if args.max_utts:
        ds = ds.shuffle(seed=0).select(range(args.max_utts))
        print(f"Using first {args.max_utts} shuffled utterances")

    results: Dict[str, float] = {}
    
    if "whisper" in args.models:
        w_model = load_whisper(args.whisper_size)
        results[f"whisper-{args.whisper_size}"] = evaluate_model(lambda p: transcribe_whisper(w_model, p), ds, ref_key, args.separation, args.separate_cache)

    if "wav2vec2" in args.models:
        w2v_proc_model = load_wav2vec()
        results["wav2vec2-large-960h"] = evaluate_model(lambda p: transcribe_wav2vec(w2v_proc_model, p), ds, ref_key, args.separation, args.separate_cache)

    if "stt" in args.models:
        stt_model = load_stt(args.stt_model, args.stt_scorer)
        results["coqui-stt"] = evaluate_model(lambda p: transcribe_stt(stt_model, p), ds, ref_key, args.separation, args.separate_cache)

    for name, wer in results.items():
        print(f"{name:20} {wer:.3f}")


if __name__ == "__main__":
    main()
