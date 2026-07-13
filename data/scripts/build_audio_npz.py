#!/usr/bin/env python
"""build_audio_npz.py — Extract MFCC embeddings from ChiFraud audio -> chifraud.npz

Usage:  python data/scripts/build_audio_npz.py

Output: data/ChiFraud/chifraud.npz with keys:
  - embeddings:      MFCC-based audio embeddings (n_samples, 128)
  - labels:          fraud/normal labels from manifest dual_speaker
  - speaker_labels:  speaker IDs from manifest rank
"""
from __future__ import annotations
import csv
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("build_audio_npz")

ROOT = Path(__file__).resolve().parent.parent.parent


def extract_mfcc(wav_path: Path, n_mfcc: int = 20) -> np.ndarray:
    import librosa
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(y) == 0:
        return np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def make_embedding(mfcc: np.ndarray, target_dim: int = 128) -> np.ndarray:
    if len(mfcc) >= target_dim:
        return mfcc[:target_dim]
    repeats = target_dim // len(mfcc) + 1
    return np.tile(mfcc, repeats)[:target_dim]


def main():
    audio_dir = ROOT / "data" / "ChiFraud" / "audio"
    manifest_path = audio_dir / "manifest.csv"
    dst = ROOT / "data" / "ChiFraud" / "chifraud.npz"

    if not audio_dir.is_dir():
        logger.error("Audio directory not found: %s", audio_dir)
        return
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return

    # Read manifest
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Match manifest entries to actual WAV files
    # Actual files: "001_tts 9626.wav"  Manifest: "tts 9626.wav"
    wav_files = []
    for row in rows:
        fname = row["filename"]  # "tts 9626.wav"
        tts_num = fname.replace("tts ", "").replace(".wav", "").strip()
        rank = str(row.get("rank", "")).strip().zfill(3)
        candidate = audio_dir / f"{rank}_{fname}"
        if candidate.exists():
            wav_files.append((candidate, row))
        else:
            # Fallback: glob by tts number
            matching = list(audio_dir.glob(f"*{tts_num}*"))
            if matching:
                wav_files.append((matching[0], row))

    if not wav_files:
        logger.error("No WAV files found matching manifest entries")
        return

    logger.info("Processing %d WAV files ...", len(wav_files))

    embeddings, labels, speaker_labels = [], [], []
    for wav, row in wav_files:
        try:
            mfcc = extract_mfcc(wav)
            emb = make_embedding(mfcc, target_dim=128)
            embeddings.append(emb)

            # Label: high dual_speaker -> two people talking -> fraud call
            dual = float(row.get("dual_speaker", 50))
            labels.append(1 if dual > 95 else 0)

            spk = f"spk_{str(row.get('rank', '0')).zfill(3)}"
            speaker_labels.append(spk)
        except Exception as e:
            logger.warning("Failed %s: %s", wav.name, e)

    if not embeddings:
        logger.error("No audio files processed")
        return

    embeddings = np.stack(embeddings).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, embeddings=embeddings, labels=labels_arr,
                        speaker_labels=speaker_labels)
    logger.info("Saved %s: %d samples, dim %d, fraud ratio %.0f%%",
                dst, len(embeddings), embeddings.shape[1], 100 * labels_arr.mean())


if __name__ == "__main__":
    main()
