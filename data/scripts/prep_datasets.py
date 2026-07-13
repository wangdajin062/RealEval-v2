#!/usr/bin/env python
"""prep_datasets.py — Convert local data files to JSONL format expected by data.py loaders.

Run from project root:  python data/scripts/prep_datasets.py

Creates:
  data/AdvFraud3k/advfraud3k.jsonl   <- from advfraud_3k_compiled.json
  data/TAF28k/taf28k.jsonl           <- already exists, verifies
  data/ChiFraud/chifraud.jsonl       <- from CSV if available, else placeholder
"""
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("prep_datasets")

ROOT = Path(__file__).resolve().parent.parent.parent


def prep_taf28k() -> bool:
    path = ROOT / "data" / "TAF28k" / "taf28k.jsonl"
    if not path.exists():
        logger.warning("TAF-28k JSONL not found at %s", path)
        return False
    with open(path, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    logger.info("TAF-28k: %d samples at %s", count, path)
    return True


def prep_advfraud3k() -> bool:
    src = ROOT / "data" / "AdvFraud3k" / "advfraud_3k_compiled.json"
    dst = ROOT / "data" / "AdvFraud3k" / "advfraud3k.jsonl"
    if not src.exists():
        logger.warning("AdvFraud-3k compiled JSON not found at %s", src)
        return False
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    count = 0
    with open(dst, "w", encoding="utf-8") as out:
        for s in samples:
            text = s.get("adversarial_text") or s.get("original_concept") or ""
            label = 1 if s.get("label", "").lower() in ("fraud", "1", "true") else 0
            out.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            count += 1
    logger.info("AdvFraud-3k: %d samples written to %s", count, dst)
    return True


def prep_chifraud() -> bool:
    dst = ROOT / "data" / "ChiFraud" / "chifraud.jsonl"
    csv_dir = ROOT / "data" / "ChiFraud" / "dataset"
    csv_files = sorted(csv_dir.glob("*.csv"))
    valid_csvs = [f for f in csv_files if f.stat().st_size > 0]

    if valid_csvs:
        count = 0
        with open(dst, "w", encoding="utf-8") as out:
            for csv_path in valid_csvs:
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("text", row.get("content", row.get("sentence", "")))
                        raw_label = row.get("label", row.get("is_fraud", "0"))
                        label = 1 if str(raw_label).strip().lower() in ("fraud", "1", "true", "yes") else 0
                        if text:
                            out.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
                            count += 1
        logger.info("ChiFraud: %d samples written from CSV to %s", count, dst)
        return True

    logger.warning("ChiFraud CSV files empty (0 bytes); no chifraud.jsonl generated")
    return False


def main():
    ok = all([prep_taf28k(), prep_advfraud3k()])
    prep_chifraud()
    if ok:
        print("\n[OK] Data files ready. Upload to RunPod:")
        print("  scp -P 14699 data/TAF28k/taf28k.jsonl root@216.243.220.229:/workspace/data/TAF28k/")
        print("  scp -P 14699 data/AdvFraud3k/advfraud3k.jsonl root@216.243.220.229:/workspace/data/AdvFraud3k/")
    else:
        print("\n[WARN] Some datasets missing. Check logs above.")


if __name__ == "__main__":
    main()
