"""sync_audit.py - lightweight paper-data/result alignment audit.

Usage:
    python experiments/sync_audit.py --results /workspace/H100_package_realeval/outputs/results

This script is intentionally read-only. It checks whether key figure-driving values
in docs/figure_scripts/paper_data.py are backed by the current results directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _load_latest_results(results_dir: Path) -> dict[str, dict]:
    by_exp: dict[str, dict] = {}
    if not results_dir.is_dir():
        return by_exp
    for f in sorted(results_dir.glob("exp*_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            exp = data.get("experiment") or f.stem.split("_", 1)[0]
            by_exp[exp] = data
        except Exception:
            continue
    return by_exp


def _rho_pcts(exp3: dict) -> list[int]:
    rho = exp3.get("rho_sweep", {}) if isinstance(exp3, dict) else {}
    out: list[int] = []
    for k in rho.keys():
        if not k.startswith("rho_"):
            continue
        try:
            out.append(int(round(float(k.split("_", 1)[1]) * 100)))
        except Exception:
            continue
    return sorted(set(out))


def _fmt_status(tag: str, msg: str):
    print(f"  [{tag:<10}] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit paper_data/result alignment")
    ap.add_argument("--results", type=str, required=True, help="Path to outputs/results directory")
    args = ap.parse_args()

    results_dir = Path(args.results).resolve()
    if not results_dir.is_dir():
        print(f"[ERROR] results directory not found: {results_dir}")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    figure_scripts = repo_root / "docs" / "figure_scripts"
    sys.path.insert(0, str(figure_scripts))
    os.environ["PAPER_DATA_USE_LIVE"] = "1"

    try:
        import paper_data
    except Exception as e:
        print(f"[ERROR] failed to import paper_data: {e}")
        return 2

    results = _load_latest_results(results_dir)
    exp3 = results.get("exp3", {})

    canonical = [0, 10, 20, 30, 40, 50]
    live_points = paper_data.EXP10_OVF_STEP_RATIO
    live_pcts = sorted({int(p.get("ratio_pct")) for p in live_points if isinstance(p, dict) and "ratio_pct" in p})
    measured_pcts = _rho_pcts(exp3)

    measured_cnt = 0
    fallback_cnt = 0
    mismatch_cnt = 0

    print("\n" + "=" * 78)
    print("EXP10_OVF_STEP_RATIO audit")
    print("=" * 78)

    for pct in canonical:
        if pct in live_pcts:
            if pct in measured_pcts:
                measured_cnt += 1
                _fmt_status("MEASURED", f"EXP10_OVF_STEP_RATIO[{pct}%] backed by exp3 rho_sweep")
            else:
                fallback_cnt += 1
                _fmt_status("FALLBACK", f"EXP10_OVF_STEP_RATIO[{pct}%] present but not in exp3 rho_sweep")
        else:
            mismatch_cnt += 1
            _fmt_status("MISMATCH", f"EXP10_OVF_STEP_RATIO[{pct}%] key absent in paper_data live mapping")

    unused = [p for p in measured_pcts if p not in live_pcts]
    if unused:
        mismatch_cnt += 1
        _fmt_status("MISMATCH", f"EXP10_OVF_STEP_RATIO unused measurements: exp3 measured {unused} but paper_data never reads them")

    print("\n" + "=" * 78)
    print(f"  MEASURED   {measured_cnt}")
    print(f"  FALLBACK   {fallback_cnt}")
    print(f"  MISMATCH   {mismatch_cnt}")
    print(f"  results    {len(results)} experiment json(s) loaded from {results_dir}")

    src_report = getattr(paper_data, "paper_data_source_report", None)
    if callable(src_report):
        print("\nsource report:")
        for line in src_report():
            print(f"  - {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
