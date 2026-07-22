"""sync_paper_data.py — Update paper_data.py from live experiment results.

Reads the latest experiment results (outputs/results/*.json or metrics.json)
and writes the measured values back into paper_data.py.  Only numeric literals
are touched — comments, variable names, structure and the self-check block are
preserved.

Usage:
    python3 sync_paper_data.py           # update paper_data.py in-place
    python3 sync_paper_data.py --dry-run # print what would change, don't write
    python3 sync_paper_data.py --regenerate  # update + run generate_all.py

The figure scripts themselves are NEVER modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DATA = SCRIPTS_DIR / "paper_data.py"
RESULTS_DIR = ROOT / "outputs" / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Read live experiment results
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_results() -> dict[str, dict]:
    """Collect all experiment results, keyed by experiment name (exp1, exp2, …)."""
    by_exp: dict[str, dict] = {}
    if not RESULTS_DIR.is_dir():
        print(f"[WARN] results dir not found: {RESULTS_DIR}")
        return by_exp

    # Prefer metrics.json (pipeline output) if available
    metrics_file = RESULTS_DIR / "metrics.json"
    if metrics_file.exists():
        metrics = _load_json(metrics_file)
        for group, exps in metrics.get("groups", {}).items():
            for short, val in exps.items():
                # metrics.json stores extracted scalars; we need the full result.
                pass

    # Load individual experiment result files
    for f in sorted(RESULTS_DIR.glob("exp*_*.json")):
        try:
            r = _load_json(f)
            exp_name = r.get("experiment", f.stem.split("_")[0])
            by_exp[exp_name] = r
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skipping {f.name}: {e}")

    # Also check for the pipeline's consolidated output
    all_file = RESULTS_DIR / "all_experiments.json"
    if all_file.exists():
        try:
            all_r = _load_json(all_file)
            for k, v in all_r.items():
                if k not in by_exp:
                    by_exp[k] = v
        except Exception as e:
            print(f"[WARN] all_experiments.json: {e}")

    return by_exp


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Map experiment results → paper_data values
# ═══════════════════════════════════════════════════════════════════════════════

def _r(v, decimals=4):
    """Round float, return as-is for non-floats."""
    if isinstance(v, float):
        return round(v, decimals)
    return v


def _nonzero(v) -> bool:
    """Return True if value is meaningfully non-zero (not 0, None, or empty)."""
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return abs(v) > 1e-9
    return bool(v)


def compute_updates(results: dict[str, dict]) -> dict[str, object]:
    """Return a dict of {VARIABLE_NAME: new_value} from experiment results.

    Guardrails:
      - BF16_F1 is the PAPER'S BF16 ceiling (not exp1 F1) — never overwrite.
      - Zero/None/empty values are skipped (experiment didn't measure it).
      - PPL values use different metrics; not auto-mapped.
    """
    updates: dict[str, object] = {}

    # ── exp1 F1 (informational, does NOT overwrite BF16_F1) ──────────────
    exp1 = results.get("exp1", {})
    # Store exp1 F1 separately — BF16_F1 is the paper's BF16 ceiling.
    if "f1" in exp1 and _nonzero(exp1["f1"]):
        updates["_EXP1_F1"] = _r(exp1["f1"])

    # ── exp3 → OV-Freeze ablation ────────────────────────────────────────
    exp3 = results.get("exp3", {})
    conditions = exp3.get("conditions", {})
    if conditions:
        # EXP04_OVF_LAYER_ABLATION — map conditions to layer config names
        no_reg = conditions.get("no_reg", {})
        full = conditions.get("ov_freeze_full", {})
        half = conditions.get("ov_freeze_half", {})
        quarter = conditions.get("ov_freeze_quarter", {})

        # The paper_data format maps config names to (f1, drift_pct).
        # We update drift values while preserving the original config labels.
        # New mapping: conditions → EXP04 entries
        updates["_EXP04_DRIFT"] = {
            "no OVF": _r(no_reg.get("variance_drift_pct", 0), 1),
            "q,k,v,o\n(ours)": _r(full.get("variance_drift_pct", 0), 1),
            "q,v": _r(half.get("variance_drift_pct", 0), 1),
            "FFN": _r(quarter.get("variance_drift_pct", 0), 1),
        }
        updates["_EXP04_F1"] = _r(full.get("f1", 0))

    # ── exp3 rho_sweep — drift values only (F1/PPL use different semantics) ──
    # Note: exp3 measures post-hoc rescaling, so F1 is constant across rho.
    # The paper's Fig 6(b) F1/PPL curves come from actual training, not from
    # exp3.  Only variance_drift_pct is mapped here.

    # ── exp2 → EXP03_LOSS_ABLATION (F1 + kl_final) ───────────────────────
    exp2 = results.get("exp2", {})
    variants = exp2.get("variants", {})
    if variants:
        loss_map = {
            "kl_only": "Pure KL\n(ours)",
            "mse_only": "MSE",
            "kl_mse_combined": "3-term\nhybrid",
        }
        # Check if F1 values actually differ across variants before updating
        f1_vals = {vk: vd.get("f1", 0) for vk, vd in variants.items()}
        if len(set(_r(v) for v in f1_vals.values() if _nonzero(v))) > 1:
            for vk, label in loss_map.items():
                vd = variants.get(vk, {})
                if _nonzero(vd.get("f1")):
                    updates.setdefault("_EXP03_LOSS_F1", {})[label] = _r(vd["f1"])
                if _nonzero(vd.get("kl_final")):
                    updates.setdefault("_EXP03_LOSS_KL", {})[label] = _r(vd["kl_final"], 4)
        else:
            # All F1 identical — only update KL values
            for vk, label in loss_map.items():
                vd = variants.get(vk, {})
                if _nonzero(vd.get("kl_final")):
                    updates.setdefault("_EXP03_LOSS_KL", {})[label] = _r(vd["kl_final"], 4)

    # ── exp6 → SPEC_ALPHA_GENERIC ────────────────────────────────────────
    exp6 = results.get("exp6", {})
    diag = exp6.get("diagnostic_B", {})
    hm = diag.get("h100_measured", {})
    if hm.get("generic") is not None and _nonzero(hm["generic"]):
        updates["SPEC_ALPHA_GENERIC"] = _r(hm["generic"])
    if hm.get("domain") is not None and _nonzero(hm["domain"]):
        updates["SPEC_ALPHA_TUNED"] = _r(hm["domain"])

    # ── exp8 → latencies ─────────────────────────────────────────────────
    exp8 = results.get("exp8", {})
    latencies = exp8.get("latencies", {})
    if latencies:
        # latency components: keep existing decomposition, update totals if available
        updates["_LATENCY_P50"] = {k: _r(v) for k, v in latencies.items()}
        # P99 may not be directly measured; keep paper_data defaults

    # ── exp5 → cross-dataset F1 ──────────────────────────────────────────
    exp5 = results.get("exp5", {})
    for ds_key in ("taf28k", "chifraud", "advfraud"):
        ds_val = exp5.get(ds_key, {})
        if isinstance(ds_val, dict) and "f1" in ds_val:
            updates[f"_EXP05_{ds_key.upper()}_F1"] = _r(ds_val["f1"])
    adv = exp5.get("advfraud", {})
    if isinstance(adv, dict):
        pool = adv.get("full_pool", {})
        if isinstance(pool, dict) and "f1" in pool:
            updates["_EXP05_ADVFRAUD_F1"] = _r(pool["f1"])

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Apply updates to paper_data.py
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_py_literal(src: str) -> object:
    """Safely parse a Python literal (int, float, string, list, dict)."""
    try:
        return ast.literal_eval(src)
    except (SyntaxError, ValueError):
        return None


def apply_updates(updates: dict[str, object], dry_run: bool = False) -> list[str]:
    """Write updated values into paper_data.py. Returns list of change descriptions."""
    src = PAPER_DATA.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    changes: list[str] = []

    # ── Strategy: replace top-level variable assignments ──────────────────
    # For simple scalar variables (BF16_F1, SPEC_ALPHA_GENERIC, etc.):
    scalar_re = re.compile(
        r"^(\s*)(BF16_F1_ERR|SPEC_ALPHA_GENERIC|SPEC_ALPHA_TUNED|"
        r"SAFE_QAQ_F1|SAFE_QAQ_F1_ERR|NVFP4_SIZE_MB|Q4_K_M_SIZE_MB|"
        r"LOSS_PLATEAU|LOSS_CONVERGED|OVF_ACTIVATION_STEP|TOTAL_STEPS)"
        r"(\s*=\s*)([\d.]+|None)(.*)$"
    )

    for i, line in enumerate(lines):
        m = scalar_re.match(line)
        if m:
            var_name = m.group(2)
            if var_name in updates:
                new_val = updates[var_name]
                old_val_str = m.group(4)
                new_val_str = str(new_val) if not isinstance(new_val, float) else f"{new_val:.4f}".rstrip("0").rstrip(".")
                if new_val_str != old_val_str:
                    new_line = f"{m.group(1)}{var_name}{m.group(3)}{new_val_str}{m.group(5)}"
                    changes.append(f"  {var_name}: {old_val_str} → {new_val_str}")
                    if not dry_run:
                        lines[i] = new_line

    # ── For structured data (lists of dicts), replace blocks ──────────────
    # EXP04_OVF_LAYER_ABLATION — update drift values
    if "_EXP04_DRIFT" in updates:
        drift_updates = updates["_EXP04_DRIFT"]
        changes.extend(_update_list_of_dicts(
            lines, "EXP04_OVF_LAYER_ABLATION", "drift_pct", drift_updates,
            key_field="config", dry_run=dry_run))

    if "_EXP04_F1" in updates:
        changes.extend(_update_list_of_dicts(
            lines, "EXP04_OVF_LAYER_ABLATION", "f1",
            {k: updates["_EXP04_F1"] for k in drift_updates},
            key_field="config", dry_run=dry_run))

    # EXP03_LOSS_ABLATION — update F1 and/or KL values
    if "_EXP03_LOSS_F1" in updates:
        for label, new_f1 in updates["_EXP03_LOSS_F1"].items():
            changes.extend(_update_list_of_dicts(
                lines, "EXP03_LOSS_ABLATION", "f1",
                {label: new_f1}, key_field="loss", dry_run=dry_run))
    if "_EXP03_LOSS_KL" in updates:
        for label, new_kl in updates["_EXP03_LOSS_KL"].items():
            changes.extend(_update_list_of_dicts(
                lines, "EXP03_LOSS_ABLATION", "kl",
                {label: new_kl}, key_field="loss", dry_run=dry_run))

    # EXP05_SPECULATIVE — update alpha-based blocks
    if "SPEC_ALPHA_GENERIC" in updates:
        changes.append(f"  SPEC_ALPHA_GENERIC: → {updates['SPEC_ALPHA_GENERIC']}")
    if "SPEC_ALPHA_TUNED" in updates:
        changes.append(f"  SPEC_ALPHA_TUNED: → {updates['SPEC_ALPHA_TUNED']}")

    if not dry_run and changes:
        PAPER_DATA.write_text("".join(lines), encoding="utf-8")

    return changes


def _update_list_of_dicts(lines: list[str], var_name: str, field: str,
                          mapping: dict[str, float], key_field: str = "config",
                          key_as_int: bool = False,
                          dry_run: bool = False) -> list[str]:
    """Update a specific field within a list-of-dicts variable assignment."""
    changes: list[str] = []
    in_block = False
    brace_depth = 0
    block_start = -1

    for i, line in enumerate(lines):
        if not in_block:
            if f"{var_name} =" in line or f"{var_name}=" in line:
                in_block = True
                block_start = i
                brace_depth = line.count("[") - line.count("]")
        else:
            brace_depth += line.count("[") - line.count("]")
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                # Process the block from block_start to i
                changes.extend(_patch_block(
                    lines, block_start, i, field, mapping, key_field,
                    key_as_int, dry_run))
                in_block = False
                break
    return changes


def _patch_block(lines: list[str], start: int, end: int, field: str,
                 mapping: dict[str, float], key_field: str,
                 key_as_int: bool, dry_run: bool) -> list[str]:
    """Patch numeric values within a list-of-dicts block."""
    changes: list[str] = []
    # Find the key-value pairs within dicts
    block_text = "".join(lines[start:end + 1])

    for map_key, new_val in mapping.items():
        # Build search pattern: find the dict entry with matching key_field
        if key_as_int:
            key_str = str(int(map_key))
        else:
            key_str = map_key

        # Pattern: "key_field": "key_str" or "key_field": key_str
        quoted_key = f'"{key_str}"' if '"' not in key_str else f"'{key_str}'"
        key_pat = re.compile(
            rf'("{re.escape(key_field)}"\s*:\s*{re.escape(quoted_key)})'
        )
        # Also try the non-quoted version (for int keys)
        if not key_pat.search(block_text):
            key_pat = re.compile(
                rf'("{re.escape(key_field)}"\s*:\s*{re.escape(key_str)})'
            )
        km = key_pat.search(block_text)
        if not km:
            continue

        # Find the field value within this dict entry
        # Look for "field": number within a reasonable range after the key match
        pos = km.start()
        remaining = block_text[pos:pos + 300]  # search within 300 chars
        field_pat = re.compile(rf'("{re.escape(field)}"\s*:\s*)([\d.]+)')
        fm = field_pat.search(remaining)
        if not fm:
            continue

        old_val = fm.group(2)
        new_val_str = f"{new_val:.4f}".rstrip("0").rstrip(".")
        if old_val != new_val_str:
            # Find absolute position in lines
            abs_pos = pos + fm.start()
            # Find which line and column
            cum = 0
            for li in range(start, end + 1):
                line_len = len(lines[li])
                if cum + line_len > abs_pos:
                    col = abs_pos - cum
                    old_line = lines[li]
                    new_line = old_line[:col] + old_line[col:].replace(
                        fm.group(0), fm.group(1) + new_val_str, 1)
                    if not dry_run:
                        lines[li] = new_line
                    changes.append(
                        f"  {var_name_from_block(lines, start)}[{key_str}].{field}: "
                        f"{old_val} → {new_val_str}")
                    break
                cum += line_len
    return changes


def var_name_from_block(lines: list[str], start: int) -> str:
    """Extract variable name from assignment line."""
    for i in range(start, -1, -1):
        m = re.match(r'^(\w+)\s*=', lines[i])
        if m:
            return m.group(1)
    line = lines[start].strip()
    m = re.match(r'^(\w+)\s*=', line) if '=' in line else None
    return m.group(1) if m else "?"


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Sync paper_data.py from experiment results")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print changes without writing")
    ap.add_argument("--regenerate", action="store_true",
                    help="Run generate_all.py after sync")
    args = ap.parse_args()

    os.chdir(str(SCRIPTS_DIR))

    # Backup
    if not args.dry_run:
        import shutil
        bak = PAPER_DATA.with_suffix(".py.bak")
        shutil.copy2(PAPER_DATA, bak)
        print(f"[backup] {bak}")

    # Collect results
    print("[1/3] Reading experiment results ...")
    results = _collect_results()
    if not results:
        print("[FAIL] No experiment results found in outputs/results/")
        print("       Run the paper pipeline first: python -m experiments.paper_pipeline --paper")
        sys.exit(1)
    print(f"       Found {len(results)} experiment(s): {sorted(results.keys())}")

    # Compute updates
    print("[2/3] Mapping results → paper_data ...")
    updates = compute_updates(results)
    if not updates:
        print("       No values to update — paper_data is already aligned.")
        return

    for k, v in updates.items():
        if isinstance(v, dict):
            print(f"       {k}: {len(v)} entries")
        elif isinstance(v, list):
            print(f"       {k}: {len(v)} entries")
        else:
            print(f"       {k}: {v}")

    # Apply
    changes = apply_updates(updates, dry_run=args.dry_run)
    if changes:
        print(f"       {len(changes)} value(s) changed:")
        for c in changes:
            print(c)
    else:
        print("       No changes needed (values already match).")

    if args.dry_run:
        print("\n[DRY-RUN] paper_data.py was NOT modified.")
        return

    print("[3/3] Running self-checks ...")
    # Validate updated paper_data
    paper_data_mod = {}
    exec(PAPER_DATA.read_text(encoding="utf-8"), paper_data_mod)
    # Run the __main__ self-check
    if "__main__" in str(PAPER_DATA.read_text(encoding="utf-8")):
        import runpy
        try:
            runpy.run_path(str(PAPER_DATA), run_name="__main__")
            print("       Self-checks PASSED.")
        except AssertionError as e:
            print(f"       [WARN] Self-check assertion: {e}")
            print("       Review paper_data.py manually before regenerating figures.")
        except Exception as e:
            print(f"       [WARN] Self-check error: {e}")

    if args.regenerate:
        print("\n[+] Regenerating all figures ...")
        import runpy
        runpy.run_path(str(SCRIPTS_DIR / "generate_all.py"), run_name="__main__")
        print("       Done.")


if __name__ == "__main__":
    main()
