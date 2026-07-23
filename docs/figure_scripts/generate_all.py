"""
generate_all.py  --  Regenerate every QAD-MultiGuard paper figure in
paper insertion order (Figure 1 -> Figure 8).

Usage:
    python3 generate_all.py

Each script is self-contained and reads its numbers from paper_data.py
(the single audited source of truth, now auto-loaded from experiment results).
paper_style.py supplies the shared SCI/IEEE-Elsevier styling.
"""
import os
import runpy
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── Clear old figures ────────────────────────────────────────────────────
_FIGURE_DIR = _THIS_DIR.parent / "figure"
_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
for _f in _FIGURE_DIR.glob("*.png"):
    _f.unlink()
for _f in _FIGURE_DIR.glob("*.pdf"):
    _f.unlink()
for _f in _FIGURE_DIR.glob("*.tiff"):
    _f.unlink()
print("Cleared old figures.\n")

# ── Figure scripts in paper insertion order ──────────────────────────────
# Each entry: (Figure label, script, required experiments)
SCRIPTS = [
    ("Figure 1", "fig1_architecture.py",         []),                        # diagram
    ("Figure 2", "fig2_acoustic_embedding.py",   []),                        # diagram
    ("Figure 3", "fig3_main_results.py",         ["exp11"]),                 # needs exp11 for QAT row
    ("Figure 4", "fig4_loss_convergence.py",     ["exp1"]),                  # needs exp1 trajectory
    ("Figure 5", "fig5_loss_teacher_ablation.py", ["exp2"]),                 # exp10 optional (paper_data fallback)
    ("Figure 6", "fig6_ovf_ablation.py",         ["exp3"]),                  # needs exp3
    ("Figure 7", "fig7_speculative_decoding.py",  ["exp6"]),                 # needs exp6
    ("Figure 8", "fig8_revision_ablations.py",    ["exp5", "exp7"]),        # needs exp5/exp7
]

# Check which experiment data exists
_RESULTS = _THIS_DIR.parent.parent / "outputs" / "results"
_available = set()
if _RESULTS.is_dir():
    for f in _RESULTS.glob("exp*_*.json"):
        try:
            import json
            r = json.loads(f.read_text(encoding="utf-8"))
            _available.add(r.get("experiment", f.stem.split("_")[0]))
        except Exception:
            pass

if __name__ == "__main__":
    # Validate paper_data first from the figure-scripts directory, regardless of cwd.
    import paper_data
    if hasattr(paper_data, "paper_data_source_report"):
        print("Data source audit:")
        for line in paper_data.paper_data_source_report():
            print(f"  - {line}")
        print()
    print("Running data self-checks ...")
    runpy.run_path(str(_THIS_DIR / "paper_data.py"), run_name="__main__")
    print()

    for label, script, required in SCRIPTS:
        missing = [e for e in required if e not in _available]
        if missing and required:
            print(f"[SKIP] {label} ({script}) — missing experiment data: {missing}")
            continue
        print(f"[{label}] {script}")
        try:
            runpy.run_path(str(_THIS_DIR / script), run_name="__main__")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # List generated figures
    print("\nGenerated figures:")
    for f in sorted(_FIGURE_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name:45s} {size/1024:7.1f} KB")
    print(f"\nDone. Figures in {_FIGURE_DIR}")
