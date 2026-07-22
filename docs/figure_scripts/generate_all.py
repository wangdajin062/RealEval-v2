"""
generate_all.py  --  Regenerate every QAD-MultiGuard paper figure in
paper insertion order (Figure 1 -> Figure 7).

Usage:
    python3 generate_all.py

Each script is self-contained and reads its numbers from paper_data.py
(the single audited source of truth). paper_style.py supplies the shared
SCI/IEEE-Elsevier styling.
"""
import runpy
import sys

SCRIPTS = [
    ("Figure 2", "fig2_acoustic_embedding.py"),
    ("Figure 3", "fig3_main_results.py"),
    ("Figure 4", "fig4_loss_convergence.py"),
    ("Figure 5", "fig5_loss_teacher_ablation.py"),
    ("Figure 6", "fig6_ovf_ablation.py"),
    ("Figure 7", "fig7_speculative_decoding.py"),
    ("Figure 8", "fig8_revision_ablations.py"),
]

if __name__ == "__main__":
    # Validate the data module first.
    import paper_data
    print("Running data self-checks ...")
    runpy.run_path("paper_data.py", run_name="__main__")
    print()
    for label, script in SCRIPTS:
        print(f"[{label}] {script}")
        runpy.run_path(script, run_name="__main__")
    print("\nAll 8 figures regenerated.")
