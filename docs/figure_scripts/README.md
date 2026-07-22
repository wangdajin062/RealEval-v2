# QAD-MultiGuard — Paper Figure Scripts

Python scripts that regenerate all seven paper figures, in **paper insertion
order**, from a single audited data source. Every numeric value has been
cross-checked against the paper's tables and body text and the engineering
experiment runs (EXP01–EXP10).

## Files

| Order | Script | Figure |
|-------|--------|--------|
| 1 | `fig1_architecture.py`         | Three-tier edge–cloud architecture |
| 2 | `fig2_acoustic_embedding.py`   | 128-d non-invertible *F*ᵥ construction |
| 3 | `fig3_main_results.py`         | Main results on TAF-28k (F1 + recovery) |
| 4 | `fig4_loss_convergence.py`     | KL-loss convergence + SNR stability |
| 5 | `fig5_loss_teacher_ablation.py`| Loss-function + teacher-selection ablation |
| 6 | `fig6_ovf_ablation.py`         | OV-Freeze layer + step-ratio ablation |
| 7 | `fig7_speculative_decoding.py` | Speculative-decoding speedup |

Supporting modules:

- `paper_data.py` — single source of truth for every number (with self-checks).
- `paper_style.py` — shared SCI/IEEE-Elsevier matplotlib styling + palette.
- `generate_all.py` — runs all seven scripts in order.
- `FIGURE_CAPTIONS.md` — verified, ready-to-paste figure captions.

## Usage

```bash
pip install matplotlib numpy
python3 generate_all.py        # regenerate all 7 PNGs
# or run any single figure:
python3 fig3_main_results.py
```

Each script writes a 420-dpi PNG named after itself (e.g. `fig3_main_results.png`).

## Consistency guarantees

- **Figure 3 panel (b)** recovery values are *recomputed* from panel (a) F1
  values (recovery = F1 / 0.931 × 100), so the two panels — and Table 4 — can
  never drift. This is the panel that previously shipped stale 95.x values.
- `paper_data.py` asserts on import-as-main that recovery rates match F1,
  latency components sum to 268 / 342 ms, and the speedup anchors are 3.52× /
  4.25×.

## Note on deployment data

The paper frames the field pilot as a **planned 2,000-user** deployment, not a
completed one. The completed-deployment metrics that existed in the original
engineering codebase (5,000 students; precision 93.2 %, recall 98.8 %) are **not
used by any figure** and are intentionally excluded here. See the cross-check
summary at the end of `FIGURE_CAPTIONS.md`.
