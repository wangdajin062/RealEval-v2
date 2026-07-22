"""
fig5_loss_teacher_ablation.py  --  Paper Figure 5 (insertion order #5)

Loss-function and teacher-selection ablation.
  (a) F1 (blue bars, left axis) and KL divergence to the BF16 teacher
      (red bars, right axis) across five loss variants. Pure KL gives the best
      trade-off (F1 = 0.916, KL = 0.005); cross-entropy (= QAT) suffers severe
      distribution drift (KL = 0.311).
  (b) Teacher selection: homologous 0.5B BF16 teacher vs larger heterogeneous
      teachers (1.8B-7B), under a fixed 0.5B-token budget vs training to
      convergence. The homologous teacher reaches the best F1 with the fewest
      tokens, validating the self-distillation design.

All numbers from paper_data (EXP03 / EXP09).

Run:  python3 fig5_loss_teacher_ablation.py
Out:  fig5_loss_teacher_ablation.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps
from paper_data import EXP03_LOSS_ABLATION, EXP09_TEACHER
import os

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.5),
                               gridspec_kw={"wspace": 0.47})

# --- (a) loss-function ablation: F1 bars + KL bars (twin axis) -------------
labels = [d["loss"] for d in EXP03_LOSS_ABLATION]
f1 = [d["f1"] for d in EXP03_LOSS_ABLATION]
kl = [d["kl"] for d in EXP03_LOSS_ABLATION]
errs = [d["std"] for d in EXP03_LOSS_ABLATION]
x = np.arange(len(labels))
w = 0.40

# highlight Pure KL (index 0) in orange, others blue
f1_cols = [ps.PALETTE["highlight"] if i == 0 else ps.PALETTE["primary"]
           for i in range(len(labels))]
axL.bar(x - w / 2, f1, w, yerr=errs, color=f1_cols, edgecolor="black", lw=0.5,
        error_kw=dict(ecolor="#333", lw=0.8, capsize=2), label="$F_1$")
axL.set_ylabel("$F_1$ score", color=ps.PALETTE["primary"])
axL.set_ylim(0.80, 0.96)
axL.set_xticks(x)
axL.set_xticklabels(labels)
axL.axhline(0.916, color=ps.PALETTE["highlight"], ls=":", lw=0.8, alpha=0.7)
axL.set_title("(a) Loss function ablation", weight="bold")
for xi, v, e in zip(x, f1, errs):
    axL.text(xi - w / 2, v + e + 0.004, f"{v:.3f}", ha="center", fontsize=6.8)

axK = axL.twinx()
axK.bar(x + w / 2, kl, w, color=ps.PALETTE["secondary"], edgecolor="black",
        lw=0.5, alpha=0.85, label="KL")
axK.set_ylabel("KL divergence to BF16 teacher", color=ps.PALETTE["secondary"])
axK.set_ylim(0, 0.37)
axK.tick_params(axis="y", colors=ps.PALETTE["secondary"])
axK.grid(False)
axK.spines["top"].set_visible(False)
for xi, v in zip(x, kl):
    axK.text(xi + w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.6,
             color=ps.PALETTE["secondary"])
axL.annotate("best $F_1$ &\nlowest KL", xy=(0 - w / 2, 0.916 + errs[0]),
             xytext=(0.55, 0.938), fontsize=7, color=ps.PALETTE["highlight"],
             ha="center",
             arrowprops=dict(arrowstyle="->", color=ps.PALETTE["highlight"], lw=0.9))

# --- (b) teacher selection -------------------------------------------------
tlabels = [d["teacher"] for d in EXP09_TEACHER]
f1_fixed = [d["f1_fixed"] for d in EXP09_TEACHER]
f1_conv = [d["f1_conv"] for d in EXP09_TEACHER]
tokens = [d["tokens_B"] for d in EXP09_TEACHER]
xt = np.arange(len(tlabels))

# highlight homologous 0.5B (index 0)
fixed_cols = [ps.PALETTE["highlight"] if i == 0 else ps.PALETTE["primary"]
              for i in range(len(tlabels))]
axR.bar(xt - w / 2, f1_fixed, w, color=fixed_cols, edgecolor="black", lw=0.55,
        label="Fixed 0.5B tokens")
axR.bar(xt + w / 2, f1_conv, w, color=ps.PALETTE["tertiary"], edgecolor="black",
        lw=0.55, label="To convergence")
axR.set_ylabel("$F_1$ score")
axR.set_ylim(0.85, 0.93)
axR.set_xticks(xt)
axR.set_xticklabels(tlabels)
axR.axhline(0.916, color="#888", ls="--", lw=0.8)
axR.text(len(tlabels) - 1, 0.917, "0.5B ref", ha="right", fontsize=6.8, color="#666")
axR.set_title("(b) Teacher selection", weight="bold")
axR.legend(loc="upper right", fontsize=6.5)
for xi, tok in zip(xt, tokens):
    axR.text(xi + w / 2, 0.86, f"{tok:.1f}B", ha="center", fontsize=6.5,
             color=ps.PALETTE["tertiary"])

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig5_loss_teacher_ablation.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig5_loss_teacher_ablation.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig5_loss_teacher_ablation.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig5_loss_teacher_ablation.png')}, .pdf and .tiff")
