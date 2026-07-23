"""
fig6_ovf_ablation.py  --  Paper Figure 6 (insertion order #6)

OV-Freeze ablation.
  (a) Layer selection: F1 (bars, left axis) and output-variance drift
      (red line, right axis) across attention-projection configurations.
      Full q,k,v,o-proj OV-Freeze gives the best F1 = 0.923 with drift cut
      from +18.2% (no OVF) to +1.3%.
  (b) Activation step-ratio: F1 (left axis) and PPL (right axis) across OVF
      activation windows. The final 30% window is Pareto-optimal.

All numbers from paper_data (EXP04 / EXP10).

Run:  python3 fig6_ovf_ablation.py
Out:  fig6_ovf_ablation.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps
from paper_data import EXP04_OVF_LAYER_ABLATION, EXP10_OVF_STEP_RATIO
import os

LIVE_MODE = os.environ.get("PAPER_DATA_USE_LIVE", "0") == "1"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                               gridspec_kw={"wspace": 0.45,
                                        "width_ratios": [1, 1.1]})

# --- (a) layer-selection ablation -----------------------------------------
labels = [d["config"] for d in EXP04_OVF_LAYER_ABLATION]
f1 = [d["f1"] for d in EXP04_OVF_LAYER_ABLATION]
drift = [d["drift_pct"] for d in EXP04_OVF_LAYER_ABLATION]
x = np.arange(len(labels))

best_idx = int(np.argmax(f1))   # q,k,v,o (ours)
best_drift_idx = int(np.argmin(drift))
cols = [ps.PALETTE["highlight"] if i == best_idx else ps.PALETTE["primary"]
        for i in range(len(labels))]
axL.bar(x, f1, 0.62, color=cols, edgecolor="black", lw=0.5)
axL.set_ylabel("$F_1$ score", color=ps.PALETTE["primary"])
f1_low = 0.910 if not LIVE_MODE else max(0.0, min(f1) - 0.02)
f1_high = 0.9265 if not LIVE_MODE else min(1.05, max(f1) + 0.02)
axL.set_ylim(f1_low, f1_high)
axL.set_xticks(x)
axL.set_xticklabels(labels)
axL.set_title("(a) Layer-selection ablation", weight="bold", pad=10)
for xi, v in zip(x, f1):
    axL.text(xi, min(v + max(0.0004, (f1_high - f1_low) * 0.02), f1_high - (f1_high - f1_low) * 0.03), f"{v:.3f}", ha="center", fontsize=6.4)

axD = axL.twinx()
axD.plot(x, drift, color=ps.PALETTE["secondary"], lw=1.4, marker="o", ms=4,
         label="variance drift")
axD.set_ylabel("Output variance drift (%)", color=ps.PALETTE["secondary"])
axD.set_ylim(0, 22)
axD.tick_params(axis="y", colors=ps.PALETTE["secondary"])
axD.grid(False)
axD.spines["top"].set_visible(False)
# place each drift value above-right of its marker so it clears the descending
# line and the right-hand axis title
for xi, v in zip(x, drift):
    axD.annotate(f"{v:.1f}", xy=(xi, v), xytext=(xi + 0.18, v + 1.1),
                 ha="left", va="bottom", fontsize=6.2,
                 color=ps.PALETTE["secondary"])
live_note = "best $F_1$ +\nlowest drift" if best_idx == best_drift_idx else "best $F_1$"
axL.annotate(live_note, xy=(best_idx, f1[best_idx]),
             xytext=(best_idx - 2.4, f1_high - (f1_high - f1_low) * 0.12), fontsize=7,
             color=ps.PALETTE["highlight"],
             arrowprops=dict(arrowstyle="->", color=ps.PALETTE["highlight"], lw=0.9))

# --- (b) activation step-ratio --------------------------------------------
ratios = [d["ratio_pct"] for d in EXP10_OVF_STEP_RATIO]
f1b = [d["f1"] for d in EXP10_OVF_STEP_RATIO]
ppl = [d["ppl"] for d in EXP10_OVF_STEP_RATIO]
best_b = int(np.argmax(f1b))

axR.plot(ratios, f1b, color=ps.PALETTE["primary"], lw=1.6, marker="o", ms=5,
         label="$F_1$")
axR.scatter([ratios[best_b]], [f1b[best_b]], s=120, facecolor="none",
            edgecolor=ps.PALETTE["highlight"], lw=1.8, zorder=5)
axR.set_ylabel("$F_1$ score", color=ps.PALETTE["primary"])
axR.set_xlabel("OV-Freeze activation step ratio (%)")
f1b_low = 0.914 if not LIVE_MODE else max(0.0, min(f1b) - 0.003)
f1b_high = 0.9245 if not LIVE_MODE else min(1.05, max(f1b) + 0.0035)
axR.set_ylim(f1b_low, f1b_high)
axR.set_xlim(min(ratios) - 5, max(ratios) + 5)
axR.set_title("(b) Step-ratio ablation", weight="bold", pad=10)
label_b = "ours: 30%" if 30 in ratios else f"best: {ratios[best_b]}%"
axR.annotate(label_b, xy=(ratios[best_b], f1b[best_b]),
             xytext=(ratios[best_b] + 3, min(f1b_high - 0.0008, f1b[best_b] + (f1b_high - f1b_low) * 0.18)), fontsize=7.5,
             color=ps.PALETTE["highlight"])
for r, v in zip(ratios, f1b):
    axR.text(r, min(v + max(0.0004, (f1b_high - f1b_low) * 0.04), f1b_high - (f1b_high - f1b_low) * 0.05), f"{v:.3f}", ha="center", fontsize=6.4)

axP = axR.twinx()
axP.plot(ratios, ppl, color=ps.PALETTE["secondary"], lw=1.3, ls="--",
         marker="s", ms=4, label="PPL")
axP.set_ylabel("Perplexity (PPL)", color=ps.PALETTE["secondary"])
ppl_low = 8.55 if not LIVE_MODE else min(ppl) - 0.08
ppl_high = 8.80 if not LIVE_MODE else max(ppl) + 0.08
axP.set_ylim(ppl_low, ppl_high)
axP.tick_params(axis="y", colors=ps.PALETTE["secondary"])
axP.grid(False)
axP.spines["top"].set_visible(False)
axP.axvspan(45, 55, color=ps.PALETTE["secondary"], alpha=0.06)

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig6_ovf_ablation.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig6_ovf_ablation.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig6_ovf_ablation.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig6_ovf_ablation.png')}, .pdf and .tiff")
