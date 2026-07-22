"""
fig7_speculative_decoding.py  --  Paper Figure 7 (insertion order #7)

Speculative-decoding analysis.
  (a) Theoretical speedup curves S(alpha) = (1 - alpha^(gamma+1)) / (1 - alpha)
      for gamma in {3,5,7,10}, with the two operating points marked: generic
      draft (alpha=0.78, gamma=5, 3.52x) and our anti-fraud domain-tuned draft
      (alpha=0.86, gamma=5, 4.25x).
  (b) Measured wall-clock speedup on H100 and Snapdragon 8 Gen 3 at the
      deployed alpha=0.86 across gamma, with gamma=5 marked Pareto-optimal
      (matches Table 8: H100 3.49x, SD8G3 3.32x).

All numbers from paper_data (EXP05 / Table 8).

Run:  python3 fig7_speculative_decoding.py
Out:  fig7_speculative_decoding.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps
from paper_data import (speedup, EXP05_SPECULATIVE, SPEC_ALPHA_GENERIC,
                        SPEC_ALPHA_TUNED, SPEC_GAMMA_DEPLOY)
import os

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                               gridspec_kw={"wspace": 0.150,
                                        "width_ratios": [1.05, 1]})

# --- (a) theoretical speedup curves ---------------------------------------
alphas = np.linspace(0.5, 0.95, 200)
gamma_styles = {3: ("#bbbbbb", "-"), 5: (ps.PALETTE["primary"], "-"),
                7: (ps.PALETTE["tertiary"], "-"), 10: (ps.PALETTE["purple"], "-")}
for g, (c, ls) in gamma_styles.items():
    axA.plot(alphas, [speedup(a, g) for a in alphas], color=c, ls=ls,
             lw=(2.0 if g == 5 else 1.2), label=f"$\\gamma$ = {g}")

# operating points
g = SPEC_GAMMA_DEPLOY
axA.scatter([SPEC_ALPHA_GENERIC], [speedup(SPEC_ALPHA_GENERIC, g)], s=70,
            color=ps.PALETTE["secondary"], zorder=6, edgecolor="black", lw=0.6)
axA.annotate(f"generic\n($\\alpha$={SPEC_ALPHA_GENERIC}, $\\gamma$=5, "
             f"{speedup(SPEC_ALPHA_GENERIC, g):.2f}$\\times$)",
             xy=(SPEC_ALPHA_GENERIC, speedup(SPEC_ALPHA_GENERIC, g)),
             xytext=(0.55, 4.1), fontsize=7,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axA.scatter([SPEC_ALPHA_TUNED], [speedup(SPEC_ALPHA_TUNED, g)], s=90,
            marker="*", color=ps.PALETTE["highlight"], zorder=6,
            edgecolor="black", lw=0.6)
axA.annotate(f"anti-fraud tuned\n($\\alpha$={SPEC_ALPHA_TUNED}, $\\gamma$=5, "
             f"{speedup(SPEC_ALPHA_TUNED, g):.2f}$\\times$)",
             xy=(SPEC_ALPHA_TUNED, speedup(SPEC_ALPHA_TUNED, g)),
             xytext=(0.8, 2.3), fontsize=7, color=ps.PALETTE["highlight"],
             arrowprops=dict(arrowstyle="->", color=ps.PALETTE["highlight"], lw=0.9))
axA.set_xlabel(r"Token acceptance rate $\alpha$")
axA.set_ylabel("Theoretical speedup")
axA.set_title(r"(a) Speed-up curve ($\alpha$ vs $\gamma$)", weight="bold", pad=12)
axA.set_xlim(0.5, 1)
axA.set_ylim(1.5, 6.5)
axA.legend(loc="upper left", ncol=1)

# --- (b) measured speedup at alpha=0.86 -----------------------------------
rows = EXP05_SPECULATIVE[SPEC_ALPHA_TUNED]
gammas = [r["gamma"] for r in rows]
h100 = [r["h100"] for r in rows]
sd8 = [r["sd8g3"] for r in rows]
theor = [speedup(SPEC_ALPHA_TUNED, g) for g in gammas]
x = np.arange(len(gammas))
w = 0.26

axB.bar(x - w, theor, w, color="#cccccc", edgecolor="black", lw=0.4,
        label="Theoretical")
axB.bar(x, h100, w, color=ps.PALETTE["primary"], edgecolor="black", lw=0.4,
        label="Measured (H100)")
axB.bar(x + w, sd8, w, color=ps.PALETTE["highlight"], edgecolor="black", lw=0.4,
        label="Measured (SD8G3)")
axB.set_xticks(x)
axB.set_xticklabels([f"$\\gamma$={g}" for g in gammas])
# highlight the deployed gamma=5 tick label
gi = gammas.index(SPEC_GAMMA_DEPLOY)
axB.get_xticklabels()[gi].set_color(ps.PALETTE["highlight"])
axB.get_xticklabels()[gi].set_fontweight("bold")
axB.set_ylabel(r"Speed-up factor ($\times$)")
axB.set_title(r"(b) Measured at $\alpha$ = 0.86", weight="bold", pad=12)
axB.set_ylim(0, 6.3)
axB.legend(loc="upper left", fontsize=6)
# mark gamma=5 as the deployed Pareto-optimal point with a small bracket label
gi = gammas.index(SPEC_GAMMA_DEPLOY)
axB.annotate("Pareto-optimal\n(deployed)", xy=(gi, -0.02),
             xytext=(gi, -0.7), textcoords="data", xycoords="data",
             ha="center", va="top", fontsize=6.6, color=ps.PALETTE["highlight"],
             annotation_clip=False)
for xi, (h, s) in zip(x, zip(h100, sd8)):
    axB.text(xi, h + 0.08, f"{h:.2f}", ha="center", fontsize=5.8,
             color=ps.PALETTE["primary"])
    axB.text(xi + w, s + 0.08, f"{s:.2f}", ha="center", fontsize=5.8,
             color=ps.PALETTE["highlight"])

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig7_speculative_decoding.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig7_speculative_decoding.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig7_speculative_decoding.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig7_speculative_decoding.png')}, .pdf and .tiff")
