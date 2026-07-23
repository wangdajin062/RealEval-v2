"""
fig3_main_results.py  --  Paper Figure 3 (insertion order #3)

Main results on TAF-28k. Two horizontal-bar panels:
  (a) F1 score vs 11 baselines (blue), with the BF16 ceiling (0.931) and the
      three QAD rows (QAD, QAD+OVF, Q4_K_M+OVF) highlighted in orange;
  (b) Accuracy recovery rate (%) = F1 / 0.931 * 100, with the 99% target line.

All numbers come from paper_data (Table 4). Panel (b) values are computed so
that they can never drift from panel (a): recovery = F1 / BF16_F1 * 100.
This is the figure whose panel (b) previously carried stale 95.x values; the
recomputation here guarantees Figure 3 <-> Table 4 consistency.

Run:  python3 fig3_main_results.py
Out:  fig3_main_results.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps
from paper_data import (
    BF16_F1, BF16_F1_ERR, EXP01_QUANT_QUALITY, QAT_QAD_OVF,
    SAFE_QAQ_F1, SAFE_QAQ_F1_ERR,
)
import os

LIVE_MODE = os.environ.get("PAPER_DATA_USE_LIVE", "0") == "1"

# --- assemble the unified, ordered method list ----------------------------
# (name, F1, recovery, F1_err, colour-key)
methods = [("BF16 (upper)", BF16_F1, 100.0, BF16_F1_ERR, "ref")]
for m in EXP01_QUANT_QUALITY:
    methods.append((m["name"], m["f1"], m["recovery"], m["std"], "ptq"))
for m in QAT_QAD_OVF:
    key = "qat" if "QAT" in m["name"] else "ours"
    methods.append((m["name"], m["f1"], m["recovery"], m["f1_err"], key))
methods.append(("SAFE-QAQ", SAFE_QAQ_F1, None, SAFE_QAQ_F1_ERR, "safe"))

# recompute recovery from F1 so the two panels can never disagree
methods = [(n, f1, (round(f1 / BF16_F1 * 100, 1) if rec is not None else None),
            err, key) for (n, f1, rec, err, key) in methods]

color_map = {"ref": ps.PALETTE["neutral"], "ptq": ps.PALETTE["primary"],
             "bitdist": ps.PALETTE["purple"], "qat": ps.PALETTE["secondary"],
             "ours": ps.PALETTE["highlight"], "safe": ps.PALETTE["tertiary"]}
# BitDistiller gets its own colour
methods = [(n, f1, rec, err, ("bitdist" if "BitDist" in n else key))
           for (n, f1, rec, err, key) in methods]

short = {"BF16 (upper)": "BF16", "Plain RTN PTQ": "RTN", "NVFP4 + AWQ": "AWQ",
         "NVFP4 + GPTQ": "GPTQ", "NVFP4 + SpinQuant": "SpinQ",
         "NVFP4 + QuaRot": "QuaRot", "NVFP4 + BitDistill": "BitDist",
         "NVFP4 QAT (CE)": "QAT", "NVFP4 QAD": "QAD",
         "NVFP4 QAD + OV-Freeze": "QAD+OVF",
         "Q4_K_M QAD + OV-Freeze": "Q4K+OVF", "SAFE-QAQ": "SAFE-QAQ"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.7, 4.0),
                               gridspec_kw={"wspace": 0.25,
                                            "width_ratios": [1.25, 1]})

# --- panel (a): F1 ---------------------------------------------------------
names = [m[0] for m in methods]
y = np.arange(len(methods))[::-1]
f1s = [m[1] for m in methods]
errs = [m[3] for m in methods]
cols = [color_map[m[4]] for m in methods]
ax1.barh(y, f1s, xerr=errs, color=cols, edgecolor="black", lw=0.5,
         error_kw=dict(ecolor="#333", lw=0.8, capsize=2))
ax1.set_yticks(y)
ax1.set_yticklabels(names)
f1_left = 0.79 if not LIVE_MODE else min(0.79, min(f1s) - 0.01)
f1_right = 0.965 if not LIVE_MODE else max(0.965, max(f1s) + max(errs) + 0.01)
ax1.set_xlim(f1_left, f1_right)

ax1.set_title("(a) $F_1$ across all methods", weight="bold")
ax1.title.set_position((0.35, 1.0))
ax1.axvline(BF16_F1, color="#555", ls="--", lw=0.9)
ax1.text(BF16_F1 + 0.002, len(methods) / 2, "BF16 ceiling", rotation=90,
         ha="left", va="center", fontsize=7, color="#888")
for yi, f1, e in zip(y, f1s, errs):
    ax1.text(min(f1 + (e or 0) + 0.004, f1_right - 0.01), yi, f"{f1:.3f}", va="center", fontsize=7.2)
ax1.grid(axis="x", alpha=0.25)
ax1.grid(axis="y", visible=False)

# --- panel (b): recovery ---------------------------------------------------
mb = [m for m in methods if m[2] is not None]
yb = np.arange(len(mb))[::-1]
recs = [m[2] for m in mb]
colsb = [color_map[m[4]] for m in mb]
ax2.barh(yb, recs, color=colsb, edgecolor="black", lw=0.5)
ax2.set_yticks(yb)
ax2.set_yticklabels([short[m[0]] for m in mb])
rec_left = 89 if not LIVE_MODE else min(89, min(recs) - 1.0)
rec_right = 102 if not LIVE_MODE else max(102, max(recs) + 1.8)
ax2.set_xlim(rec_left, rec_right)

ax2.set_title("(b) Accuracy recovery", weight="bold")
ax2.title.set_position((0.4, 1.0))
ax2.axvline(99.0, color=ps.PALETTE["highlight"], ls=":", lw=1.0)
ax2.text(99.8, len(mb) / 2.0, "99% target", rotation=90,
         ha="right", va="center", fontsize=7, color=ps.PALETTE["highlight"])
for yi, r in zip(yb, recs):
    ax2.text(min(r + 0.15, rec_right - 0.6), yi, f"{r:.1f}", va="center", fontsize=7.2)
ax2.grid(axis="x", alpha=0.25)
ax2.grid(axis="y", visible=False)

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig3_main_results.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig3_main_results.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig3_main_results.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig3_main_results.png')}, .pdf and .tiff")
