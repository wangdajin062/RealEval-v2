#!/usr/bin/env python3
"""
fig8_revision_ablations.py
===========================
Generates Figure 12 for QAD-MultiGuard v9: a three-panel figure visualising the
three new quantitative results introduced during the revision rounds.

  (a) Heterogeneous vs homogeneous quantization ablation        [Sec 3.2.3, M3]
  (b) AdvFraud-3k curated-subset vs full-pool robustness        [Sec 4,     M4]
  (c) epsilon-LDP privacy-utility trade-off                     [Sec 5,     B3]

This DATA dict is the single source of truth for the figure and MUST match the
numbers reported in paper1_en_v9.tex. Any change here requires the same change
in the .tex (and vice versa). The script prints all plotted values at the end
for a quick consistency check against the manuscript.

Output: fig8_revision_ablations.png  (400 dpi high-resolution)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paper_style as ps  # shared journal-grade styling (font, palette, dpi)
from paper_data import EXP12_REVISION_ABLATIONS

LIVE_MODE = os.environ.get("PAPER_DATA_USE_LIVE", "0") == "1"

# Keep plotting-field names unchanged; values now come from paper_data bridge.
DATA = EXP12_REVISION_ABLATIONS

# --------------------------------------------------------------------------- #
# Colour aliases from the shared paper_style palette
# --------------------------------------------------------------------------- #
C_PRIMARY = ps.PALETTE["highlight"]   # orange  (our / preferred)
C_SECOND  = ps.PALETTE["primary"]     # blue    (baseline / alternative)
C_REF     = ps.PALETTE["secondary"]   # red     (reference line)
C_LAT     = ps.PALETTE["tertiary"]    # green   (latency)


def _annotate_bars(ax, bars, fmt="{:.3f}", dy=0.0008):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=7.5)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # ---- Panel (a): quantization scheme ---------------------------------- #
    ax = axes[0]
    q = DATA["quant"]
    x = np.arange(len(q["labels"]))
    bars = ax.bar(x, q["f1"], width=0.55,
                  color=[C_SECOND, C_PRIMARY], edgecolor="black", linewidth=0.6)
    ax.axhline(q["bf16_ref"], color=C_REF, ls="--", lw=1.0,
               label=f"BF16 baseline ({q['bf16_ref']:.3f})")
    _annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(q["labels"])
    ax.set_ylim(0.90, 0.935)
    ax.set_ylabel(r"TAF-28k $F_1$")
    ax.set_title("(a) Quantization scheme", weight="bold")
    ax.legend(loc="lower left", frameon=False)
    ax.annotate(f"+{q['delta']:.3f}\n($p<0.05$)",
                xy=(1, q["f1"][1]), xytext=(0.5, 0.917),
                ha="center", fontsize=7.5, color=C_PRIMARY)

    # ---- Panel (b): AdvFraud curated vs full pool ------------------------ #
    ax = axes[1]
    a = DATA["advfraud"]
    x = np.arange(len(a["labels"]))
    bars = ax.bar(x, a["f1"], width=0.55,
                  color=[C_SECOND, C_PRIMARY], edgecolor="black", linewidth=0.6)
    ax.axhline(a["bf16_matched"], color=C_REF, ls="--", lw=1.0,
               label=f"BF16 matched ({a['bf16_matched']:.3f})")
    _annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(a["labels"])
    adv_low = 0.80 if not LIVE_MODE else min(0.80, min(a["f1"]) - 0.01)
    adv_high = 0.89 if not LIVE_MODE else max(0.89, max(a["f1"] + [a["bf16_matched"]]) + 0.01)
    ax.set_ylim(adv_low, adv_high)
    ax.set_ylabel(r"AdvFraud-3k $F_1$ (QAD+OVF)")
    ax.set_title("(b) AdvFraud-3k: curated vs full pool", weight="bold")
    ax.legend(loc="lower left", frameon=False)

    # ---- Panel (c): epsilon-LDP privacy-utility trade-off ---------------- #
    ax = axes[2]
    l = DATA["ldp"]
    x = np.arange(len(l["labels"]))
    w = 0.38
    bars1 = ax.bar(x - w / 2, l["f1"], width=w, color=C_PRIMARY,
                   edgecolor="black", linewidth=0.6, label=r"TAF-28k $F_1$")
    _annotate_bars(ax, bars1, fmt="{:.3f}")
    ax.set_ylim(0.88, 0.93)
    ax.set_ylabel(r"TAF-28k $F_1$", color=C_PRIMARY)
    ax.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(l["labels"])
    ax.set_title(r"(c) $\epsilon$-LDP privacy-utility", weight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w / 2, l["latency"], width=w, color=C_LAT,
                    edgecolor="black", linewidth=0.6, label="P50 latency (ms)")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.3, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(260, 280)
    ax2.set_ylabel("P50 latency (ms)", color=C_LAT)
    ax2.tick_params(axis="y", labelcolor=C_LAT)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figure")
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, "fig8_revision_ablations.png")
    pdf = os.path.join(out, "fig8_revision_ablations.pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    tiff = os.path.join(out, "fig8_revision_ablations.tiff")
    fig.savefig(tiff, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # ---- Consistency print-out ------------------------------------------ #
    print("Saved:", png, ",", pdf, "and", tiff)
    print("\n=== fig8 plotted values (cross-check against paper1_en_v9.tex) ===")
    print("(a) quant   : homogeneous INT4 F1 =", DATA["quant"]["f1"][0],
          "| heterogeneous F1 =", DATA["quant"]["f1"][1],
          "| BF16 ref =", DATA["quant"]["bf16_ref"],
          "| delta =", DATA["quant"]["delta"])
    print("(b) advfraud: full-pool F1 =", DATA["advfraud"]["f1"][0],
          "| curated-517 F1 =", DATA["advfraud"]["f1"][1],
          "| BF16 matched =", DATA["advfraud"]["bf16_matched"])
    print("(c) ldp     : F1 no-LDP =", DATA["ldp"]["f1"][0],
          "-> eps-LDP =", DATA["ldp"]["f1"][1],
          "(drop", round(DATA["ldp"]["f1"][0] - DATA["ldp"]["f1"][1], 3), ")",
          "| latency", DATA["ldp"]["latency"][0], "->", DATA["ldp"]["latency"][1], "ms")


if __name__ == "__main__":
    main()
