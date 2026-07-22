"""
fig1_architecture.py  --  Paper Figure 1 (insertion order #1)

QAD-MultiGuard three-tier edge-cloud collaborative architecture.

Schematic only (no experiment numbers): a PIPL privacy banner, four input
modalities, Tier-1 on-device detection, Tier-2 cloud deep reasoning, and
Tier-3 multimodal risk fusion, with the cross-tier F_v upload and fast-path
data flows. The on-device latency annotations (P50 268 ms / P99 342 ms) and
the epsilon-LDP (epsilon=1.5) note match the paper body text.

Run:  python3 fig1_architecture.py
Out:  fig1_architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "stix"

# Colours
C_EDGE_EC, C_EDGE_BG     = "#2ca02c", "#F0F9F4"
C_CLOUD_EC, C_CLOUD_BG   = "#1f77b4", "#F0F6FC"
C_FUSION_EC, C_FUSION_BG = "#ff7f0e", "#FFF8F0"
C_ALERT, C_PRIVACY       = "#d62728", "#1f4060"

fig, ax = plt.subplots(figsize=(11.0, 7.4))
ax.set_xlim(0, 22.4)
ax.set_ylim(0, 15)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0, r=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.04,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=2))


def arrow(x1, y1, x2, y2, c="#444", lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=12,
                 color=c, lw=lw, zorder=1))


# --- Privacy banner --------------------------------------------------------
box(0.4, 14.0, 21.2, 0.7, "#EAF2FB", C_PRIVACY, 1.0)
ax.text(11.0, 14.35,
        r"Privacy (PIPL Art.23): raw audio never leaves the device; only the 128-d "
        r"non-invertible vector $F_v$ is transmitted ($\epsilon$-LDP, $\epsilon{=}1.5$, optional)",
        ha="center", va="center", fontsize=8.8, color=C_PRIVACY, weight="bold")

# --- Input modalities ------------------------------------------------------
box(0.4, 12.7, 21.2, 1.0, "#FFFFFF", C_PRIVACY, 1.0)
mods = ["SMS", "Voice call", "URL links", "Call metadata"]
for i, m in enumerate(mods):
    cx = 3.0 + i * 5.3
    ax.text(cx, 13.2, m, ha="center", va="center", fontsize=10, weight="bold")

# --- Tier 1: on-device detection ------------------------------------------
box(0.4, 9.0, 21.2, 3.1, C_EDGE_BG, C_EDGE_EC, 1.4)
ax.text(0.7, 11.8, "Tier-1  On-device detection  (Snapdragon 8 Gen 3, 240 MB Q4_K_M)",
        ha="left", va="center", fontsize=10, weight="bold", color=C_EDGE_EC)
edge_mods = [
    ("SMS module\n(12-d text feat.)", 3.0),
    ("URL module\n(6-d struct. feat.)", 7.4),
    ("Acoustic module\nWhisper-tiny enc.\n(no decoder head)", 11.8),
    ("On-device student\nQwen2.5-0.5B\ndraft $\\gamma{=}5$", 16.2),
    ("Local risk score\n(fast path)", 20.0),
]
for label, cx in edge_mods:
    box(cx - 1.85, 9.4, 3.7, 1.9, "#FFFFFF", C_EDGE_EC, 1.0)
    ax.text(cx, 10.35, label, ha="center", va="center", fontsize=8.2)

# --- Tier 2: cloud deep reasoning -----------------------------------------
box(0.4, 5.0, 21.2, 3.1, C_CLOUD_BG, C_CLOUD_EC, 1.4)
ax.text(0.7, 7.8, "Tier-2  Cloud deep reasoning  (NVFP4 0.5B, Blackwell GPU)",
        ha="left", va="center", fontsize=10, weight="bold", color=C_CLOUD_EC)
cloud_mods = [
    ("CoT reasoning\nstep-wise analysis", 3.6),
    ("Cloud model\nNVFP4 QAD\n+ OV-Freeze", 8.4),
    ("Layer-output\nvariance freeze\n(q,k,v,o-proj)", 13.2),
    ("Federated aggregator\n(privacy-preserving;\nno raw data)", 18.2),
]
for label, cx in cloud_mods:
    box(cx - 2.0, 5.4, 4.0, 1.9, "#FFFFFF", C_CLOUD_EC, 1.0)
    ax.text(cx, 6.35, label, ha="center", va="center", fontsize=8.2)

# --- Tier 3: multimodal risk fusion ---------------------------------------
box(0.4, 1.0, 21.2, 3.1, C_FUSION_BG, C_FUSION_EC, 1.4)
ax.text(0.7, 3.8, "Tier-3  Multimodal risk fusion  (on-device, L-BFGS sigmoid)",
        ha="left", va="center", fontsize=10, weight="bold", color=C_FUSION_EC)
for i, (lab, w) in enumerate([("$r_{text}$", 0.40), ("$r_{audio}$", 0.30),
                              ("$r_{url}$", 0.20), ("$r_{meta}$", 0.10)]):
    cx = 2.8 + i * 3.4
    box(cx - 1.5, 1.5, 3.0, 1.6, "#FFFFFF", C_FUSION_EC, 1.0)
    ax.text(cx, 2.55, lab, ha="center", va="center", fontsize=11)
    ax.text(cx, 1.95, f"w = {w:.2f}", ha="center", va="center", fontsize=8.5, color="#555")
# fusion equation + alert
box(16.6, 1.5, 4.6, 1.6, "#FFF0F0", C_ALERT, 1.2)
ax.text(18.9, 2.55, "Risk alert", ha="center", va="center",
        fontsize=10, weight="bold", color=C_ALERT)
ax.text(18.9, 1.95, "Safe / Medium / High", ha="center", va="center",
        fontsize=8.0, color="#555")
ax.text(11.0, 0.7, r"$r = \sigma(w_{text}\,r_{text} + w_{audio}\,r_{audio}"
        r" + w_{url}\,r_{url} + w_{meta}\,r_{meta} + b)$",
        ha="center", va="center", fontsize=10)

# --- Cross-tier flows ------------------------------------------------------
arrow(8.5, 12.7, 8.5, 12.1, c="#444")                   # inputs -> tier1
ax.text(9.0, 12.4, "Input modalities", ha="left", va="center",
        fontsize=8.0, style="italic", color="#666")
arrow(11.8, 9.4, 11.8, 8.1, c=C_CLOUD_EC, lw=1.6)       # F_v upload edge->cloud
ax.text(12.1, 8.7, r"$F_v$ upload ($\epsilon$-LDP)", ha="left", va="center",
        fontsize=8.0, color=C_CLOUD_EC)
arrow(21.0, 9.4, 21.0, 4.1, c=C_FUSION_EC, lw=1.4, style="-|>")  # fast path
ax.text(21.25, 6.6, "fast path", ha="left", va="center", rotation=90,
        fontsize=7.6, color=C_FUSION_EC)
arrow(8.4, 5.4, 8.4, 4.1, c=C_FUSION_EC, lw=1.6)        # cloud->fusion
ax.text(8.7, 4.7, "scores + hidden states", ha="left", va="center",
        fontsize=8.0, color="#555")

# Latency annotation
ax.text(0.7, 0.35, "End-to-end per-window latency: P50 = 268 ms, P99 = 342 ms",
        ha="left", va="center", fontsize=8.5, style="italic", color="#333")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig1_architecture.png"), dpi=400, bbox_inches="tight", pad_inches=0.08)
fig.savefig(os.path.join(out, "fig1_architecture.tiff"), dpi=400, bbox_inches="tight", pad_inches=0.08)
print(f"saved {os.path.join(out, 'fig1_architecture.png')} and .tiff")
