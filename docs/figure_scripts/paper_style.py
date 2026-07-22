"""
paper_style.py - Shared SCI/IEEE-Elsevier styling for all QAD-MultiGuard figures.

Every figure script imports from here so that fonts, colours, sizes and the
save() routine stay identical across the whole paper. The colour palette is
colour-blind-safe and journal-grade.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Global rcParams -------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Colour palette (colour-blind safe) ------------------------------------
PALETTE = {
    "primary":   "#1f77b4",   # PTQ baselines (blue)
    "secondary": "#d62728",   # QAT / drift (red)
    "tertiary":  "#2ca02c",   # SAFE-QAQ / auxiliary (green)
    "neutral":   "#7f7f7f",   # BF16 reference (grey)
    "highlight": "#ff7f0e",   # our method (orange)
    "purple":    "#9467bd",   # BitDistiller
    "teal":      "#17becf",
    "brown":     "#8c564b",
}

FIG_W_SINGLE = 3.5     # single column (inches)
FIG_W_DOUBLE = 7.16    # double column (inches)
FIG_H_DEFAULT = 2.5


def save(fig, path, w=FIG_W_DOUBLE, h=FIG_H_DEFAULT):
    """Set canonical size and write a 400-dpi PNG."""
    fig.set_size_inches(w, h)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {path}")
