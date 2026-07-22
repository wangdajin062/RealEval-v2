"""
fig2_acoustic_embedding.py  --  Paper Figure 2 (insertion order #2)

128-dimensional non-invertible acoustic embedding F_v construction.

Three panels:
  (a) original 64-band log-mel spectrogram (T ~ 300 frames per 3-s window);
  (b) time-averaged 64-d MFCC vector (collapses frame-level dynamics);
  (c) final 128-d F_v = [ time-averaged MFCC (64-d) ; W_proj * pooled
      Whisper-tiny encoder feature (64-d) ].

Schematic / illustrative panels (synthetic mel surface with a fixed seed);
the privacy headline (WER >= 0.95 under white-box + black-box GLO attacks)
matches Table 9 and the paper body.

Run:  python3 fig2_acoustic_embedding.py
Out:  fig2_acoustic_embedding.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps  # noqa: F401  (applies global rcParams on import)
import os

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9),
                         gridspec_kw={"wspace": 0.50,
                                      "width_ratios": [1, 1, 1.1]})

# --- synthetic speech-like mel surface (fixed seed for reproducibility) ----
rng = np.random.RandomState(7)
n_time, n_mel = 200, 64
t = np.arange(n_time)
m = np.arange(n_mel)
T, M = np.meshgrid(t, m, indexing="xy")
mel = (np.sin(T * 0.07) * np.exp(-((M - 10) ** 2) / 60) +
       np.cos(T * 0.05 + 1.2) * np.exp(-((M - 25) ** 2) / 90) * 0.7 +
       np.sin(T * 0.10 + 2.0) * np.exp(-((M - 45) ** 2) / 200) * 0.5)
mel += rng.normal(0, 0.10, size=mel.shape)
mel = (mel - mel.min()) / (mel.max() - mel.min())

# (a) original log-mel spectrogram
ax = axes[0]
ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
ax.set_xlabel("Time (frames)")
ax.set_ylabel("Mel bin")
ax.set_title("(a) Original log-mel spectrogram")
ax.text(0.98, 0.96, r"$T \sim 300$ frames", ha="right", va="top",
        transform=ax.transAxes, fontsize=6.5, color="white", style="italic",
        bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))

# (b) time-averaged 64-d MFCC
ax = axes[1]
mfcc_avg = mel.mean(axis=1)
ax.bar(np.arange(64), mfcc_avg, color="#1f77b4", edgecolor="none", width=0.95)
ax.set_xlabel("MFCC coefficient")
ax.set_ylabel("Time-averaged log-magnitude")
ax.set_title("(b) Time-averaged 64-d MFCC")
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0, mfcc_avg.max() * 1.34)
ax.text(31.5, mfcc_avg.max() * 1.22,
        r"non-invertible: $\sum_t \to \bar{f}_{\rm mfcc}$",
        ha="center", fontsize=6.2, color="#7a3000", style="italic")

# (c) final 128-d F_v
ax = axes[2]
W_proj = rng.normal(0, 1.0 / np.sqrt(384), size=(64, 384))
h_w = rng.normal(0, 1, 384)
proj = W_proj @ h_w
F_v = np.concatenate([mfcc_avg, proj])
F_v_norm = (F_v - F_v.min()) / (F_v.max() - F_v.min())
cols = ["#1f77b4"] * 64 + ["#ff7f0e"] * 64
ax.bar(np.arange(128), F_v_norm, color=cols, edgecolor="none", width=0.95)
ax.set_xlabel(r"$F_v$ dimension")
ax.set_ylabel("Normalised value")
ax.set_title(r"(c) Final 128-d $F_v$ for transmission")
ax.set_xlim(-0.5, 127.5)
ax.axvline(63.5, color="#444", lw=0.8, ls="--", alpha=0.6)
ax.text(32, 1.18, r"$\bar{f}_{\rm mfcc}\in\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#1f4060", weight="bold")
ax.text(96, 1.18, r"$W_{\rm proj}\,\bar{h}_w\in\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#8a4a00", weight="bold")
ax.set_ylim(0, 1.30)

fig.text(0.5, -0.07,
         r"Both steps (time-averaging + global pooling) drop $O(T)$ bits of "
         r"frame-level information; WER $\geq$ 0.95 under white-box + "
         r"black-box GLO attacks",
         ha="center", fontsize=6.5, style="italic", color="#222")

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig2_acoustic_embedding.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig2_acoustic_embedding.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig2_acoustic_embedding.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig2_acoustic_embedding.png')}, .pdf and .tiff")
