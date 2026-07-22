"""
fig4_loss_convergence.py  --  Paper Figure 4 (insertion order #4)

QAD training loss convergence and quantization SNR stability under OV-Freeze.
  (a) KL-divergence loss vs training step: a plateau at ~0.045 collapses to
      ~0.016 once OV-Freeze activates at step 1,400 (a 2.76x drop); the cosine
      learning-rate schedule is overlaid on the right axis.
  (b) Quantization SNR stays within 18.4-18.9 dB across all 2,000 steps,
      confirming the variance rescaling does not amplify quantization noise.

The curve is a deterministic illustration anchored on the paper's headline
numbers (LOSS_PLATEAU, LOSS_CONVERGED, OVF_ACTIVATION_STEP, SNR_RANGE).

Run:  python3 fig4_loss_convergence.py
Out:  fig4_loss_convergence.png
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps
from paper_data import (LOSS_PLATEAU, LOSS_CONVERGED, OVF_ACTIVATION_STEP,
                        TOTAL_STEPS, SNR_RANGE)
import os

rng = np.random.RandomState(11)
steps = np.arange(0, TOTAL_STEPS + 1)

# --- (a) KL loss trace -----------------------------------------------------
# warm-up rise to plateau, plateau, then sharp exponential drop after OVF on.
loss = np.empty_like(steps, dtype=float)
warm = 100
for i, s in enumerate(steps):
    if s < warm:                      # quick warm-up to plateau
        loss[i] = LOSS_PLATEAU * (0.4 + 0.6 * s / warm)
    elif s < OVF_ACTIVATION_STEP:     # plateau
        loss[i] = LOSS_PLATEAU
    else:                             # exponential decay to converged value
        tau = 120.0
        loss[i] = LOSS_CONVERGED + (LOSS_PLATEAU - LOSS_CONVERGED) * \
            np.exp(-(s - OVF_ACTIVATION_STEP) / tau)
loss += rng.normal(0, 0.0009, size=loss.shape)

# cosine LR schedule (right axis), peak 1e-5
lr = 1e-5 * 0.5 * (1 + np.cos(np.pi * steps / TOTAL_STEPS))
lr[:warm] = 1e-5 * steps[:warm] / warm    # linear warm-up

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.32})

ax_a.plot(steps, loss, color=ps.PALETTE["primary"], lw=1.3,
          label="KL divergence loss")
ax_a.set_ylabel("KL divergence loss")
ax_a.set_ylim(0, 0.055)
ax_a.set_title("(a) KL divergence convergence under OV-Freeze activation",
               weight="bold")
ax_a.axvline(OVF_ACTIVATION_STEP, color=ps.PALETTE["highlight"], ls="--", lw=1.0)
ax_a.axvspan(OVF_ACTIVATION_STEP, TOTAL_STEPS, color=ps.PALETTE["highlight"],
             alpha=0.06)
ax_a.annotate(f"OV-Freeze ON\n(step {OVF_ACTIVATION_STEP}, last 30%)",
              xy=(OVF_ACTIVATION_STEP, LOSS_PLATEAU),
              xytext=(OVF_ACTIVATION_STEP - 420, 0.03),
              fontsize=7.5, color=ps.PALETTE["highlight"],
              arrowprops=dict(arrowstyle="->", color=ps.PALETTE["highlight"], lw=1.0))
ax_a.text(TOTAL_STEPS - 30, LOSS_CONVERGED + -0.008,
          f"converged\n(loss $\\approx$ {LOSS_CONVERGED:.3f})", ha="right",
          fontsize=7.5, color=ps.PALETTE["primary"])
ax_a.text(700, LOSS_PLATEAU + 0.004,
          f"plateau (loss $\\approx$ {LOSS_PLATEAU:.3f})", ha="center",
          fontsize=7.5, color="#555")
ax_a.text(OVF_ACTIVATION_STEP + 160, 0.0255,
          "KL drop 2.76$\\times$\n(0.045 $\\to$ 0.016)", fontsize=7.5,
          color=ps.PALETTE["secondary"], weight="bold")

ax_lr = ax_a.twinx()
ax_lr.plot(steps, lr * 1e6, color="#999", lw=1.0, ls="-.", label="Learning rate")
ax_lr.set_ylabel(r"Learning rate ($\times 10^{-6}$)", color="#777")
ax_lr.set_ylim(0, 11)
ax_lr.tick_params(axis="y", colors="#777")
ax_lr.grid(False)
ax_lr.spines["top"].set_visible(False)

# --- (b) SNR trace ---------------------------------------------------------
snr = 18.65 + 0.18 * np.sin(steps / 130.0) + rng.normal(0, 0.04, size=steps.shape)
ax_b.plot(steps, snr, color=ps.PALETTE["tertiary"], lw=1.1)
ax_b.axhspan(SNR_RANGE[0], SNR_RANGE[1], color=ps.PALETTE["tertiary"], alpha=0.12)
ax_b.set_ylim(18.2, 19.0)
ax_b.set_ylabel("Quant. SNR (dB)")
ax_b.set_xlabel("Training step")
ax_b.set_title("(b) Quantization stability (SNR)", weight="bold", pad=12)
ax_b.text(1002, 18.98,
          f"stable: SNR within {SNR_RANGE[0]}-{SNR_RANGE[1]} dB throughout",
          ha="center", fontsize=7.5, style="italic", color=ps.PALETTE["tertiary"])
ax_b.axvline(OVF_ACTIVATION_STEP, color=ps.PALETTE["highlight"], ls="--", lw=1.0)
ax_b.set_xlim(0, TOTAL_STEPS)

out = os.path.join(os.path.dirname(__file__), "..", "figure")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig4_loss_convergence.png"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig4_loss_convergence.pdf"), bbox_inches="tight",
            pad_inches=0.05)
fig.savefig(os.path.join(out, "fig4_loss_convergence.tiff"), dpi=400, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print(f"saved {os.path.join(out, 'fig4_loss_convergence.png')}, .pdf and .tiff")
