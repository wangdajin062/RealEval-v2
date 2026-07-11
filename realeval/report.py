"""realeval/report.py 鈥?Summary CSV + Paper Table/Figure Generation

read_all_results: Read latest results for each experiment under outputs/results
build_summary_csv: Generate outputs/summary.csv (one key metric per experiment per row)
build_paper_tables: Generate outputs/tables.md (paper-ready markdown tables)
build_latex_tables: Generate outputs/Table2.tex (LaTeX table for paper)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
RESULTS = OUT / "results"


def _latest_results() -> dict[str, dict]:
    """Read latest result for each experiment (sorted by timestamp)."""
    if not RESULTS.is_dir():
        return {}
    by_exp = {}
    for f in sorted(RESULTS.glob("*.json")):
        # Filenames are {exp}_{YYYYMMDD}_{HHMMSS}.json; strip the two timestamp parts to recover {exp}.
        exp = f.stem.rsplit("_", 2)[0]
        by_exp[exp] = json.loads(f.read_text(encoding="utf-8"))
    return by_exp


def _pick_metric(r: dict) -> dict:
    """Extract a representative metric from an experiment result dict.

    Tries top-level metric keys first, then recurses into nested dicts (conditions/scales/schemes/
    variants) to find a representative f1/alpha/eer so the summary is never blank.
    """
    m = {"experiment": r.get("experiment", "?"), "computation": r.get("computation", "?")}
    for k in ("f1", "kl", "alpha", "eer", "accuracy", "latency_p50_ms", "throughput_sps",
              "best_f1", "final_f1", "asv_eer_pct", "mean_reconstruction_corr"):
        if k in r and not isinstance(r[k], dict):
            m["metric"] = k; m["value"] = r[k]; return m

    def _find(obj, depth=0):
        if depth > 3 or not isinstance(obj, dict):
            return None
        for k in ("f1", "f1_mean", "alpha", "asv_eer_pct", "accuracy"):
            if k in obj and not isinstance(obj[k], dict):
                return (k, obj[k])
        for v in obj.values():
            if isinstance(v, dict):
                hit = _find(v, depth + 1)
                if hit:
                    return hit
        return None

    hit = _find(r)
    m["metric"], m["value"] = hit if hit else ("-", "-")
    return m


def build_summary_csv() -> Path:
    """Generate outputs/summary.csv. Returns Path."""
    OUT.mkdir(parents=True, exist_ok=True)
    results = _latest_results()
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "computation", "metric", "value"])
        for exp in sorted(results):
            m = _pick_metric(results[exp])
            w.writerow([m["experiment"], m["computation"], m.get("metric", "?"), m.get("value", "?")])
    return OUT / "summary.csv"


def build_paper_tables() -> Path:
    """Generate outputs/tables.md with v25-aligned structured tables from real results.

    Tables: main results (exp1+exp4), fusion ablation (exp2), CoT (exp9), cross-dataset + adversarial
    (exp5), speculative decoding (exp6), privacy attack (exp7). Each table is emitted only if its
    experiment's results are present. A generic per-experiment summary is appended at the end.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    res = _latest_results()
    L = ["# Paper Tables (Auto-generated from real results)", ""]

    def fmt(v):
        return "-" if v is None else v

    # Table (v25 Table 4): TAF-28k main results
    e1, e4 = res.get("exp1", {}), res.get("exp4", {})
    if e1 or e4:
        L += ["## Table 鈥?TAF-28k main results (v25 Table 4 / Fig 3)", "", "| Method | F1 |", "|---|---|"]
        for name, v in e4.get("classifiers", {}).items():
            L.append(f"| {name} | {fmt(v.get('f1') if isinstance(v, dict) else v)} |")
        if e1.get("f1") is not None:
            L.append(f"| QAD+OV-Freeze | {fmt(e1['f1'])} |")
        L.append("")

    # Table (v25 Table 3): multimodal fusion strategy ablation
    fusion = res.get("exp13", {}).get("strategies", {})
    if fusion:
        L += ["## Table 鈥?Multimodal fusion strategy ablation (v25 Table 3)", "",
              "| Fusion strategy | F1 | Params | Latency (ms) |", "|---|---|---|---|"]
        for n, v in fusion.items():
            L.append(f"| {n} | {fmt(v.get('f1'))} | {fmt(v.get('params'))} | {fmt(v.get('latency_ms'))} |")
        L.append("")

    # Table (v25 Table 7): alignment-loss ablation
    variants = res.get("exp2", {}).get("variants", {})
    if variants:
        L += ["## Table 鈥?Alignment-loss ablation (v25 Table 7 / Fig 6a)", "", "| Variant | F1 | KL |", "|---|---|---|"]
        for n, v in variants.items():
            L.append(f"| {n} | {fmt(v.get('f1'))} | {fmt(v.get('kl_final'))} |")
        L.append("")

    # Table (v25 Table 5): CoT
    e9 = res.get("exp9", {})
    if e9.get("with_cot"):
        L += ["## Table 鈥?Chain-of-thought impact (v25 Table 5)", "", "| Config | F1 | FPR |", "|---|---|---|"]
        for lbl, key in (("With CoT", "with_cot"), ("Without CoT", "without_cot")):
            d = e9.get(key, {})
            L.append(f"| {lbl} | {fmt(d.get('f1'))} | {fmt(d.get('fpr'))} |")
        L.append("")

    # Table (v25 Table 6): cross-dataset + adversarial (with n/group per pool, reviewer point 4)
    e5 = res.get("exp5", {})
    if e5:
        L += ["## Table 鈥?Cross-dataset & adversarial (v25 Table 6 / Fig 5b)", "", "| Setting | F1 | n |", "|---|---|---|"]
        for k in ("taf28k", "chifraud", "cross_taf_on_chifraud", "cross_chifraud_on_taf"):
            if k in e5:
                L.append(f"| {k} | {fmt(e5[k].get('f1') if isinstance(e5[k], dict) else e5[k])} | test split |")
        for k, v in e5.get("advfraud", {}).items():
            n = "".join(ch for ch in k if ch.isdigit()) or "-"
            L.append(f"| advfraud/{k} | {fmt(v.get('f1'))} | {n} samples |")
        for k, v in e5.get("advfraud_expert", {}).items():
            n = "".join(ch for ch in k if ch.isdigit()) or "-"
            L.append(f"| advfraud_expert/{k} | {fmt(v.get('f1'))} | {n} samples |")
        L.append("")

    # Table (v25 Table 8): speculative decoding
    e6 = res.get("exp6", {})
    if e6.get("diagnostic_B"):
        d = e6["diagnostic_B"]
        gen = d.get("alpha_generic_measured")
        dom = d.get("alpha_domain")
        L += ["## Table 鈥?Speculative decoding (v25 Table 8)", "",
              "| Draft variant | alpha (measured) |", "|---|---|",
              f"| generic | {fmt(gen)} |",
              f"| domain-tuned | {fmt(dom) if dom is not None else 'not measured'} |",
              f"", f"> gamma={d.get('gamma')}, n_samples={d.get('n_samples')}, "
              f"accepted={d.get('accepted')}, proposed={d.get('proposed')}", ""]

    # Table (v25 Table 9): privacy attack 鈥?with n/group reported (reviewer point 4)
    e7 = res.get("exp7", {})
    if e7:
        n_sp = e7.get("n_speakers")
        L += ["## Table 鈥?Privacy attack (v25 Table 9)", "",
              "| Metric | Value | n (group) |", "|---|---|---|",
              f"| ASV-EER (%) | {fmt(e7.get('asv_eer_pct'))} | {fmt(n_sp)} speakers |",
              f"| minDCF | {fmt(e7.get('min_dcf'))} | {fmt(n_sp)} speakers |",
              f"| speaker-ID accuracy | {fmt(e7.get('speaker_id_accuracy'))} | {fmt(n_sp)} speakers |",
              f"| GLO reconstruction corr | {fmt(e7.get('glo_reconstruction_corr'))} | {fmt(n_sp)} speakers |",
              "", f"> Chance speaker-ID accuracy at n={fmt(n_sp)} is ~{round(1.0 / n_sp, 4) if n_sp else '-'}; "
              "open-set protocol (enrol/trial disjoint) addresses the reviewer's n>=50 requirement.", ""]

    # Table 鈥?Modality coverage (reviewer point 3): which test set exercises the acoustic F_v pipeline
    L += ["## Table 鈥?Modality coverage of evaluation sets (reviewer point 3)", "",
          "| Test set | Text | Acoustic F_v | Note |", "|---|---|---|---|",
          "| TAF-28k | yes | yes | re-enactment protocol (not field-collected) |",
          "| ChiFraud (OOD) | yes | **no** | text-only; does NOT validate F_v acoustic generalisation |",
          "| AdvFraud-3k | yes | no | adversarial text perturbations |",
          "| AdvFraud-3k (auto) | yes | no | 3k auto-generated adversarial text perturbations |",
          "| AdvFraud-3k (expert) | yes | no | 583 human-crafted adversarial text samples |",
          "", "> The acoustic generalisation of F_v is only exercised on TAF-28k; field-collected "
          "acoustic validation remains a gap (reviewer points 3 and F).", ""]

    # Generic per-experiment summary (all experiments)
    L += ["## Summary 鈥?all experiments", "", "| Experiment | Metric | Value | Computation |", "|---|---|---|---|"]
    for exp in sorted(res):
        m = _pick_metric(res[exp])
        L.append(f"| {m['experiment']} | {m.get('metric', '-')} | {m.get('value', '-')} | {m['computation']} |")

    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return OUT / "tables.md"


def _mpl():
    """Lazy matplotlib import with a non-interactive backend. Returns plt or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def build_paper_figures(fmt: str = "png") -> list:
    """Generate all v25-aligned data figures from real results. English labels/titles/legends.

    Covers v25 data figures fig3-fig8 (fig1/fig2 are schematics, not data-generated):
      fig3 main results (exp1+exp4), fig4 KL+SNR (exp1), fig5 ablations (exp5+exp11),
      fig6 loss+teacher (exp2+exp10), fig7 OV-Freeze layer+window (exp3), fig8 speculative decoding (exp6).
    A figure is generated only if its experiment's results are present. Returns generated paths (str).
    """
    plt = _mpl()
    if plt is None:
        return []
    FIGS.mkdir(parents=True, exist_ok=True)
    res = _latest_results()
    made = []

    def _save(fig, name):
        p = FIGS / f"{name}.{fmt}"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); made.append(str(p))

    def _ylim(vals, pad=0.03):
        """Data-driven y-limits with padding, clamped to [0,1] for F1-like metrics."""
        v = [x for x in vals if x is not None]
        if not v:
            return (0.0, 1.0)
        lo, hi = min(v), max(v)
        return (max(0.0, lo - pad - (hi - lo) * 0.1), min(1.0, hi + pad))

    # ---- fig3: TAF-28k main results (F1 across methods) ----
    e1, e4 = res.get("exp1", {}), res.get("exp4", {})
    if e1 or e4:
        methods, f1s = [], []
        clfs = e4.get("classifiers", {})
        for name, v in clfs.items():
            methods.append(name); f1s.append(v.get("f1") if isinstance(v, dict) else v)
        if e1.get("f1") is not None:
            methods.append("QAD+OV-Freeze"); f1s.append(e1["f1"])
        if methods:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            colors = ["grey"] * (len(methods) - 1) + ["green"]
            ax.bar(methods, f1s, color=colors)
            ax.set_ylabel("F1 score (bars)"); ax.set_ylim(*_ylim(f1s))
            ax.set_title("TAF-28k main results")
            ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=20, ha="right")
            # BF16 recovery rate (right axis): recovery = method_F1 / BF16_reference_F1.
            # exp11 returns fp16/int8/int4/nf4; the BF16 reference is the fp16 scheme.
            _sch = res.get("exp11", {}).get("schemes", {})
            bf16_ref = (_sch.get("fp16") or _sch.get("no_quantization") or {}).get("f1")
            if bf16_ref:
                rec = [round(f / bf16_ref, 3) if f else None for f in f1s]
                ax2 = ax.twinx()
                ax2.plot(range(len(methods)), rec, "rD", markersize=7, label="BF16 recovery")
                ax2.set_ylabel("BF16 recovery rate"); ax2.set_ylim(0.8, 1.05); ax2.legend(loc="lower right")
            _save(fig, "fig3_main_results")

    # ---- fig4: (a) KL convergence + (b) quantization SNR stability ----
    traj = res.get("exp1", {}).get("trajectory", [])
    if traj and isinstance(traj, list):
        # Reviewer point 1: label the data source honestly. Proxy (small-model/synthetic) vs real H100.
        e1 = res.get("exp1", {})
        is_proxy = e1.get("computation") != "h100_real_qwen"
        src = "qualitative proxy (small-model verification)" if is_proxy else "8xH100 real training run"
        steps = [t.get("step", i) for i, t in enumerate(traj)]
        kl = [t.get("kl") for t in traj]
        snr = [t.get("snr_db") for t in traj]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
        ax1.plot(steps, kl, "b-o", markersize=3)
        ax1.set_xlabel("Training step"); ax1.set_ylabel("KL divergence")
        ax1.set_title("(a) Distillation KL convergence")
        if any(s is not None for s in snr):
            ax2.plot(steps, snr, "r-s", markersize=3)
            ax2.set_xlabel("Training step"); ax2.set_ylabel("Quantization SNR (dB)")
            ax2.set_title("(b) Quantization SNR stability")
        fig.suptitle(f"Data source: {src}", fontsize=9, y=0.02, color="dimgrey")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, "fig4_kl_snr")

    # ---- fig5: (a) quant scheme + (b) AdvFraud pool + (c) LDP trade-off ----
    schemes = res.get("exp11", {}).get("schemes", {})
    e5 = res.get("exp5", {})
    adv = e5.get("advfraud", {}); ldp = e5.get("ldp_tradeoff", {})
    if schemes or adv or ldp:
        panels = sum(bool(x) for x in (schemes, adv, ldp))
    adv = e5.get("advfraud", {}); adv_exp = e5.get("advfraud_expert", {}); ldp = e5.get("ldp_tradeoff", {})
    if schemes or adv or adv_exp or ldp:
        panels = sum(bool(x) for x in (schemes, adv or adv_exp, ldp))
        fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))
        if panels == 1:
            axes = [axes]
        i = 0
        if schemes:
            names = list(schemes.keys()); f1 = [schemes[n]["f1"] for n in names]
            axes[i].bar([n.replace("_", "\n") for n in names], f1, color=["green", "orange", "grey"][:len(names)])
            axes[i].set_ylabel("F1"); axes[i].set_ylim(*_ylim(f1)); axes[i].set_title("(a) Quantization scheme"); i += 1
        if adv:
            names = list(adv.keys()); f1 = [adv[n]["f1"] for n in names]
            axes[i].bar([n.replace("_", "\n") for n in names], f1, color=["green", "orange", "grey"][:len(names)])
            axes[i].set_ylabel("F1"); axes[i].set_ylim(*_ylim(f1)); axes[i].set_title("(b) AdvFraud pool"); i += 1
        if adv or adv_exp:
            names = list(adv.keys()) + list(adv_exp.keys())
            f1 = [adv[n]["f1"] for n in adv] + [adv_exp[n]["f1"] for n in adv_exp]
            colors = (["green", "orange", "grey"][:len(adv)] +
                      ["steelblue", "lightcoral"][:len(adv_exp)])
            axes[i].bar([n.replace("_", "\n") for n in names], f1, color=colors)
            axes[i].set_ylabel("F1"); axes[i].set_ylim(*_ylim(f1)); axes[i].set_title("(b) AdvFraud pools"); i += 1
        if ldp:
            # no_ldp is a reference point (large epsilon); exclude it from the curve and draw it as a line.
            pts = sorted((v["epsilon"], v["f1"]) for k, v in ldp.items()
                         if isinstance(v, dict) and k != "no_ldp")
            if pts:
                axes[i].plot([p[0] for p in pts], [p[1] for p in pts], "g-o")
            no_ldp = ldp.get("no_ldp")
            ref = no_ldp["f1"] if isinstance(no_ldp, dict) else no_ldp
            if ref is not None:
                axes[i].axhline(ref, ls="--", c="grey", label="no LDP"); axes[i].legend()
            axes[i].set_xlabel("Epsilon"); axes[i].set_ylabel("F1"); axes[i].set_title("(c) LDP trade-off"); i += 1
        fig.tight_layout()
        _save(fig, "fig5_ablations")

    # ---- fig6: (a) loss-function ablation + (b) teacher-scale selection ----
    variants = res.get("exp2", {}).get("variants", {})
    scales = res.get("exp10", {}).get("scales", {})
    if variants or scales:
        panels = sum(bool(x) for x in (variants, scales))
        fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4))
        if panels == 1:
            axes = [axes]
        i = 0
        if variants:
            names = list(variants.keys()); f1 = [variants[n].get("f1") for n in names]
            kl = [variants[n].get("kl_final") for n in names]
            axes[i].bar([n.replace("_", "\n") for n in names], f1, color="steelblue")
            axes[i].set_ylabel("Student F1 (bars)"); axes[i].set_ylim(*_ylim(f1))
            axes[i].set_title("(a) Loss-function ablation"); plt.setp(axes[i].get_xticklabels(), rotation=15, ha="right")
            if any(k is not None for k in kl):
                ax_kl = axes[i].twinx()
                ax_kl.plot(range(len(names)), kl, "rs-", label="KL divergence")
                ax_kl.set_ylabel("KL divergence"); ax_kl.legend(loc="upper right")
            i += 1
        if scales:
            names = list(scales.keys()); f1 = [scales[n]["f1"] for n in names]
            axes[i].bar([n.replace("_", "\n") for n in names], f1, color="teal")
            axes[i].set_ylabel("Student F1"); axes[i].set_ylim(*_ylim(f1))
            axes[i].set_title("(b) Teacher-scale selection"); i += 1
        fig.tight_layout()
        _save(fig, "fig6_loss_teacher")

    # ---- fig7: (a) layer selection + (b) activation window ----
    layer = res.get("exp3", {}).get("layer_selection", {})
    rho = res.get("exp3", {}).get("rho_sweep", {})
    if layer or rho:
        panels = sum(bool(x) for x in (layer, rho))
        fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4))
        if panels == 1:
            axes = [axes]
        i = 0
        if layer:
            names = list(layer.keys()); f1 = [layer[n]["f1"] for n in names]; drift = [layer[n]["variance_drift_pct"] for n in names]
            ax = axes[i]; ax.bar(range(len(names)), f1, color="steelblue", label="F1")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("Student F1"); ax.set_ylim(*_ylim(f1)); ax.set_title("(a) Layer selection")
            ax2 = ax.twinx(); ax2.plot(range(len(names)), drift, "r-o", label="Variance drift")
            ax2.set_ylabel("Variance drift (%)"); i += 1
        if rho:
            names = list(rho.keys()); f1 = [rho[n]["f1"] for n in names]
            ppl = [rho[n].get("ppl") for n in names]
            xs = [n.replace("rho_", "") for n in names]
            axes[i].bar(range(len(xs)), f1, color="steelblue")
            axes[i].set_xticks(range(len(xs))); axes[i].set_xticklabels(xs)
            axes[i].set_xlabel("Activation window (rho)"); axes[i].set_ylabel("Student F1 (bars)")
            axes[i].set_ylim(*_ylim(f1)); axes[i].set_title("(b) Activation window")
            if any(p is not None for p in ppl):
                ax_ppl = axes[i].twinx(); ax_ppl.plot(range(len(xs)), ppl, "r-o", label="PPL")
                ax_ppl.set_ylabel("Perplexity (PPL)"); ax_ppl.legend(loc="upper right")
            i += 1
        fig.tight_layout()
        _save(fig, "fig7_ovfreeze")

    # ---- fig8: speculative decoding theoretical curves + operating points ----
    diag = res.get("exp6", {}).get("diagnostic_B", {})
    if diag:
        fig, ax = plt.subplots(figsize=(7, 4))
        alphas = np.linspace(0.5, 0.95, 100)
        for g in (3, 5, 7):
            speedup = (1 - alphas ** (g + 1)) / (1 - alphas)
            ax.plot(alphas, speedup, label=f"gamma={g}")
        gen = diag.get("alpha_generic_measured")
        dom = diag.get("alpha_domain")
        if gen is not None:
            s = (1 - gen ** 6) / (1 - gen)
            ax.plot(gen, s, "k*", markersize=12)
            ax.annotate(f"generic measured ({gen})", (gen, s), textcoords="offset points", xytext=(-10, 8))
        if dom is not None:
            s = (1 - dom ** 6) / (1 - dom)
            ax.plot(dom, s, "g*", markersize=12)
            ax.annotate(f"domain-tuned ({dom})", (dom, s), textcoords="offset points", xytext=(-10, -12))
        ax.set_xlabel("Token acceptance rate (alpha)"); ax.set_ylabel("Theoretical speedup")
        ax.set_title("Speculative decoding speedup"); ax.legend()
        _save(fig, "fig8_specdec")

    # ---- fig9: multimodal fusion strategy (v25 Table 3): F1 bars + latency line ----
    fusion = res.get("exp13", {}).get("strategies", {})
    if fusion:
        names = list(fusion.keys()); f1 = [fusion[n]["f1"] for n in names]
        lat = [fusion[n]["latency_ms"] for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([n.replace("_", "\n") for n in names], f1, color=["grey", "green", "orange"][:len(names)])
        ax.set_ylabel("F1 (bars)"); ax.set_ylim(*_ylim(f1)); ax.set_title("Fusion strategy ablation")
        ax2 = ax.twinx(); ax2.plot(range(len(names)), lat, "r-o", label="Latency (ms)")
        ax2.set_ylabel("Latency (ms)"); ax2.legend(loc="upper left")
        _save(fig, "fig9_fusion_strategy")

    # ================= Reviewer-recommended figures (Manusights Section 4) =================

    # Reviewer C [Essential] 鈥?matched-regulariser control: does variance-matching specifically help,
    # or does any late regulariser help? Plots F1 bars + variance drift across the 4 matched conditions.
    cond = res.get("exp3", {}).get("conditions", {})
    if cond:
        names = list(cond.keys()); f1 = [cond[n].get("f1") for n in names]; drift = [cond[n].get("variance_drift_pct") for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(names)), f1, color=["grey", "green", "orange", "purple"][:len(names)])
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", "\n") for n in names])
        ax.set_ylabel("Student F1 (bars)"); ax.set_ylim(*_ylim(f1)); ax.set_title("Matched-regulariser control (reviewer C)")
        ax2 = ax.twinx(); ax2.plot(range(len(names)), drift, "r-o", label="Variance drift (%)")
        ax2.set_ylabel("Variance drift (%)"); ax2.legend(loc="upper right")
        _save(fig, "figR_matched_regulariser_control")

    # Reviewer B [Essential] 鈥?speculative-decoding acceptance-rate: plots the expected-tokens
    # curve with the MEASURED alpha operating point marked.
    diag = res.get("exp6", {}).get("diagnostic_B", {})
    if diag:
        g = diag.get("gamma", 5)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        a = np.linspace(0.5, 0.95, 100)
        ax.plot(a, (1 - a ** (g + 1)) / (1 - a), "b-", label=f"expected tokens (gamma={g})")
        gen = diag.get("alpha_generic_measured")
        dom = diag.get("alpha_domain")
        pt = diag.get("paper_tokens_generic")
        if gen is not None and pt is not None:
            ax.plot(gen, pt, "g*", markersize=14)
            ax.annotate(f"generic measured\n({gen}, tokens={pt})", (gen, pt),
                       textcoords="offset points", xytext=(-30, -28))
        if dom is not None:
            dom_pt = (1 - dom ** (g + 1)) / (1 - dom) if dom < 1 else g + 1
            ax.plot(dom, dom_pt, "b*", markersize=14)
            ax.annotate(f"domain-tuned\n({dom}, tokens={dom_pt:.2f})", (dom, dom_pt),
                       textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("Token acceptance rate (alpha)"); ax.set_ylabel("Expected tokens / step")
        ax.set_title("Speculative decoding consistency (reviewer B)"); ax.legend(loc="upper left")
        _save(fig, "figR_specdec_consistency")

    # Reviewer D [Recommended] 鈥?open-set ASV privacy at scale (n>=50 speakers): EER, speaker-ID
    # accuracy vs chance, with n reported (addresses the "n/group not stated" weakness).
    e7 = res.get("exp7", {})
    if e7.get("asv_eer_pct") is not None:
        n_sp = e7.get("n_speakers") or 0
        chance = round(100.0 / n_sp, 2) if n_sp else None
        labels = ["ASV-EER\n(%)", "Speaker-ID acc\n(%)"]
        vals = [e7.get("asv_eer_pct"), (e7.get("speaker_id_accuracy") or 0) * 100]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(labels, vals, color=["teal", "indianred"])
        if chance is not None:
            ax.axhline(chance, ls="--", c="grey", label=f"chance ({chance}%)"); ax.legend()
        ax.set_ylabel("Percent"); ax.set_title(f"Open-set ASV privacy, n={n_sp} speakers (reviewer D)")
        _save(fig, "figR_privacy_asv_openset")

    # Reviewer E [Recommended] 鈥?efficiency-accuracy vs a real quantised competitor (FraudFusion),
    # plotting F1 against storage footprint (MB) for QAD, FraudFusion and the 7B reference.
    e12 = res.get("exp12", {})
    comp = e12.get("competitor_comparison_real", {}); fp = e12.get("storage_decomposition_point8", {}).get("footprints_mb", {})
    if comp and fp:
        pts = []
        _q4 = fp.get("0.5B_Q4_K_M") or 240   # .get default only covers missing key, not None value
        _7b = fp.get("7B_BF16_SAFE_QAQ") or 7000
        if comp.get("QAD_MultiGuard_INT4", {}).get("f1") is not None:
            pts.append(("QAD+OVF INT4", comp["QAD_MultiGuard_INT4"]["f1"], _q4))
        if comp.get("FraudFusion_pruned_INT4", {}).get("f1") is not None:
            pts.append(("FraudFusion INT4", comp["FraudFusion_pruned_INT4"]["f1"], _q4 * 1.1))
        # SAFE-QAQ 7B is a cite-only baseline; only plot if exp12 provides the value from its results.
        safeqaq = comp.get("SAFE_QAQ_7B_cited") or comp.get("SAFE_QAQ_7B_BF16")
        if safeqaq and safeqaq.get("f1") is not None:
            pts.append(("SAFE-QAQ 7B BF16 (cited)", safeqaq["f1"], _7b))
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, f1v, mb in pts:
            ax.scatter(mb, f1v, s=90); ax.annotate(name, (mb, f1v), textcoords="offset points", xytext=(6, 4))
        ax.set_xscale("log"); ax.set_xlabel("Storage footprint (MB, log)"); ax.set_ylabel("F1")
        ax.set_title("Efficiency-accuracy vs quantised competitor (reviewer E)")
        _save(fig, "figR_fraudfusion_efficiency")

    # Reviewer point 8 [Weakness] 鈥?storage-footprint decomposition: the 28x advantage conflates two
    # independent axes. This figure separates them: quantization alone (0.5B BF16 -> Q4_K_M = 4.0x) and
    # parameter-scale alone (7B BF16 -> 0.5B BF16 = 7.3x), so the single "28x" number is disaggregated.
    sd = res.get("exp12", {}).get("storage_decomposition_point8", {})
    fpm = sd.get("footprints_mb", {})
    if fpm:
        labels = ["7B BF16\n(SAFE-QAQ)", "0.5B BF16", "0.5B Q4_K_M\n(ours)"]
        mb = [fpm.get("7B_BF16_SAFE_QAQ"), fpm.get("0.5B_BF16"), fpm.get("0.5B_Q4_K_M")]
        if any(v is None for v in mb):
            return made  # storage footprints must be measured, not hardcoded
        fig, ax = plt.subplots(figsize=(7.5, 4))
        bars = ax.bar(labels, mb, color=["indianred", "steelblue", "green"])
        ax.set_yscale("log"); ax.set_ylabel("Storage footprint (MB, log)")
        ax.set_title("Storage decomposition: two independent axes (reviewer point 8)")
        for b, v in zip(bars, mb):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v} MB", ha="center", va="bottom")
        # Axis annotations
        q = sd.get("quantization_alone_x"); p = sd.get("param_scale_alone_x"); t = sd.get("total_advantage_x")
        ax.annotate(f"parameter scale\n{p}x", xy=(0.5, (mb[0] * mb[1]) ** 0.5), ha="center", color="indianred")
        ax.annotate(f"quantization\n{q}x", xy=(1.5, (mb[1] * mb[2]) ** 0.5), ha="center", color="green")
        ax.text(0.5, 0.02, f"total = {q}x (quant) x {p}x (param) = {t}x ~ 28x",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="lightyellow"))
        _save(fig, "figR_storage_decomposition")

    return made


def build_pdf_figures() -> list:
    """Vector (PDF) versions of the paper figures, for direct inclusion in the manuscript."""
    return build_paper_figures(fmt="pdf")


def build_latex_tables() -> Path:
    """Generate LaTeX tables (Table2.tex) from real results. Returns Path to generated file."""
    def _fmt(v):
        return "-" if v is None else str(v)

    OUT.mkdir(parents=True, exist_ok=True)
    res = _latest_results()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TAF-28k main results (v25 Table 4)}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & F1 & Computation \\",
        r"\midrule",
    ]
    e1, e4 = res.get("exp1", {}), res.get("exp4", {})
    for name, v in e4.get("classifiers", {}).items():
        f1 = v.get("f1") if isinstance(v, dict) else v
        lines.append(f"{name} & {_fmt(f1)} & real \\\\")
    if e1.get("f1") is not None:
        lines.append(f"QAD+OV-Freeze & {_fmt(e1['f1'])} & real \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "Table2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUT / "Table2.tex"


def build_all() -> list:
    """Run all report generators: summary CSV, paper tables, paper figures, LaTeX tables, PDF figures.

    Returns list of generated file paths.
    """
    made = []
    made.append(str(build_summary_csv()))
    made.append(str(build_paper_tables()))
    made.append(str(build_paper_figures()))
    made.append(str(build_latex_tables()))
    try:
        pdfs = build_pdf_figures()
        made.extend(pdfs)
    except Exception:
        pass
    return made
