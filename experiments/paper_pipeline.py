"""paper_pipeline.py — one-command H100 paper-validation pipeline.

Orchestrates the full flow and writes the paper deliverables to results/:

    CUDA check -> GPU detect -> env report -> model load -> benchmark -> metrics -> save

Outputs (results/):
    metrics.json      aggregated real experiment metrics (accuracy/F1/FPR per experiment group)
    latency.csv       per-batch-size latency (p50/p90/p99) on the real device
    throughput.csv    tokens/samples per second per batch size
    memory.csv        peak/used device memory per batch size
    paper_table.md    ready-to-paste paper Table (Main / Ablation / Efficiency / Robustness)

Run:  bash run_h100.sh   (wraps `python -m experiments.paper_pipeline --paper`)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("paper_pipeline")

RESULTS = Path(__file__).resolve().parent.parent / "outputs" / "results"

# Paper experiment groups -> the underlying experiment short names that produce their metrics.
PAPER_GROUPS = {
    "01_baseline":     ["exp4"],           # BF16 teacher + classical baselines
    "02_quantization": ["exp11"],          # INT4 / NVFP4 scheme comparison
    "03_QAD":          ["exp1", "exp2"],   # Pure-KL homologous distillation
    "04_OV-Freeze":    ["exp3"],           # OV-Freeze ablation + matched-regulariser control
    "05_latency":      ["exp8", "exp6"],   # H100 latency/throughput + speculative-decoding speedup
    "06_robustness":   ["exp5", "exp7"],   # OOD/cross-dataset + adversarial/privacy
}


def _cuda_check():
    """CUDA -> GPU detect -> env report. Returns (has_cuda, env_dict)."""
    logger.info("[1/7] CUDA check ...")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception as e:
        logger.error("torch import failed: %s", e)
        return False, {}
    logger.info("      CUDA available: %s", has_cuda)

    logger.info("[2/7] GPU detect ...")
    n_gpu = torch.cuda.device_count() if has_cuda else 0
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        logger.info("      GPU %d: %s (%.1f GB)", i, p.name, p.total_memory / 1e9)
    if has_cuda and not any("H100" in torch.cuda.get_device_properties(i).name for i in range(n_gpu)):
        logger.warning("      No H100 detected — pipeline runs but numbers are not H100-grade.")

    logger.info("[3/7] Environment report ...")
    try:
        from realeval import envreport
        env = envreport.collect()
    except Exception as e:
        logger.warning("      envreport failed: %s", e); env = {}
    return has_cuda, env


def _apply_h100_optims(config: dict, has_cuda: bool) -> dict:
    """Enable H100-oriented settings: BF16, FlashAttention-2, (optional) torch.compile, NCCL/DDP."""
    hw = config.setdefault("hardware", {})
    if has_cuda:
        hw.setdefault("use_flash_attn", True)     # FlashAttention-2
        hw.setdefault("bf16", True)               # BF16 compute
        hw.setdefault("use_torch_compile", hw.get("use_torch_compile", False))
        import torch
        if torch.cuda.device_count() > 1:
            hw["ddp"] = True                      # NCCL multi-GPU
            logger.info("      Multi-GPU (%d) -> DDP/NCCL enabled", torch.cuda.device_count())
        # NVFP4/QAD path is driven by config['models'] + quantize=int4/nf4 (already wired).
        logger.info("      H100 optims: BF16=%s FlashAttn=%s compile=%s ddp=%s",
                    hw.get("bf16"), hw.get("use_flash_attn"), hw.get("use_torch_compile"), hw.get("ddp"))
    return config


def _run_experiments(config: dict, smoke: bool) -> dict:
    """[4/7] model load + [5/7] benchmark: run each experiment group, collect real metrics."""
    logger.info("[4/7] Model load + [5/7] Benchmark (running experiment groups) ...")
    from experiments.runner import _import_exp, _SHORT_TO_FULL
    from realeval.io import save_results
    all_results = {}
    for group, shorts in PAPER_GROUPS.items():
        for short in shorts:
            try:
                mod = _import_exp(_SHORT_TO_FULL[short])
                res = mod.run(config)
                save_results(short, res)
                all_results[short] = res
                logger.info("      %s/%s -> %s", group, short, res.get("computation", "?"))
            except Exception as e:
                logger.error("      %s/%s failed: %s", group, short, e, exc_info=True)
                all_results[short] = {"error": str(e)}
    return all_results


def _device_benchmark(config: dict, has_cuda: bool):
    """Real device latency/throughput/memory benchmark -> latency.csv / throughput.csv / memory.csv."""
    if not has_cuda:
        logger.info("      No CUDA: skipping device benchmark CSVs (run on H100 for real numbers).")
        return None
    try:
        import torch
        from realeval import models, benchmark
        model, tok = models.load_causal_lm(config["models"]["teacher"], quantize="int4", bf16=True)
        sample_ids = tok("Detect fraud in this message.", return_tensors="pt").input_ids.squeeze(0)
        res = benchmark.benchmark_forward(model, sample_ids, warmup=10, repeat=100,
                                          batch_sizes=(1, 8, 32))
        benchmark._write_csv(res)   # writes latency/throughput/memory CSVs
        return benchmark.benchmark_summary(res)
    except Exception as e:
        logger.error("      device benchmark failed: %s", e)
        return None


def _extract(short, all_results):
    """Pull headline metrics from each experiment's own result shape."""
    r = all_results.get(short, {})
    if short == "exp1":
        return {"F1": r.get("f1")}
    if short == "exp2":
        v = r.get("variants", {})
        return {f"kl_final[{k}]": x.get("kl_final") for k, x in v.items()}
    if short == "exp3":
        c = r.get("conditions", {})
        return {f"drift[{k}]": x.get("variance_drift_pct") for k, x in c.items()}
    if short == "exp4":
        c = r.get("classifiers", {})
        return {f"F1[{k}]": x.get("f1") for k, x in c.items()}
    if short == "exp5":
        taf = r.get("taf28k", {}).get("f1")
        chi = r.get("chifraud", {}).get("f1")
        adv = r.get("advfraud", {}).get("full_pool", {}).get("f1") if isinstance(r.get("advfraud"), dict) else None
        cross_tc = r.get("cross_taf_on_chifraud", {}).get("f1")
        cross_ct = r.get("cross_chifraud_on_taf", {}).get("f1")
        out = {}
        if taf is not None: out["taf28k"] = taf
        if chi is not None: out["chifraud"] = chi
        if adv is not None: out["advfraud"] = adv
        if cross_tc is not None: out["cross_taf->chi"] = cross_tc
        if cross_ct is not None: out["cross_chi->taf"] = cross_ct
        return out or {"cross_taf->chi": None, "cross_chi->taf": None}
    if short == "exp6":
        d = r.get("diagnostic_B", {})
        hm = d.get("h100_measured", {})
        out = {}
        if hm.get("generic") is not None: out["alpha_generic"] = hm["generic"]
        if hm.get("domain") is not None: out["alpha_domain"] = hm["domain"]
        return out or {"alpha_generic": None}
    if short == "exp7":
        return {"speaker_id_acc": r.get("speaker_id_accuracy"), "asv_eer_pct": r.get("asv_eer_pct")}
    if short == "exp8":
        return {f"lat_ms[{k}]": v for k, v in r.get("latencies", {}).items()}
    if short == "exp9":
        wc = r.get("with_cot", {}); wo = r.get("without_cot", {})
        return {"cot_f1": wc.get("f1"), "direct_f1": wo.get("f1"),
                "cot_fpr": wc.get("fpr"), "direct_fpr": wo.get("fpr")}
    if short == "exp10":
        return {f"F1[{k}]": x.get("f1") for k, x in r.get("scales", {}).items()}
    if short == "exp11":
        return {f"F1[{k}]": x.get("f1") for k, x in r.get("schemes", {}).items()}
    if short == "exp12":
        comp = r.get("competitor_comparison_real", {})
        storage = r.get("storage_decomposition_point8", {})
        out = {f"F1[{k}]": x.get("f1") for k, x in comp.items()}
        for k, v in storage.get("footprints_mb", {}).items():
            out[f"fp[{k}]"] = v
        if storage.get("total_advantage_x") is not None:
            out["total_advantage_x"] = storage["total_advantage_x"]
        return out
    if short == "exp13":
        return {f"F1[{k}]": x.get("f1") for k, x in r.get("strategies", {}).items()}
    if short == "exp14":
        models = r.get("models", {})
        return {f"F1[{k}]": x.get("f1") for k, x in models.items() if x.get("f1") is not None}
    if "f1" in r:
        return {"F1": r["f1"]}
    return {}


def _aggregate_and_save(all_results: dict, bench_summary, env: dict):
    """[6/7] metrics stats + [7/7] save: metrics.json, latency/throughput/memory.csv, LaTeX + md tables."""
    logger.info("[6/7] Aggregating metrics ...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "paper_tables").mkdir(exist_ok=True)

    metrics = {"env": env, "groups": {}}
    for group, shorts in PAPER_GROUPS.items():
        metrics["groups"][group] = {s: _extract(s, all_results) for s in shorts}
    if bench_summary:
        metrics["benchmark"] = bench_summary
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Separate efficiency CSVs (latency / throughput / memory) from the benchmark summary rows.
    rows = (bench_summary or {}).get("rows", [])
    if rows:
        def _csv(name, cols):
            import csv as _csv_mod
            with open(RESULTS / name, "w", newline="") as f:
                w = _csv_mod.writer(f); w.writerow([c[0] for c in cols])
                for row in rows:
                    w.writerow([row.get(c[1]) for c in cols])
        _csv("latency.csv", [("batch_size", "batch_size"), ("p50_ms", "latency_p50_ms"),
                             ("p90_ms", "latency_p90_ms"), ("p99_ms", "latency_p99_ms")])
        _csv("throughput.csv", [("batch_size", "batch_size"), ("samples_per_sec", "throughput_sps")])
        _csv("memory.csv", [("batch_size", "batch_size"), ("peak_mem_mb", "peak_mem_mb")])

    logger.info("[7/7] Writing paper tables (md + LaTeX) ...")
    # Markdown summary
    md = ["# Paper Tables (auto-generated from real results)", ""]
    for group, shorts in PAPER_GROUPS.items():
        md.append(f"## {group}")
        for s in shorts:
            ex = _extract(s, all_results)
            comp = all_results.get(s, {}).get("computation", "-")
            md.append(f"- **{s}** ({comp}): " + ", ".join(f"{k}={v}" for k, v in ex.items()))
        md.append("")
    (RESULTS / "paper_table.md").write_text("\n".join(md) + "\n")

    # LaTeX tables: table1_main, table2_ablation, table3_efficiency
    def _latex(fname, title, header, body_rows):
        L = ["\\begin{table}[t]", "\\centering", f"\\caption{{{title}}}",
             "\\begin{tabular}{" + "l" * len(header) + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
        L += [" & ".join(str(c) for c in r) + " \\\\" for r in body_rows]
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (RESULTS / "paper_tables" / fname).write_text("\n".join(L) + "\n")

    main_rows = [[s, all_results.get(s, {}).get("computation", "-"),
                  _extract(s, all_results).get("F1", "-")] for s in ("exp4", "exp1")]
    _latex("table1_main.tex", "Main Result (F1)", ["Experiment", "Computation", "F1"], main_rows)
    abl_rows = [[k, v] for k, v in _extract("exp3", all_results).items()]
    _latex("table2_ablation.tex", "OV-Freeze Ablation (variance drift \\%)",
           ["Condition", "Drift(\\%)"], abl_rows)
    eff_rows = [[r.get("batch_size"), r.get("latency_p50_ms"), r.get("throughput_sps"),
                 r.get("peak_mem_mb")] for r in rows]
    _latex("table3_efficiency.tex", "Efficiency (H100 benchmark)",
           ["Batch", "p50(ms)", "samp/s", "peak mem(MB)"], eff_rows or [["-", "-", "-", "-"]])

    _print_summary(all_results, bench_summary)
    logger.info("Done. Deliverables in %s/", RESULTS)


def _print_summary(all_results, bench_summary):
    """Final RealEval banner."""
    def g(short, *path, default="n/a"):
        r = all_results.get(short, {})
        for p in path:
            r = r.get(p, {}) if isinstance(r, dict) else {}
        return r if r not in ({}, None) else default
    print("\n=== RealEval-v2 H100 Benchmark ===")
    print(f"QAD:         F1 {all_results.get('exp1', {}).get('f1', 'n/a')}")
    ov = _extract("exp3", all_results)
    print(f"OV-Freeze:   drift {ov.get('drift[ov_freeze_full]', 'n/a')}% (vs no_reg {ov.get('drift[no_reg]', 'n/a')}%)")
    sd = _extract("exp6", all_results)
    print(f"Speculative: alpha generic={sd.get('alpha_generic', 'n/a')} "
          f"domain={sd.get('alpha_domain', 'NOT MEASURED')}")
    print(f"Privacy:     speaker-ID acc {all_results.get('exp7', {}).get('speaker_id_accuracy', 'n/a')}, "
          f"ASV-EER {all_results.get('exp7', {}).get('asv_eer_pct', 'n/a')}%")
    if bench_summary and bench_summary.get("rows"):
        r0 = bench_summary["rows"][0]
        print(f"Latency:     P50 {r0.get('latency_p50_ms', 'n/a')} ms  P99 {r0.get('latency_p99_ms', 'n/a')} ms")
    else:
        print("Latency:     (run --paper on H100 for real latency)")
    print("DONE\n")


def main():
    ap = argparse.ArgumentParser(description="One-command H100 paper-validation pipeline")
    ap.add_argument("--paper", action="store_true", help="Paper-grade run (real Qwen + H100)")
    ap.add_argument("--smoke", action="store_true", help="Sandbox verification (no GPU/weights)")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    smoke = args.smoke or not args.paper

    from realeval.io import load_config
    config = load_config(args.config)
    config["_smoke"] = smoke
    if args.paper:
        config["_paper"] = True

    has_cuda, env = _cuda_check()
    config = _apply_h100_optims(config, has_cuda)
    all_results = _run_experiments(config, smoke)
    bench_summary = _device_benchmark(config, has_cuda) if args.paper else None
    _aggregate_and_save(all_results, bench_summary, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
