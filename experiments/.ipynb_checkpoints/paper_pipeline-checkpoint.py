"""paper_pipeline.py — one-command H100 paper-validation pipeline.
Orchestrates the full flow and writes the paper deliverables to results/.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import importlib.util
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("paper_pipeline")

RESULTS = Path(__file__).resolve().parent.parent / "outputs" / "results"

PAPER_GROUPS = {
    "01_baseline":     ["exp4"],
    "02_quantization": ["exp11"],
    "03_QAD":          ["exp1", "exp2"],
    "04_OV-Freeze":    ["exp3"],
    "05_latency":      ["exp8", "exp6"],
    "06_robustness":   ["exp5", "exp7"],
}

def _cuda_check():
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
    return has_cuda, {}

def _apply_h100_optims(config: dict, has_cuda: bool) -> dict:
    hw = config.setdefault("hardware", {})
    if has_cuda:
        hw.setdefault("use_flash_attn", True)
        hw.setdefault("bf16", True)
        import torch
        if torch.cuda.device_count() > 1:
            hw["ddp"] = True
    return config

def _run_experiments(config: dict, smoke: bool) -> dict:
    logger.info("[4/7] Model load + [5/7] Benchmark (running experiment groups) ...")
    config["_smoke"] = smoke
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
                logger.error("      %s/%s failed: %s", group, short, e)
                all_results[short] = {"error": str(e)}
    return all_results

def _device_benchmark(config: dict, has_cuda: bool):
    if not has_cuda:
        return None
    try:
        import torch
        from realeval import models
        # 使用 importlib 动态隔离加载物理路径，防止内存劫持代理
        spec = importlib.util.spec_from_file_location("raw_benchmark", "/workspace/realeval/benchmark.py")
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)

        model, tok = models.load_causal_lm(config["models"]["teacher"], quantize="int4", bf16=True)
        sample_ids = tok("Detect fraud in this message.", return_tensors="pt").input_ids.squeeze(0)
        
        res = benchmark.benchmark_forward(model, sample_ids, warmup=10, repeat=100, batch_sizes=(1, 8, 32))
        benchmark._write_csv(res)
        return benchmark.benchmark_summary(res)
    except Exception as e:
        logger.error("      device benchmark failed via raw_import: %s", e, exc_info=True)
        return None

def _extract(short, all_results):
    r = all_results.get(short, {})
    if short == "exp1": return {"F1": r.get("f1")}
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
        return {"cross_taf->chi": r.get("cross_taf_on_chifraud", {}).get("f1"),
                "cross_chi->taf": r.get("cross_chifraud_on_taf", {}).get("f1")}
    if short == "exp6":
        sd = r.get("diagnostic_B", {})
        h100m = sd.get("h100_measured", {}) if isinstance(sd, dict) else {}
        return {"alpha_generic": h100m.get("generic", "n/a") if isinstance(h100m, dict) else "n/a"}
    if short == "exp7":
        return {"speaker_id_acc": r.get("speaker_id_accuracy"), "asv_eer_pct": r.get("asv_eer_pct")}
    if short == "exp8":
        return {f"lat_ms[{k}]": v for k, v in r.get("latencies", {}).items()}
    if short == "exp11":
        return {f"F1[{k}]": x.get("f1") for k, x in r.get("schemes", {}).items()}
    if "f1" in r: return {"F1": r["f1"]}
    return {}

def _aggregate_and_save(all_results: dict, bench_summary, env: dict):
    logger.info("[6/7] Aggregating metrics ...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = {"env": env, "groups": {}}
    for group, shorts in PAPER_GROUPS.items():
        metrics["groups"][group] = {s: _extract(s, all_results) for s in shorts}
    if bench_summary:
        metrics["benchmark"] = bench_summary
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    rows = (bench_summary or {}).get("rows", [])
    if rows:
        import csv as _csv_mod
        def _csv(name, cols):
            with open(RESULTS / name, "w", newline="") as f:
                w = _csv_mod.writer(f)
                w.writerow([c[0] for c in cols])
                for row in rows: w.writerow([row.get(c[1]) for c in cols])
        _csv("latency.csv", [("batch_size", "batch_size"), ("p50_ms", "latency_p50_ms"), ("p99_ms", "latency_p99_ms")])
        _csv("throughput.csv", [("batch_size", "batch_size"), ("samples_per_sec", "throughput_sps")])
        _csv_mod = None  # clean up reference
    
    logger.info("[7/7] Writing paper tables ...")
    md = ["# Paper Tables", ""]
    for group, shorts in PAPER_GROUPS.items():
        md.append(f"## {group}")
        for s in shorts:
            ex = _extract(s, all_results)
            md.append(f"- **{s}**: " + ", ".join(f"{k}={v}" for k, v in ex.items()))
    (RESULTS / "paper_table.md").write_text("\n".join(md) + "\n")
    _print_summary(all_results, bench_summary)

def _print_summary(all_results, bench_summary):
    print("\n=== RealEval-v2 H100 Benchmark ===")
    print(f"QAD:         F1 {all_results.get('exp1', {}).get('f1', 'n/a')}")
    ov = _extract("exp3", all_results)
    print(f"OV-Freeze:   drift {ov.get('drift[ov_freeze_full]', 'n/a')}% (vs no_reg {ov.get('drift[no_reg]', 'n/a')}%)")
    sd = all_results.get("exp6", {}).get("diagnostic_B", {})
    h100m = sd.get("h100_measured", {}) if isinstance(sd, dict) else {}
    print(f"Speculative: alpha generic={h100m.get('generic', 'n/a')}")
    print(f"Privacy:     speaker-ID acc {all_results.get('exp7', {}).get('speaker_id_accuracy', 'n/a')}")
    if bench_summary and bench_summary.get("rows"):
        r0 = bench_summary["rows"][0]
        print(f"Latency:     P50 {r0.get('latency_p50_ms', 'n/a')} ms")
    print("DONE\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    smoke = args.smoke or not args.paper

    from realeval.io import load_config
    config = load_config(args.config)
    if args.paper: config["_paper"] = True

    has_cuda, env = _cuda_check()
    config = _apply_h100_optims(config, has_cuda)
    all_results = _run_experiments(config, smoke)
    bench_summary = _device_benchmark(config, has_cuda) if args.paper else None
    _aggregate_and_save(all_results, bench_summary, env)
    return 0

if __name__ == "__main__":
    sys.exit(main()) = _device_benchmark(config, has_cuda) if args.paper else None
    _aggregate_and_save(all_results, bench_summary, env)
    return 0

if __name__ == "__main__":
    sys.exit(main())