"""experiments/runner.py — Unified Experiment Orchestrator

Usage:
    python -m experiments.runner --smoke          # Quick verification (small model path)
    python -m experiments.runner --paper          # Paper-grade (real Qwen + H100)
    python -m experiments.runner --exp 1,3,6      # Specific experiments
    python -m experiments.runner --check          # Hardware check
    python -m experiments.runner --storage-check  # Storage mount check
    python -m experiments.runner --benchmark      # Benchmark only
    python -m experiments.runner --resume         # Skip completed experiments
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("runner")

# Experiment registry: (module_name, short_name, description)
EXPERIMENTS = [
    ("exp1_qad_production", "exp1", "QAD Production (teacher->student KL distillation with OV-Freeze)"),
    ("exp2_qad_loss_ablation", "exp2", "Loss Ablation (pure-KL / three-term / logits-MSE)"),
    ("exp3_ov_freeze_control", "exp3", "OV-Freeze Control (4-condition + rho sweep)"),
    ("exp4_baseline_comparison", "exp4", "Baseline Comparison (GBM / RF / Logistic Regression)"),
    ("exp5_cross_dataset", "exp5", "Cross-Dataset (TAF-28k / ChiFraud / AdvFraud-3k / LDP)"),
    ("exp6_speculative_decoding", "exp6", "Speculative Decoding (alpha + diagnostic B)"),
    ("exp7_privacy_verification", "exp7", "Privacy Verification (ASV-EER + GLO attack)"),
    ("exp8_latency_benchmark", "exp8", "Latency Benchmark (end-to-end wall-clock)"),
    ("exp9_cot_ablation", "exp9", "CoT Ablation (chain-of-thought vs direct)"),
    ("exp10_teacher_scale", "exp10", "Teacher Scale (0.5B/1.5B/7B)"),
    ("exp11_quantization_scheme", "exp11", "Quantization Scheme (NVFP4+Q4KM vs INT4)"),
    ("exp12_fraudfusion_baseline", "exp12", "FraudFusion Baseline (quantized competitor + storage decomposition)"),
    ("exp13_fusion_strategy", "exp13", "Fusion Strategy (multimodal fusion ablation)"),
    ("exp14_gguf_comparison", "exp14", "Multi-model same-data comparison (BF16 transformers vs Q4_K_M GGUF llama.cpp)"),
]

# Map short names to full module names (for backward compatibility and validation)
_SHORT_TO_FULL = {short: mod for mod, short, _ in EXPERIMENTS}


def _import_exp(modname: str):
    """Import an experiment module by name."""
    import importlib
    return importlib.import_module(f"experiments.{modname}")


def run_experiment(modname: str, short: str, config: dict) -> dict:
    """Run a single experiment, save result, return result dict."""
    from realeval.io import save_results
    from realeval.runlog import log_experiment_start, log_experiment_end
    log_experiment_start(short, config)
    mod = _import_exp(modname)
    result = mod.run(config)
    save_results(short, result)
    log_experiment_end(short, result)
    return result


def run_all(config: dict, selected: list[str] | None = None, resume: bool = False) -> dict:
    """Run all (or selected) experiments. Returns {exp_short: result}."""
    from realeval.io import load_config, RESULTS
    import json

    cfg = load_config() if config is None else config
    results = {}
    for modname, short, desc in EXPERIMENTS:
        if selected and short not in selected:
            continue
        if resume:
            existing = list(RESULTS.glob(f"{short}_*.json"))
            if existing:
                results[short] = json.loads(max(existing).read_text(encoding="utf-8"))
                logger.info("Skipping %s (already completed)", short)
                continue
        logger.info("Running %s: %s", short, desc)
        try:
            r = run_experiment(modname, short, cfg)
            results[short] = r
        except Exception as e:
            logger.error("Experiment %s failed: %s", short, e)
            results[short] = {"experiment": short, "error": str(e), "computation": "failed"}
    return results


def main():
    parser = argparse.ArgumentParser(description="QAD-MultiGuard Experiment Runner")
    # Positional experiments argument (backward compatible with B)
    parser.add_argument("experiments", nargs="?", default=None,
                        help="Comma-separated experiment names (e.g. exp1,exp4,exp7) or 'all'")
    parser.add_argument("--smoke", action="store_true", help="Quick verification (small model path)")
    parser.add_argument("--paper", action="store_true", help="Paper-grade (real Qwen + H100)")
    parser.add_argument("--exp", type=str, default=None, help="Comma-separated experiment numbers (e.g. 1,3,6)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--check", action="store_true", help="Hardware check")
    parser.add_argument("--storage-check", action="store_true", help="Storage mount check")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark only")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--report", action="store_true", help="Generate paper tables/figures from existing results (no experiments run)")
    args = parser.parse_args()

    # --- Standalone checks ---
    if args.check:
        from realeval.hwenv import check
        ok, issues = check(strict=True)
        if ok:
            print("Hardware check PASSED (paper-grade ready)")
        else:
            print("Hardware check ISSUES:")
            for i in issues:
                print(f"  - {i}")
        return

    if args.storage_check:
        from realeval.paths import storage_report, list_local_models
        rep = storage_report()
        print("Storage Report:")
        for k, v in rep.items():
            print(f"  {k}: {v}")
        lm = list_local_models()
        print(f"Local models ({lm['count']}):")
        for m in lm["found"]:
            print(f"  - {m}")
        return

    if args.report:
        from realeval.report import build_all
        logger.info("Generating paper tables and figures from existing results...")
        made = build_all()
        logger.info("Generated %d artifacts", len(made))
        for p in made:
            print(f"  {p}")
        return

    if args.benchmark:
        from realeval.benchmark import benchmark, summary
        import torch
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))
        r = benchmark(model, torch.randn(64), warmup=10, repeat=100, batch_sizes=(1, 16, 64))
        s = summary(r)
        print("Benchmark summary:", s)
        return

    # --- Load config ---
    from realeval import io as io_mod, validation, envreport
    config = io_mod.load_config(args.config)
    if args.smoke:
        config["_smoke"] = True
        logger.info("SMOKE MODE: using small model verification path")
    if args.paper:
        config["_paper"] = True
        logger.info("PAPER MODE: using real Qwen + H100 backend")

    # Validate config
    try:
        validation.validate_config(config)
    except validation.ValidationError as e:
        logger.error("Config validation failed: %s", e)
        sys.exit(1)

    # Environment report + audit log
    envreport.write_report()
    from realeval.audit import log_environment
    log_environment(config)

    # Resolve experiment selection
    if args.exp is not None:
        raw = args.exp
    elif args.experiments is not None:
        raw = args.experiments
    else:
        raw = "all"

    if raw == "all":
        selected = [short for _, short, _ in EXPERIMENTS]
    else:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        digits = []
        for p in parts:
            if p.isdigit():
                digits.append(p)
            elif p.startswith("exp") and p[3:].isdigit():
                digits.append(p[3:])
            else:
                logger.error("Invalid experiment identifier: %s", p)
                sys.exit(1)
        from realeval.validation import validate_experiment_selection
        selected = validate_experiment_selection(
            ",".join(digits),
            [(short, desc) for _, short, desc in EXPERIMENTS])

    # Run experiments
    results = run_all(config, selected, resume=args.resume)

    logger.info("Experiments completed: %s", list(results.keys()))
    logger.info("Results saved to outputs/results/. Run 'python -m experiments.runner --report' to generate tables/figures.")


if __name__ == "__main__":
    main()
