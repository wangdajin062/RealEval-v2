"""claim_engine.py — the Research Workflow Engine core.

Claim -> Experiment -> Execution (multi-seed) -> Evidence -> Statistics -> Conclusion.

Each claim (claims/*.yaml) declares a hypothesis, the experiment that tests it, the two conditions to
contrast, and machine-checkable acceptance criteria. The engine runs the experiment across seeds,
extracts the raw dependent-variable samples (Evidence First), summarises them with bootstrap CIs,
evaluates the acceptance criteria, and emits PASS / FAIL / UNSUPPORTED together with a full evidence
trace (which experiment, which seeds, which numbers, why). Nothing is hand-judged; the paper's
conclusions become reproducible functions of the raw evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from realeval import statistics as st
from realeval.runlog import provenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("claim_engine")

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "claims"
OUT = ROOT / "outputs" / "claims"

_SHORT_TO_MOD = {
    "exp1": "exp1_qad_production", "exp2": "exp2_qad_loss_ablation", "exp3": "exp3_ov_freeze_control",
    "exp4": "exp4_baseline_comparison", "exp5": "exp5_cross_dataset", "exp6": "exp6_speculative_decoding",
    "exp7": "exp7_privacy_verification", "exp8": "exp8_latency_benchmark", "exp9": "exp9_cot_ablation",
    "exp10": "exp10_teacher_scale", "exp11": "exp11_quantization_scheme",
    "exp12": "exp12_fraudfusion_baseline", "exp13": "exp13_fusion_strategy", "exp14": "exp14_gguf_comparison",
}


def _dig(d, path):
    for p in path:
        d = d.get(p, {}) if isinstance(d, dict) else {}
    return d


def _run_experiment_seeds(short, base_config, seeds):
    """Execute the experiment once per seed; return the list of raw result dicts (the evidence)."""
    import importlib
    mod = importlib.import_module(f"experiments.{_SHORT_TO_MOD[short]}")
    results = []
    for s in range(seeds):
        cfg = json.loads(json.dumps(base_config))  # deep copy
        cfg["seed"] = 42 + s
        cfg["_smoke"] = base_config.get("_smoke", True)
        results.append(mod.run(cfg))
    return results


def _collect_samples(results, evidence_path, condition, dep_var):
    """Evidence First: pull the per-seed dependent-variable values for one condition."""
    vals = []
    for r in results:
        node = _dig(r, evidence_path)
        cond = node.get(condition, {}) if isinstance(node, dict) else {}
        v = cond.get(dep_var) if isinstance(cond, dict) else None
        if v is not None:
            vals.append(v)
    return vals


def evaluate_claim(claim: dict, base_config: dict) -> dict:
    short = claim["experiment"]
    seeds = int(claim.get("seeds", 1))
    logger.info("[%s] %s", claim["id"], claim["hypothesis"])
    results = _run_experiment_seeds(short, base_config, seeds)

    trace = {"claim": claim["id"], "experiment": short, "seeds": seeds,
             "provenance": provenance(base_config), "evidence": {}, "stats": {}}

    cmp = claim.get("compare")
    ctx = {}  # names available to the acceptance expressions
    if cmp:
        dep = claim["dependent_variable"]
        t_vals = _collect_samples(results, claim["evidence_path"], cmp["treatment"], dep)
        b_vals = _collect_samples(results, claim["evidence_path"], cmp["baseline"], dep)
        trace["evidence"] = {cmp["treatment"]: t_vals, cmp["baseline"]: b_vals}
        t_sum, b_sum = st.summarize(t_vals), st.summarize(b_vals)
        comparison = st.compare(t_vals, b_vals, paired=(len(t_vals) == len(b_vals)))
        trace["stats"] = {"treatment": t_sum, "baseline": b_sum, "comparison": comparison}
        # expose names for acceptance criteria
        ctx["treatment"] = type("N", (), t_sum)
        ctx["baseline"] = type("N", (), b_sum)
        setattr(ctx["treatment"], dep, t_sum["mean"])
        setattr(ctx["baseline"], dep, b_sum["mean"])
        if t_sum["mean"] is not None and b_sum["mean"] is not None:
            ctx["f1_drop"] = b_sum["mean"] - t_sum["mean"]
            ctx["reduction_pct"] = (100 * (b_sum["mean"] - t_sum["mean"]) / b_sum["mean"]
                                    if b_sum["mean"] else None)
        ctx["cohens_d"] = comparison.get("cohens_d")
        ctx["p_value"] = comparison.get("t_p")
    else:
        # single-quantity claim (e.g. speedup from measured alpha)
        node = _dig(results[0], claim["evidence_path"])
        measured_alpha = node.get("alpha") if isinstance(node, dict) else None
        trace["evidence"] = {"node": node}
        ctx["measured_alpha"] = measured_alpha
        gamma = base_config.get("distillation", {}).get("gamma", 5)
        ctx["speedup"] = ((1 - measured_alpha ** (gamma + 1)) / (1 - measured_alpha)
                          if measured_alpha else None)

    # Evaluate acceptance criteria -> PASS / FAIL / UNSUPPORTED
    verdicts = []
    unsupported = False
    for crit in claim.get("acceptance", []):
        try:
            ok = bool(_safe_eval(crit, ctx))
        except _Unsupported:
            ok = None; unsupported = True
        verdicts.append({"criterion": crit, "result": ok})
    if unsupported or any(v["result"] is None for v in verdicts):
        conclusion = "UNSUPPORTED"
    elif all(v["result"] for v in verdicts):
        conclusion = "PASS"
    else:
        conclusion = "FAIL"
    trace["acceptance"] = verdicts
    trace["conclusion"] = conclusion
    logger.info("[%s] -> %s", claim["id"], conclusion)
    return trace


class _Unsupported(Exception):
    pass


def _safe_eval(expr, ctx):
    """Evaluate an acceptance expression against ctx. 'is not None' checks -> UNSUPPORTED when None."""
    import re
    # explicit "X is not None"
    m = re.match(r"\s*([a-zA-Z_.]+)\s+is not None\s*$", expr)
    if m:
        val = ctx.get(m.group(1))
        if val is None:
            raise _Unsupported()
        return True
    # any referenced value that is None -> UNSUPPORTED
    for name in re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", expr):
        base = name.split(".")[0]
        if base in ctx and ctx[base] is None:
            raise _Unsupported()
    safe = {k: v for k, v in ctx.items()}
    return eval(expr, {"__builtins__": {}}, safe)  # noqa: S307 - expressions come from repo-owned YAML


def main():
    ap = argparse.ArgumentParser(description="Claim-driven research workflow engine")
    ap.add_argument("--claim", type=str, default=None, help="Run one claim id (e.g. CLAIM-01)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    from realeval.io import load_config
    base = load_config()
    base["_smoke"] = args.smoke or not args.paper

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for yml in sorted(CLAIMS.glob("claim_*.yaml")):
        claim = yaml.safe_load(yml.read_text())
        if args.claim and claim["id"] != args.claim:
            continue
        trace = evaluate_claim(claim, base)
        (OUT / f"{claim['id']}.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
        summary.append((claim["id"], trace["conclusion"], claim["hypothesis"]))

    print("\n=== Claim verdicts (Evidence -> Conclusion) ===")
    for cid, verdict, hyp in summary:
        print(f"  {cid}: {verdict:12s} {hyp}")
    print("Full evidence traces in outputs/claims/")


if __name__ == "__main__":
    main()
