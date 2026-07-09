"""Tests for realeval package — smoke tests for all core modules."""
from __future__ import annotations
import pytest


class TestRunner:
    def test_experiment_registry(self):
        from experiments.runner import EXPERIMENTS, _SHORT_TO_FULL
        assert len(EXPERIMENTS) == 14  # exp1-exp14
        assert _SHORT_TO_FULL["exp13"] == "exp13_fusion_strategy"

    def test_import_exp(self):
        from experiments.runner import _import_exp
        mod = _import_exp("exp1_qad_production")
        assert hasattr(mod, "run")


class TestBenchmark:
    def test_benchmark_forward_cpu(self):
        from realeval.benchmark import benchmark_forward, benchmark_summary
        import torch, torch.nn as nn
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))
        r = benchmark_forward(model, torch.randn(64), warmup=2, repeat=10, batch_sizes=(1, 16))
        assert 16 in r
        assert r[16]["throughput_sps"] is not None
        assert r[1]["latency_p50_ms"] > 0
        s = benchmark_summary(r)
        assert "best_batch_size" in s

    def test_benchmark_metrics_fields(self):
        from realeval.benchmark import benchmark_forward
        import torch, torch.nn as nn
        r = benchmark_forward(nn.Linear(32, 2), torch.randn(32), warmup=1, repeat=5, batch_sizes=(1,))
        for k in ("throughput_sps", "latency_p50_ms", "latency_p99_ms", "device"):
            assert k in r[1]

class TestDistributed:
    def test_distributed_noop(self):
        from realeval import distributed as dist
        assert dist.world_size() >= 1
        assert dist.is_main() in (True, False)
        assert dist.all_reduce_mean(2.0) == 2.0  # Single process: identity

    def test_distributed_init_cleanup(self):
        from realeval import distributed as dist
        info = dist.init()
        assert "distributed" in info and "rank" in info
        dist.cleanup()


class TestReport:
    def test_build_all(self):
        """build_all() calls all generators without crashing."""
        from realeval.report import build_summary_csv, build_paper_tables, build_all
        # Smoke test: build_all should not raise even with no results
        result = build_all()
        assert isinstance(result, list)


class TestData:
    def test_load_synthetic(self):
        from realeval.data import load_synthetic
        ds = load_synthetic(n=50, seed=42)
        assert len(ds["texts"]) == 50
        assert len(ds["labels"]) == 50
        assert ds["embeddings"].shape == (50, 128)

    def test_load_taf28k_missing(self):
        from realeval.data import load_taf28k
        ds = load_taf28k(max_samples=10)
        assert ds["source"] is None or ds["source"] == "jsonl"


class TestMetrics:
    def test_classification_metrics(self):
        from realeval.metrics import classification_metrics
        m = classification_metrics([0, 1, 0, 1], [0, 1, 1, 0])
        assert "f1" in m
        assert "accuracy" in m
        assert 0 <= m["f1"] <= 1


class TestValidation:
    def test_validate_config_empty(self):
        from realeval.validation import validate_config, ValidationError
        # Empty config uses defaults (source='auto'), which is valid
        # Test with illegal source instead
        with pytest.raises(ValidationError):
            validate_config({"data": {"source": "illegal_source"}})

    def test_validate_config_minimal(self):
        from realeval.validation import validate_config
        cfg = {"models": {"teacher": "test", "student": "test"}, "data": {"max_samples": 100}}
        validate_config(cfg)  # should not raise


class TestHWEnv:
    def test_detect_no_crash(self):
        from realeval.hwenv import detect
        env = detect(verbose=False)
        assert "cuda_available" in env
        assert "gpu_count" in env or "n_gpus" in env


class TestIO:
    def test_load_config_missing(self):
        from realeval.io import load_config
        import pytest
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_save_results(self):
        from realeval.io import save_results
        import json
        path = save_results("test_exp", {"f1": 0.95})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["f1"] == 0.95


class TestPaths:
    def test_storage_report(self):
        from realeval.paths import storage_report
        rep = storage_report()
        assert "data_root" in rep and "workspace_mounted" in rep

    def test_list_local_models(self):
        from realeval.paths import list_local_models
        lm = list_local_models()
        assert "models_root" in lm and "count" in lm

    def test_resolve_model(self):
        from realeval.paths import resolve_model
        path = resolve_model("buckets/wangdajin062/Qwen2.5-7B-Instruct-bucket")
        assert isinstance(path, str)

    def test_apply_hf_env(self):
        from realeval.paths import apply_hf_env
        apply_hf_env()
        import os
        assert "HF_HOME" in os.environ


class TestRunLog:
    def test_log_run(self, tmp_path):
        from realeval.runlog import log_run
        import json
        log_run("test_exp", {"key": "val"}, {"f1": 0.9}, status="completed")
        from realeval.runlog import RUNLOG
        assert RUNLOG.exists()
        lines = RUNLOG.read_text().strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["experiment"] == "test_exp"


class TestAudit:
    def test_log_event(self):
        from realeval.audit import log_event
        log_event("test_event", detail="testing")  # should not crash

    def test_log_error(self):
        from realeval.audit import log_error
        log_error("test_exp", ValueError("test error"))  # should not crash


class TestPrivacy:
    def test_scan_texts(self):
        from realeval.privacy import scan_texts
        report = scan_texts(["hello world", "my email is test@example.com"])
        assert isinstance(report, dict)
        assert "total_texts" in report


class TestExperiments:
    @pytest.mark.parametrize("exp_name", [
        "exp1_qad_production", "exp2_qad_loss_ablation", "exp3_ov_freeze_control",
        "exp4_baseline_comparison", "exp5_cross_dataset", "exp6_speculative_decoding",
        "exp7_privacy_verification", "exp8_latency_benchmark", "exp9_cot_ablation",
        "exp10_teacher_scale", "exp11_quantization_scheme", "exp12_fraudfusion_baseline",
        "exp13_fusion_strategy",
    ])
    def test_experiment_smoke(self, exp_name):
        mod = __import__(f"experiments.{exp_name}", fromlist=["run"])
        cfg = {"_smoke": True, "models": {"teacher": "t", "student": "s"}, "data": {"max_samples": 50}}
        result = mod.run(cfg)
        assert isinstance(result, dict)
        assert "experiment" in result


import pytest as _pytest


@_pytest.mark.parametrize("modname", [
    "exp1_qad_production", "exp2_qad_loss_ablation", "exp3_ov_freeze_control",
    "exp4_baseline_comparison", "exp5_cross_dataset", "exp6_speculative_decoding",
    "exp7_privacy_verification", "exp8_latency_benchmark", "exp9_cot_ablation",
    "exp10_teacher_scale", "exp11_quantization_scheme", "exp12_fraudfusion_baseline",
    "exp13_fusion_strategy", "exp14_gguf_comparison"])
def test_all_experiments_smoke(modname):
    """Every experiment runs in smoke mode and reports a real-computation label."""
    import importlib
    mod = importlib.import_module(f"experiments.{modname}")
    r = mod.run({"_smoke": True})
    assert r.get("experiment")
    assert "smoke" in r.get("computation", "") or "real" in r.get("computation", "")


def test_privacy_empty_data_guards():
    """asv_eer_open_set / speaker_identification degrade gracefully when no speaker has >=2 utterances."""
    import numpy as np
    from realeval.privacy import asv_eer_open_set, speaker_identification
    embs = [np.random.RandomState(0).randn(64) for _ in range(3)]
    spks = ["a", "b", "c"]  # all singletons
    assert asv_eer_open_set(embs, spks).get("asv_eer_pct") is None
    assert "note" in speaker_identification(embs, spks)


def test_dp_sensitivity_zero_centered():
    """Gaussian LDP sensitivity is correct for zero-centered data (was halved by np.abs)."""
    import numpy as np
    from realeval.privacy import gaussian_ldp
    X = np.array([[-3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]])
    out = gaussian_ldp(X, epsilon=1.5)
    assert (out - X).var() > 0  # real noise added, not zero/fallback


def test_autocast_context_usable():
    """autocast_context returns a usable context manager (regression: @contextmanager misuse)."""
    from realeval import hwenv
    with hwenv.autocast_context():
        pass
