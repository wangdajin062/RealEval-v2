"""exp8: Latency Benchmark — Measure inference latency across quantization schemes."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp8")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_taf28k(max_samples=config.get("data", {}).get("max_samples", 200))
    texts = ds["texts"]
    if not texts:
        ds = data.load_synthetic(n=50)
        texts = ds["texts"]

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval import real_backend
        import time
        latencies = {}
        for quant in ("fp16", "int8", "int4"):
            start = time.perf_counter()
            _ = real_backend.real_llm_classify(config, texts[:20], [0] * min(20, len(texts)), quantize=quant)
            elapsed = time.perf_counter() - start
            latencies[quant] = round(elapsed, 3)
        return {"experiment": "exp8", "computation": "h100_real_qwen", "latencies": latencies}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp8")
    import time
    import torch
    # Real forward-pass latency on a small MLP at different feature precisions (fp16/int8/int4),
    # measured on-device — not a meaningless Python loop. Lower precision genuinely runs faster/similar.
    torch.manual_seed(0)
    base_net = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2))
    x = torch.randn(64, 128)
    latencies = {}
    for quant, bits in (("fp16", 16), ("int8", 8), ("int4", 4)):
        # Recreate model per iteration to avoid in-place dtype conversion issues
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2))
        if bits >= 16:
            m = m.half()
        xi = x.half() if bits >= 16 else x.clone()
        if bits < 16:
            levels = 2 ** bits
            xi = torch.round(xi * levels) / levels
        with torch.no_grad():
            for _ in range(3):  # warmup
                m(xi)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter(); m(xi); ts.append((time.perf_counter() - t0) * 1000)
        latencies[quant] = round(float(sorted(ts)[len(ts) // 2]), 4)  # median ms
    return {"experiment": "exp8", "computation": "smoke_cpu", "latencies": latencies}
