"""exp7: Privacy Verification — Check for PII leakage in model outputs."""
from __future__ import annotations
import logging
logger = logging.getLogger("exp7")


def run(config: dict) -> dict:
    smoke = config.get("_smoke", False)
    from realeval import data
    ds = data.load_chifraud_balanced()
    texts = ds["texts"]
    used_synthetic = False
    if not texts:
        ds = data.load_synthetic(n=100)
        texts = ds["texts"]
        used_synthetic = True

    from realeval.real_backend import run_paper_safe

    def run_paper(config):
        from realeval import privacy, real_backend
        import numpy as np
        pii_report = privacy.scan_texts(texts)
        # Use REAL F_v embeddings from the dataset's NPZ
        emb = ds.get("embeddings")
        spk_labels = ds.get("speaker_labels")

        # Fallback: load ChiFraud NPZ when TAF28k embeddings unavailable
        if emb is None or spk_labels is None:
            try:
                from realeval.data import _data_root
                cf = np.load(_data_root() / "ChiFraud" / "chifraud.npz")
                emb, spk_labels = cf["embeddings"], cf["speaker_labels"].tolist()
                logger.info("Falling back to ChiFraud NPZ (%d samples)", len(emb))
            except (FileNotFoundError, KeyError) as e:
                logger.warning("ChiFraud NPZ fallback also failed: %s", e)

        real_backend.require_assets(
            emb is not None and spk_labels is not None and len(spk_labels) == len(emb),
            "Real F_v embeddings unavailable (need NPZ embeddings from real audio, not synthetic fallback)")
        emb = np.asarray(emb)
        asv = privacy.asv_eer_open_set(emb, spk_labels, n_enroll_utt=3, seed=42)
        sid = privacy.speaker_identification(emb, spk_labels, seed=42)
        glo = privacy.glo_reconstruction_attack(emb, emb[:, :64] if emb.shape[1] >= 64 else emb, steps=50, seed=42)
        return {"experiment": "exp7", "computation": "h100_real_qwen", "embedding_source": "real_fv",
                "pii_report": pii_report,
                "asv_eer_pct": asv["asv_eer_pct"], "min_dcf": asv.get("min_dcf"),
                "speaker_id_accuracy": sid["accuracy"], "glo_reconstruction_corr": glo["mean_reconstruction_corr"],
                "n_speakers": sid["n_speakers"]}

    paper_result = run_paper_safe(smoke, config, run_paper)
    if paper_result is not None:
        return paper_result

    logger.info("SMOKE: running small-model verification for exp7")
    from realeval import privacy
    import numpy as np
    pii_report = privacy.scan_texts(texts)
    # Embeddings WITH real speaker structure (each speaker is a distinct cluster centre + noise), so the
    # privacy attacks are meaningful — they measure genuine identity leakage, not random-noise chance.
    rng = np.random.RandomState(42)
    n_sp, per = 50, 5
    centres = rng.randn(n_sp, 128) * 2.0
    emb = np.stack([centres[i // per] + rng.randn(128) * 0.5 for i in range(n_sp * per)]).astype(np.float32)
    spk_labels = [f"spk_{i // per}" for i in range(n_sp * per)]
    asv = privacy.asv_eer_open_set(emb, spk_labels, n_enroll_utt=3, seed=42)
    sid = privacy.speaker_identification(emb, spk_labels, seed=42)
    glo = privacy.glo_reconstruction_attack(emb, rng.randn(n_sp * per, 64), steps=50, seed=42)
    return {"experiment": "exp7", "computation": "smoke_privacy", "embedding_source": "synthetic_speaker_structured",
            "pii_report": pii_report,
            "asv_eer_pct": asv["asv_eer_pct"], "min_dcf": asv.get("min_dcf"),
            "speaker_id_accuracy": sid["accuracy"], "glo_reconstruction_corr": glo["mean_reconstruction_corr"],
            "n_speakers": sid["n_speakers"]}
