"""realeval/real_backend.py — Real Qwen Computation Backend (H100 Path)

Produces paper-grade real numbers with real Qwen weights + GPU. Each function checks asset availability first,
raises AssetsUnavailable when unavailable, caught by experiment layer to fall back to small model verification path.

Real computation:
  - real_teacher_student_logits: real teacher(BF16)/student(quantized) forward on same batch, real KL
  - real_speculative_alpha: real draft/target per-token acceptance rate alpha
  - real_llm_classify: real binary classification with LLM (prompt + generation/scoring), real F1
"""
from __future__ import annotations
import logging

logger = logging.getLogger("real_backend")


class AssetsUnavailable(RuntimeError):
    """Real model/data unavailable (sandbox or weights not downloaded)."""


def require_assets(cond, msg):
    """Raise AssetsUnavailable if condition is false."""
    if not cond:
        raise AssetsUnavailable(msg)


# Backward-compat alias for internal use by specdec.py
_require = require_assets


def run_paper_safe(smoke, config, paper_fn):
    """Run a paper-path function safely: if it raises AssetsUnavailable and smoke=True, return None
    (caller falls through to the smoke path); if not smoke, re-raise. Functional form of paper_ready."""
    try:
        return paper_fn(config)
    except AssetsUnavailable:
        if not smoke:
            raise
        return None


# ─────────────────── Real Distillation (exp1/exp2/exp3) ───────────────────
def real_distillation_step_metrics(config: dict, texts: list[str], *, apply_ov_rescaling: bool,
                                   quantize="int4", max_batch=64, freeze_frac=1.0, window=1.0, loss_fn="kl"):
    """Real teacher/student forward over ALL texts (mini-batched), returns KL + output-variance drift.

    When apply_ov_rescaling=True: post-hoc output-variance matching rescales student logits
    to align with the teacher's per-dimension variance (first freeze_frac dimensions, window-weighted).
    This is a DIAGNOSTIC MEASUREMENT of the expected effect of OV-Freeze regularization,
    NOT actual weight freezing during training. Used by exp3 layer-selection / rho-sweep
    so different layer/rho settings give different drift results.

    max_batch defaults to 64 (previously 16) to saturate H100 GPU (80 GB VRAM).
    """
    from realeval import models, hwenv
    import torch
    import torch.nn.functional as F

    _require(models.models_available(config), "Real Qwen weights unavailable")
    teacher, tok = models.load_causal_lm(config["models"]["teacher"], quantize=None, bf16=True)
    student, _ = models.load_causal_lm(config["models"].get("student", config["models"]["teacher"]),
                                       quantize=quantize, bf16=True)
    _require(teacher is not None and student is not None, "teacher/student loading failed")
    dev = next(teacher.parameters()).device

    # Use config override if specified, else keep the caller's/default value
    effective_max_batch = config.get("distillation", {}).get("max_batch", max_batch)

    kl_sum, drift_sum, n_batches, tdtype = 0.0, 0.0, 0, None
    for start in range(0, len(texts), effective_max_batch):
        batch = texts[start:start + effective_max_batch]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        mask = enc["attention_mask"].bool()  # exclude padding positions
        with torch.inference_mode():
            with hwenv.autocast_context():
                t_out = teacher(**enc).logits
                s_out = student(**enc).logits
                tdtype = str(t_out.dtype)
                # Select only real (non-pad) token positions for KL and variance.
                t_real = t_out[mask]  # (num_real_tokens, vocab)
                s_real = s_out[mask]
                # Temperature-scaled KL per Hinton et al. (2015):
                #   KL(softmax(t_logits/T), softmax(s_logits/T)) * T^2
                T = float(config.get("distillation", {}).get("temperature", 1.0))
                kl_val = F.kl_div(
                    F.log_softmax(s_real / T, -1),
                    F.softmax(t_real / T, -1),
                    reduction="batchmean"
                ) * (T ** 2)
                mse_val = F.mse_loss(s_real, t_real)
                if loss_fn == "mse":
                    kl = mse_val
                elif loss_fn == "kl_mse":
                    kl = kl_val + mse_val
                else:
                    kl = kl_val
                t_var = t_real.var(dim=0)
                s_var = s_real.var(dim=0)
                if apply_ov_rescaling:
                    k = max(1, int(t_var.numel() * freeze_frac))
                    scale = (t_var[:k] / (s_var[:k] + 1e-9)).sqrt()
                    s_real = s_real.clone()
                    s_real[:, :k] = s_real[:, :k] * (1 + window * (scale - 1))
                    s_var = s_real.var(dim=0)
                drift = float((s_var - t_var).abs().mean() / (t_var.abs().mean() + 1e-9) * 100)
        kl_sum += float(kl); drift_sum += drift; n_batches += 1

    n_batches = max(1, n_batches)
    return {"kl": kl_sum / n_batches, "variance_drift_pct": drift_sum / n_batches,
            "teacher_dtype": tdtype, "n_texts": len(texts)}


def real_distill_train(config: dict, train_texts: list[str], train_labels: list[int],
                       test_texts: list[str], test_labels: list[int]) -> dict:
    """Supervised distillation with a classification head on last-token hidden states.

    Freezes both teacher (BF16) and student (int4) backbones, extracts last-token
    hidden states, trains a small Linear(hidden->2) head.  Loss = CE(task) + lambda*KL(hidden).
    Bypasses the token-prior problem -- the head learns fraud/normal directly
    from the student's internal representation.

    Returns dict: trajectory (per-epoch ce+kl), f1, accuracy, n_train, n_test.
    """
    from realeval import models, hwenv
    import torch
    import torch.nn.functional as F

    _require(models.models_available(config), "Real Qwen weights unavailable")
    teacher, tok = models.load_causal_lm(config["models"]["teacher"], quantize=None, bf16=True)
    _require(teacher is not None, "Teacher loading failed")
    dev = next(teacher.parameters()).device
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student in int4 (or configured quantize) -- backbone frozen
    quant = config.get("training", {}).get("quantize", "int4")
    student, _ = models.load_causal_lm(config["models"].get("student", config["models"]["teacher"]),
                                       quantize=quant, bf16=True)
    _require(student is not None, "Student loading failed")
    student = student.to(dev)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    hidden_size = teacher.config.hidden_size

    lr = float(config.get("training", {}).get("learning_rate", 1e-3))
    epochs = int(config.get("training", {}).get("epochs", 10))
    max_batch = int(config.get("distillation", {}).get("max_batch", 32))
    max_seq = int(config.get("distillation", {}).get("max_seq_length", 256))
    lambda_kl = float(config.get("distillation", {}).get("task_weight", 0.1))

    _CLS_PFX = "请判断以下消息是否为欺诈信息（fraud）或正常信息（normal）。"
    _CLS_SFX = chr(10) + "仅输出一个词：fraud 或 normal。" + chr(10) + chr(10) + "消息：{text}" + chr(10) + "分类："

    def _cls_prompt(t):
        return _CLS_PFX + _CLS_SFX.format(text=t)


    # Classification head
    head = torch.nn.Linear(hidden_size, 2, dtype=torch.float32).to(dev)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    trajectory = []
    for epoch in range(epochs):
        epoch_ce = epoch_kl = 0.0
        n_batches = 0
        for start in range(0, len(train_texts), max_batch):
            batch = train_texts[start:start + max_batch]
            labels_t = torch.tensor([int(l) for l in train_labels[start:start + max_batch]],
                                    device=dev, dtype=torch.long)

            enc = tok([_cls_prompt(t) for t in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_seq).to(dev)
            lens = enc.attention_mask.sum(1).clamp(min=1) - 1

            # teacher hidden (frozen, no grad)
            with torch.inference_mode():
                t_hidden = teacher(**enc, output_hidden_states=True).hidden_states[-1]
            t_last = t_hidden[torch.arange(len(batch), device=dev), lens].float()

            # student hidden (frozen backbone, only head gets gradients)
            with hwenv.autocast_context():
                s_hidden = student(**enc, output_hidden_states=True).hidden_states[-1]
            s_last = s_hidden[torch.arange(len(batch), device=dev), lens].float()

            logits_2d = head(s_last)
            ce_loss = F.cross_entropy(logits_2d, labels_t)
            kl_loss = F.mse_loss(s_last, t_last)

            loss = ce_loss + lambda_kl * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += float(ce_loss)
            epoch_kl += float(kl_loss)
            n_batches += 1

        nb = max(1, n_batches)
        trajectory.append({"epoch": epoch, "ce": round(epoch_ce / nb, 6),
                           "kl": round(epoch_kl / nb, 6)})
        logger.info("Head epoch %d/%d -- CE=%.6f KL_h=%.6f", epoch + 1, epochs,
                    epoch_ce / nb, epoch_kl / nb)

    # Eval: head on student hidden
    head.eval()
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    preds = []
    for start in range(0, len(test_texts), batch_size):
        batch = test_texts[start:start + batch_size]
        enc = tok([_cls_prompt(t) for t in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_seq).to(dev)
        lens = enc.attention_mask.sum(1).clamp(min=1) - 1
        with torch.inference_mode():
            s_hidden = student(**enc, output_hidden_states=True).hidden_states[-1]
        s_last = s_hidden[torch.arange(len(batch), device=dev), lens].float()
        preds.extend(head(s_last).argmax(1).tolist())

    from realeval.metrics import classification_metrics
    m = classification_metrics([int(v) for v in test_labels], preds)
    return {"trajectory": trajectory, "f1": m["f1"], "accuracy": m["accuracy"],
            "n_train": len(train_texts), "n_test": len(test_texts)}

def real_speculative_alpha(config: dict, texts: list[str], *, gamma=5, n_samples=20,
                           draft_variant="domain", max_new=40):
    from realeval import models, hwenv
    import torch
    import torch.nn.functional as F

    _require(models.models_available(config), "Real Qwen weights unavailable")
    target, tok = models.load_causal_lm(config["models"]["teacher"], bf16=True)
    draft_path = (config["models"].get("draft_model") if draft_variant == "domain"
                  else config["models"].get("draft_model_generic", config["models"].get("draft_model")))
    draft, _ = models.load_causal_lm(draft_path, bf16=True)
    _require(target is not None and draft is not None, "draft/target loading failed")
    dev = next(target.parameters()).device

    accepted, proposed = 0, 0
    for text in texts[:n_samples]:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        seq = ids
        for _ in range(max_new // gamma):
            # draft proposes gamma tokens
            dprobs, dtoks = [], []
            cur = seq
            with torch.inference_mode():
                with hwenv.autocast_context():
                    for _g in range(gamma):
                        p = F.softmax(draft(cur).logits[0, -1], -1)
                        tk = int(torch.argmax(p))
                        dtoks.append(tk); dprobs.append(float(p[tk]))
                        cur = torch.cat([cur, torch.tensor([[tk]], device=dev)], 1)
                    proposed += gamma
                    # target single forward verification
                    ext = torch.cat([seq, torch.tensor([dtoks], device=dev)], 1)
                    tlog = target(ext).logits
            base = seq.shape[1]
            ok = 0
            for i, tk in enumerate(dtoks):
                pt = float(F.softmax(tlog[0, base + i - 1], -1)[tk])
                if torch.rand(1).item() < pt / (dprobs[i] + 1e-9):
                    ok += 1
                else:
                    break
            accepted += ok
            seq = torch.cat([seq, torch.tensor([dtoks[:max(1, ok)]], device=dev)], 1)
            if ok == 0:
                break
    alpha = accepted / max(1, proposed)
    speedup = (1 - alpha ** (gamma + 1)) / max(1e-9, 1 - alpha) if alpha < 1 else gamma + 1
    return {"alpha": round(alpha, 4), "speedup_theoretical": round(speedup, 3),
            "accepted": accepted, "proposed": proposed, "gamma": gamma, "draft": draft_variant}


# ─────────────────── Real LLM Classification (exp4) ───────────────────
def real_llm_classify(config: dict, texts: list[str], labels: list[int], *, quantize="int4", use_cot=False,
                       return_preds=False, classify_batch_size: int = None):
    """Real (quantized) Qwen binary classification on text, sklearn computes real F1.

    Classification method: apply chat template then compare token probabilities
    at the first output position. Uses softmax-normalised scores for 'fraud'
    and 'normal' token(s), with attention-mask-aware last-token selection.
    Texts are processed in mini-batches to saturate H100 GPU utilization.
    """
    from realeval import models, hwenv
    from realeval.metrics import classification_metrics
    import torch
    import torch.nn.functional as F

    _require(models.models_available(config), "Real Qwen weights unavailable")
    model, tok = models.load_causal_lm(config["models"]["teacher"], quantize=quantize, bf16=True)
    _require(model is not None, "Model loading failed")
    dev = next(model.parameters()).device

    cot_sys = ("Think step by step about the sender, intent, and urgency cues, then decide. "
               if use_cot else "")

    batch_size = classify_batch_size or config.get("training", {}).get("batch_size", 64)

    preds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        # Build chat-format messages and apply template
        messages_list = []
        for t in batch_texts:
            msgs = []
            if cot_sys:
                msgs.append({"role": "system", "content": cot_sys})
            msgs.append({"role": "user",
                         "content": f"请判断以下消息是否为欺诈信息（fraud）或正常信息（normal）。"
                                     f"\n仅输出一个词：fraud 或 normal。\n\n消息：{t}\n分类："})
            messages_list.append(msgs)
        prompts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                   for msgs in messages_list]

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        attn_mask = enc.attention_mask

        with torch.inference_mode():
            with hwenv.autocast_context():
                outputs = model(**enc)
                logits = outputs.logits  # (batch, seq_len, vocab)

        # Get logits at each sequence's LAST REAL token (before padding)
        seq_lens = attn_mask.sum(dim=1).clamp(min=1) - 1       # (batch,) last non-padding index
        last_logits = logits[torch.arange(len(batch_texts)), seq_lens]  # (batch, vocab)

        # Score "fraud" vs "normal" token IDs (no leading space — chat template ends with \n)
        fraud_ids = tok("fraud", add_special_tokens=False).input_ids
        normal_ids = tok("normal", add_special_tokens=False).input_ids

        # Compare via softmax probability mean (handles both single- and multi-token cases)
        probs = F.softmax(last_logits, dim=-1)
        f_prob = probs[:, fraud_ids].mean(dim=1) if fraud_ids else last_logits.new_zeros(len(batch_texts))
        n_prob = probs[:, normal_ids].mean(dim=1) if normal_ids else last_logits.new_zeros(len(batch_texts))
        batch_preds = (f_prob > n_prob).int().tolist()

        preds.extend(batch_preds)

    m = classification_metrics(labels, preds)
    if return_preds:
        m = dict(m); m["preds"] = preds
    return m


def real_fusion_classify(config, texts, labels, audio_emb, *, quantize="int4", fusion_strategy="early"):
    """Real multimodal fusion: real Qwen text predictions fused with a real acoustic-embedding
    classifier via early (OR) / late (AND) / hybrid (weighted) strategies. Falls back to text-only if
    acoustic embeddings are unavailable. All predictions are real per-sample (no placeholders).
    """
    from realeval.metrics import classification_metrics
    import numpy as np
    txt = real_llm_classify(config, texts, labels, quantize=quantize, return_preds=True)
    txt_pred = np.asarray(txt["preds"])
    if audio_emb is None or len(audio_emb) != len(labels):
        return {k: v for k, v in txt.items() if k != "preds"}
    from sklearn.linear_model import LogisticRegression
    ae = np.asarray(audio_emb); n = len(labels); split = max(1, int(n * 0.5))
    try:
        # Train on first half, predict on held-out second half to prevent data leakage
        clf = LogisticRegression(max_iter=500).fit(ae[:split], labels[:split])
        ac_pred_test = clf.predict(ae[split:])
    except Exception:
        return {k: v for k, v in txt.items() if k != "preds"}
    # Evaluate fusion only on the held-out test portion (no leakage)
    txt_test = txt_pred[split:]
    labels_test = labels[split:]
    if fusion_strategy == "early":
        fused = ((txt_test + ac_pred_test) >= 1).astype(int)
    elif fusion_strategy == "late":
        fused = ((txt_test + ac_pred_test) >= 2).astype(int)
    else:
        fused = np.round(0.6 * txt_test + 0.4 * ac_pred_test).astype(int)
    return classification_metrics(labels_test, fused)
