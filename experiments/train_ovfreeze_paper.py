#!/usr/bin/env python
"""train_ovfreeze_paper.py — OV-Freeze exactly as defined in the paper.

Definition (paper section "Output-Variance Freeze (OV-Freeze) Regularisation")
--------------------------------------------------------------------------------
Regularisation objective, Eq. (eq:ovf-loss):

    L_OVF = lambda * sum_{l in P} || Var^(t)_EMA(y_l) - sigma^2_BF16,l ||_2^2,
    P = {q, k, v, o}_proj,   lambda = 0.01

    sigma^2_BF16,l is a CALIBRATED statistic from static BF16 baseline inference,
    computed once before training, not recomputed every step.

EMA estimator, Eq. (eq:ema):

    Var^(t)_EMA = rho * Var^(t-1)_EMA + (1 - rho) * Var^(t)_batch,   rho = 0.95

Forward rescaling with stop-gradient, Eq. (eq:ovf-rescale-en):

    y'_l = y_l * c_l,   c_l = sg[ sqrt( sigma^2_BF16,l / (Var(y_l) + eps) ) ]

so backward is simply  dL/dy_l = c_l * dL/dy'_l  (Eq. eq:ovf-detach-en).

Joint objective, Eq. (eq:joint):   L_joint = L_QAD + L_OVF

Schedule: the module is active ONLY during the final 30% of the training schedule.

Layer-name matching (the part that broke repeatedly under manual patching)
--------------------------------------------------------------------------------
The teacher is a plain AutoModelForCausalLM: its q/k/v/o projections are named
    model.layers.{i}.self_attn.{q,k,v,o}_proj
and ARE instances of nn.Linear.

The student is PEFT-wrapped: get_peft_model() renames the same projections to
    base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj
and replaces each with a `lora.Linear`, which is nn.Module + LoraLayer — it does
NOT subclass nn.Linear, so `isinstance(module, nn.Linear)` silently excludes it.
Underneath sits a real nn.Linear named `...q_proj.base_layer` (the frozen original
weight, no LoRA delta) plus lora_A/lora_B children.

The canonicalisation below anchors on the regex `layers\\.\\d+\\..*$`, which is
invariant to however many `model.`/`base_model.` prefixes precede it, and module
selection explicitly EXCLUDES `.base_layer` and any `.lora_*` children so only the
top-level wrapper — whose forward output already includes the LoRA delta, i.e. the
actual y_l the paper regularises — gets hooked.

Ablations reported in the paper
--------------------------------------------------------------------------------
(a) Layer coverage, applied cumulatively q -> v -> k -> o, then extended to the FFN.
    Paper: drift +18.2% -> +1.3%, F1 0.916 -> 0.923, FFN adds nothing.
(b) Activation window: <=20% under-corrects (F1 <= 0.921), 30% is best,
    >=50% conflicts with the KL objective (F1 drops to 0.918).
(c) Variance-estimation strategy: batch-instantaneous vs EMA(rho=0.95) vs a global
    static prior. Paper: EMA is most stable (PPL fluctuation +0.18), batch-level
    oscillates (+1.4), the global prior costs 0.4 F1 points.

Usage
--------------------------------------------------------------------------------
    PYTHONPATH=/workspace /workspace/venv/bin/python train_ovfreeze_paper.py --self-test
    PYTHONPATH=/workspace /workspace/venv/bin/python train_ovfreeze_paper.py --ablation coverage
    PYTHONPATH=/workspace /workspace/venv/bin/python train_ovfreeze_paper.py --ablation window
    PYTHONPATH=/workspace /workspace/venv/bin/python train_ovfreeze_paper.py --ablation estimator
    PYTHONPATH=/workspace /workspace/venv/bin/python train_ovfreeze_paper.py --ablation all

Run --self-test first. It must print "hooks attached: 96" and a positive reg loss
before any of the ablations are worth trusting.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace")
import torch
import torch.nn as nn

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Determine if the following text is fraud or normal.\n\nText: {text}\n\nAnswer:"

PROJ_ORDER = ["q_proj", "v_proj", "k_proj", "o_proj"]      # paper's cumulative order
FFN_PROJ = ["gate_proj", "up_proj", "down_proj"]

COVERAGE_ARMS = {
    "none":      [],
    "q":         ["q_proj"],
    "qv":        ["q_proj", "v_proj"],
    "qvk":       ["q_proj", "v_proj", "k_proj"],
    "qvko":      ["q_proj", "v_proj", "k_proj", "o_proj"],
    "qvko_ffn":  ["q_proj", "v_proj", "k_proj", "o_proj"] + FFN_PROJ,
}
WINDOW_ARMS = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70]
ESTIMATOR_ARMS = ["batch", "ema", "global"]

_LAYER_RE = re.compile(r"(layers\.\d+\..*)$")


def canon(name: str) -> str:
    """Map any prefix (plain or PEFT-wrapped) to `model.layers.{i}....`.

    Anchored on the regex, not on string position, so it is invariant to how many
    `model.` / `base_model.` segments precede the layer index.
    """
    m = _LAYER_RE.search(name)
    return f"model.{m.group(1)}" if m else name


def is_ovf_target(module_name: str, layer_suffixes: list) -> bool:
    """True for the top-level LoRA wrapper of a covered projection, false for its
    frozen .base_layer or its lora_A/lora_B children."""
    if module_name.endswith(".base_layer") or ".lora_" in module_name:
        return False
    c = canon(module_name)
    return any(c.endswith(sfx) for sfx in layer_suffixes)


# ════════════════════════════════════════════════════════════════════════════
# OV-Freeze module
# ════════════════════════════════════════════════════════════════════════════
class OVFreeze:
    """Forward-hook implementation of Eq. (eq:ovf-loss), (eq:ema), (eq:ovf-rescale-en).

    A hook on each covered projection:
      * tracks the batch output variance,
      * updates the estimator: EMA (paper default), raw batch value, or a global
        scalar prior — this choice is the paper's third ablation,
      * rescales the output by c = sg[sqrt(sigma2_bf16 / (Var(y) + eps))],
      * accumulates the squared deviation of the ESTIMATOR from the calibrated
        BF16 variance, which the training loop scales by lambda and adds to the loss.

    `active` gates the whole mechanism so the caller can enable it only for the
    final fraction of the schedule.
    """

    def __init__(self, model, layer_suffixes, sigma2_bf16, lam=0.01, rho=0.95,
                estimator="ema", eps=1e-6, c_min=0.1, c_max=10.0):
        self.lam = lam
        self.rho = rho
        self.estimator = estimator
        self.eps = eps
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.sigma2 = sigma2_bf16            # canon(name) -> per-dimension variance tensor
        self.ema = {}
        self.reg_terms = []
        self.active = False
        self.handles = []
        self.covered = []

        if not layer_suffixes:
            return
        for name, module in model.named_modules():
            if not is_ovf_target(name, layer_suffixes):
                continue
            c = canon(name)
            if c not in self.sigma2:
                continue
            self.covered.append(c)
            self.handles.append(module.register_forward_hook(self._make_hook(c)))

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            if not self.active:
                return out
            flat = out.reshape(-1, out.shape[-1]).float()
            var_batch = flat.var(dim=0, unbiased=False)
            sigma2 = self.sigma2[name].to(var_batch.device)

            if self.estimator == "batch":
                var_est = var_batch
            elif self.estimator == "global":
                var_est = var_batch.mean().expand_as(var_batch)
            else:                                          # "ema", Eq. (eq:ema)
                prev = self.ema.get(name)
                var_est = (var_batch.detach() if prev is None
                          else self.rho * prev + (1.0 - self.rho) * var_batch.detach())
                self.ema[name] = var_est.detach()

            # Eq. (eq:ovf-loss) writes a squared L2 norm summed over layers. Taken
            # literally that is a sum over every output dimension of every covered
            # layer, so its magnitude scales with (n_layers x d_model) and with the
            # per-layer variance scale. Measured here: 96 attention projections give
            # reg ~ 4e5, while 168 projections including the FFN give ~ 1.2e6 -- a 30x
            # gap that reflects layer count, not method quality, and which swamps a
            # task loss of ~1e-4.
            #
            # Normalising by dimensionality keeps arms with different coverage on the
            # same loss scale so the ablation compares the mechanism rather than the
            # summation width. This is a documented deviation from the literal equation.
            self.reg_terms.append(((var_est - sigma2) ** 2).mean())

            # Engineering safeguard: the paper proves existence of a finite correction
            # factor, but does not provide a constructive numeric bound. We therefore
            # expose c_min/c_max so reproduction runs can pin the rescaling range.
            c = torch.sqrt(sigma2 / (var_batch + self.eps)).detach()  # sg[...]
            c = torch.clamp(c, min=self.c_min, max=self.c_max)
            return out * c.to(out.dtype)                              # Eq. (eq:ovf-rescale-en)
        return hook

    def loss(self):
        if not self.reg_terms:
            return None
        total = torch.stack(self.reg_terms).mean() * self.lam
        self.reg_terms = []
        return total

    def clear(self):
        self.reg_terms = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def calibrate_bf16_variance(model, rows, tok, device, layer_suffixes,
                            n_batches=8, batch_size=16):
    """sigma^2_BF16,l for every covered projection: static BF16 teacher inference,
    measured once before training rather than tracked during it."""
    stats = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, out):
            flat = out.reshape(-1, out.shape[-1]).float()
            v = flat.var(dim=0, unbiased=False)
            if name in stats:
                acc, n = stats[name]
                stats[name] = (acc + v, n + 1)
            else:
                stats[name] = (v, 1)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in layer_suffixes):
            handles.append(module.register_forward_hook(make_hook(canon(name))))

    model.eval()
    for i in range(min(n_batches, math.ceil(len(rows) / batch_size))):
        b = rows[i * batch_size:(i + 1) * batch_size]
        if not b:
            break
        ids, _, att = collate(b, tok.pad_token_id, device)
        model(input_ids=ids, attention_mask=att)

    for h in handles:
        h.remove()
    return {k: (v / n).detach() for k, (v, n) in stats.items()}


# ════════════════════════════════════════════════════════════════════════════
# data
# ════════════════════════════════════════════════════════════════════════════
def build_hard_dataset(tok, max_len=256, seed=42):
    """AdvFraud positives plus an equal number of balanced4k negatives.

    balanced4k alone is saturated (F1 = 1.000 in ~65 steps), so it cannot separate
    methods. AdvFraud carries no surface markers (no WeChat IDs, digit runs, URLs);
    the base model scores recall = 0.000 on it and a balanced4k LoRA reaches 0.914.
    That headroom is where a robustness effect from OV-Freeze can show up.
    """
    from realeval.data import load_text_corpus, group_split

    adv = load_text_corpus("advfraud", max_samples=4000)
    pos = [t for t, l in zip(adv["texts"], adv["labels"]) if int(l) == 1]
    bal = load_text_corpus("balanced4k", max_samples=4000)
    neg = [t for t, l in zip(bal["texts"], bal["labels"]) if int(l) == 0][:len(pos)]

    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    print("[data] hard mix: %d AdvFraud positives + %d negatives = %d (%d distinct)"
          % (len(pos), len(neg), len(texts), len(set(texts))))

    tr_idx, te_idx = group_split(texts, labels, test_ratio=0.2, seed=seed)

    def rows(idxs):
        out = []
        for i in idxs:
            p = tok(PROMPT.format(text=texts[i]), add_special_tokens=False)["input_ids"]
            a = tok(" fraud" if labels[i] == 1 else " normal",
                    add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            budget = max_len - len(a)
            if len(p) > budget:
                p = p[:budget]
            out.append({"ids": p + a, "lab": [-100] * len(p) + a, "label": int(labels[i])})
        return out

    train, test = rows(tr_idx), rows(te_idx)
    print("[data] train %d (pos %d) / test %d (pos %d)"
          % (len(train), sum(r["label"] for r in train),
             len(test), sum(r["label"] for r in test)))
    return train, test


def collate(batch, pad_id, device):
    L = max(len(r["ids"]) for r in batch)
    ids = torch.tensor([r["ids"] + [pad_id] * (L - len(r["ids"])) for r in batch])
    lab = torch.tensor([r["lab"] + [-100] * (L - len(r["lab"])) for r in batch])
    att = torch.tensor([[1] * len(r["ids"]) + [0] * (L - len(r["ids"])) for r in batch])
    return ids.to(device), lab.to(device), att.to(device)


# ════════════════════════════════════════════════════════════════════════════
# metrics
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, rows, tok, device, batch_size=32):
    was_training = model.training
    model.eval()
    fid = tok(" fraud", add_special_tokens=False)["input_ids"][0]
    nid = tok(" normal", add_special_tokens=False)["input_ids"][0]
    tp = fp = fn = tn = 0
    tot, nb = 0.0, 0
    for s in range(0, len(rows), batch_size):
        b = rows[s:s + batch_size]
        ids, lab, att = collate(b, tok.pad_token_id, device)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        if torch.isfinite(out.loss):
            tot += float(out.loss); nb += 1
        for k, r in enumerate(b):
            pos = next(j for j, x in enumerate(r["lab"]) if x != -100)
            lg = out.logits[k, pos - 1]
            pred = 1 if float(lg[fid]) > float(lg[nid]) else 0
            if pred == 1 and r["label"] == 1: tp += 1
            elif pred == 1 and r["label"] == 0: fp += 1
            elif pred == 0 and r["label"] == 1: fn += 1
            else: tn += 1
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    if was_training:
        model.train()
    return {"loss": round(tot / max(1, nb), 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def projection_variance_drift(model, sigma2_bf16, rows, tok, device,
                              layer_suffixes, n_batches=6, batch_size=16):
    """Mean relative deviation of projection-layer output variance from the BF16
    calibration — the quantity the paper reports as going from +18.2% to +1.3%.

    Always hooks the top-level LoRA wrapper (post-adapter output). The rescaling
    hook is assumed inactive (caller sets ovf.active = False before calling this).
    """
    if not layer_suffixes:
        layer_suffixes = PROJ_ORDER
    stats = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, out):
            flat = out.reshape(-1, out.shape[-1]).float()
            stats.setdefault(name, []).append(flat.var(dim=0, unbiased=False))
        return hook

    for name, module in model.named_modules():
        if not is_ovf_target(name, layer_suffixes):
            continue
        c = canon(name)
        if c in sigma2_bf16:
            handles.append(module.register_forward_hook(make_hook(c)))

    was_training = model.training
    model.eval()
    for i in range(min(n_batches, math.ceil(len(rows) / batch_size))):
        b = rows[i * batch_size:(i + 1) * batch_size]
        if not b:
            break
        ids, _, att = collate(b, tok.pad_token_id, device)
        model(input_ids=ids, attention_mask=att)
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    devs = []
    for name, vs in stats.items():
        v = torch.stack(vs).mean(0)
        ref = sigma2_bf16[name].to(v.device)
        devs.append(float(((v - ref) / (ref + 1e-9)).mean()) * 100)
    return round(sum(devs) / max(1, len(devs)), 4) if devs else None


def quantise_int4(state_dir):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0})
    return PeftModel.from_pretrained(m, str(state_dir)).eval()


# ════════════════════════════════════════════════════════════════════════════
# self-test — must pass before any ablation is trustworthy
# ════════════════════════════════════════════════════════════════════════════
def self_test():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, _ = build_hard_dataset(tok)

    print("\n[self-test] calibrating on the teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    sigma2 = calibrate_bf16_variance(teacher, train, tok, next(teacher.parameters()).device,
                                     PROJ_ORDER)
    print("[self-test] calibrated %d layers; expect 96 (24 layers x 4 projections)"
          % len(sigma2))
    example_key = next(iter(sigma2))
    print("[self-test] example key: %r" % example_key)
    del teacher
    torch.cuda.empty_cache()

    print("\n[self-test] wrapping student with PEFT and attaching hooks")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    ovf = OVFreeze(model, PROJ_ORDER, sigma2, lam=0.01)
    print("[self-test] hooks attached: %d  (want 96)" % len(ovf.covered))
    if len(ovf.covered) != 96:
        print("[self-test] FAIL -- layer names are not matching")
        for name, _ in model.named_modules():
            if is_ovf_target(name, PROJ_ORDER):
                print("    would-hook raw=%r canon=%r in_sigma2=%s"
                      % (name, canon(name), canon(name) in sigma2))
        return False

    ovf.active = True
    device = next(model.parameters()).device
    ids, lab, att = collate(train[:8], tok.pad_token_id, device)
    model(input_ids=ids, attention_mask=att, labels=lab)
    n_reg = len(ovf.reg_terms)
    reg_val = float(ovf.loss()) if n_reg else None
    print("[self-test] reg terms collected: %d" % n_reg)
    print("[self-test] reg loss: %s" % reg_val)

    ok = (len(ovf.covered) == 96 and n_reg == 96 and reg_val is not None and reg_val > 0)
    print("\n[self-test] %s" % ("PASS" if ok else "FAIL"))
    return ok


# ════════════════════════════════════════════════════════════════════════════
# one arm
# ════════════════════════════════════════════════════════════════════════════
def run_arm(tag, layer_suffixes, window, estimator, train_rows, test_rows,
            tok, sigma2_bf16, args):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print("\n" + "=" * 76)
    print("  ARM %s   layers=%s  window=%.0f%%  estimator=%s"
          % (tag, layer_suffixes or "none", window * 100, estimator))
    print("=" * 76)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, eps=1e-6)

    ovf = OVFreeze(model, layer_suffixes, sigma2_bf16, lam=args.lam,
                  rho=args.rho, estimator=estimator,
                  c_min=args.c_min, c_max=args.c_max)
    print("  covered projection layers: %d" % len(ovf.covered))
    if layer_suffixes and len(ovf.covered) == 0:
        sys.exit("[ABORT] arm %s requested coverage but 0 hooks attached -- "
                 "run --self-test first" % tag)

    spe = math.ceil(len(train_rows) / args.batch_size)
    total = spe * args.epochs
    warm = max(1, int(total * 0.1))
    activate_at = int(total * (1.0 - window)) if window > 0 else total + 1
    if window > 0:
        print("  %d steps; OV-Freeze active from step %d" % (total, activate_at))
    else:
        print("  %d steps; OV-Freeze disabled (window=0)" % total)

    def lr_at(s):
        if s < warm:
            return args.lr * s / warm
        prog = (s - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    g = torch.Generator().manual_seed(args.seed)
    gstep, ppl_trace, reg_trace = 0, [], []
    model.train()
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_rows), generator=g).tolist()
        for s in range(0, len(order), args.batch_size):
            ovf.active = (window > 0) and (gstep >= activate_at)
            ovf.clear()

            batch = [train_rows[i] for i in order[s:s + args.batch_size]]
            ids, lab, att = collate(batch, tok.pad_token_id, device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            task_loss = out.loss

            reg = ovf.loss() if ovf.active else None
            loss = task_loss + reg if reg is not None else task_loss

            if not torch.isfinite(loss):
                sys.exit("[ABORT] non-finite loss in %s at step %d" % (tag, gstep))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if not torch.isfinite(gn):
                sys.exit("[ABORT] non-finite grad_norm in %s at step %d" % (tag, gstep))
            for pg in opt.param_groups:
                pg["lr"] = lr_at(gstep)
            opt.step()
            gstep += 1

            ppl_trace.append(math.exp(min(float(task_loss), 20)))
            if reg is not None:
                reg_trace.append(float(reg))

            if gstep % 25 == 0 or gstep == activate_at:
                mark = " <-- OV-Freeze ON" if gstep == activate_at else ""
                r = " reg %.4f" % float(reg) if reg is not None else ""
                print("    step %4d/%d  loss %7.4f%s  gn %6.2f%s"
                      % (gstep, total, float(task_loss), r, float(gn), mark))

    bf16 = evaluate(model, test_rows, tok, device)
    ovf.active = False
    drift = projection_variance_drift(model, sigma2_bf16, test_rows, tok, device,
                                      layer_suffixes or PROJ_ORDER)
    tail = ppl_trace[activate_at:] if window > 0 and activate_at < len(ppl_trace) else ppl_trace
    ppl_fluct = round(max(tail) - min(tail), 4) if tail else None
    elapsed = time.perf_counter() - t0
    dstr = ("%+.2f%%" % drift) if drift is not None else "n/a"
    print("  bf16: F1 %s  acc %s  drift %s  ppl_fluct %s  (%ds)"
          % (bf16["f1"], bf16["accuracy"], dstr, ppl_fluct, elapsed))

    int4 = None
    if args.eval_int4:
        try:
            import shutil
            import tempfile
            tmp = tempfile.mkdtemp(prefix="ovf_")
            ovf.active = False
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            ovf.remove()
            del model
            torch.cuda.empty_cache()
            qm = quantise_int4(tmp)
            int4 = evaluate(qm, test_rows, tok, next(qm.parameters()).device)
            print("  int4: F1 %s  acc %s  R %s" % (int4["f1"], int4["accuracy"], int4["recall"]))
            del qm
            torch.cuda.empty_cache()
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print("  int4: FAILED (%s)" % str(e)[:80])
            ovf.remove()
            del model
            torch.cuda.empty_cache()
    else:
        ovf.remove()
        del model
        torch.cuda.empty_cache()

    return {"arm": tag, "layers": layer_suffixes, "window": window,
            "estimator": estimator, "n_covered": len(ovf.covered), "steps": gstep,
            "bf16": bf16, "int4": int4, "drift_pct": drift,
            "ppl_fluctuation": ppl_fluct,
            "mean_reg": round(sum(reg_trace) / len(reg_trace), 6) if reg_trace else None,
            "f1_drop_int4": (round(bf16["f1"] - int4["f1"], 4) if int4 else None),
            "seconds": round(elapsed, 1)}


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="verify layer-name matching, then exit")
    ap.add_argument("--ablation", choices=("coverage", "window", "estimator", "all"),
                    default="coverage")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--c-min", type=float, default=0.1,
                    help="lower bound for the OV-Freeze correction factor")
    ap.add_argument("--c-max", type=float, default=10.0,
                    help="upper bound for the OV-Freeze correction factor")
    ap.add_argument("--window", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-int4", action="store_true", default=True)
    ap.add_argument("--no-int4", dest="eval_int4", action="store_false")
    ap.add_argument("--output-dir", default="/workspace/outputs/ovfreeze_paper")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows, test_rows = build_hard_dataset(tok, seed=args.seed)

    print("\n[calibration] static BF16 inference for sigma^2_BF16")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    sigma2 = calibrate_bf16_variance(teacher, train_rows, tok,
                                     next(teacher.parameters()).device,
                                     PROJ_ORDER + FFN_PROJ)
    print("[calibration] %d layers calibrated" % len(sigma2))
    del teacher
    torch.cuda.empty_cache()

    plans = []
    if args.ablation in ("coverage", "all"):
        for name, suffixes in COVERAGE_ARMS.items():
            plans.append(("cov_%s" % name, suffixes, args.window, "ema"))
    if args.ablation in ("window", "all"):
        for w in WINDOW_ARMS:
            plans.append(("win_%02d" % int(w * 100), COVERAGE_ARMS["qvko"], w, "ema"))
    if args.ablation in ("estimator", "all"):
        for est in ESTIMATOR_ARMS:
            plans.append(("est_%s" % est, COVERAGE_ARMS["qvko"], args.window, est))

    results = [run_arm(tag, sfx, win, est, train_rows, test_rows, tok, sigma2, args)
               for tag, sfx, win, est in plans]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / ("ovfreeze_%s.json" % args.ablation)).write_text(json.dumps(
        {"ablation": args.ablation, "lambda": args.lam, "rho": args.rho,
         "c_min": args.c_min, "c_max": args.c_max,
         "n_train": len(train_rows), "n_test": len(test_rows), "arms": results},
        indent=2))

    print("\n" + "=" * 76)
    print("  SUMMARY -- %s ablation" % args.ablation)
    print("=" * 76)
    print("  %-14s %4s %5s %7s %8s %8s %9s %8s"
          % ("arm", "cov", "win", "est", "bf16 F1", "int4 F1", "drift%", "pplΔ"))
    for r in results:
        i4 = r["int4"]["f1"] if r["int4"] else float("nan")
        d = r["drift_pct"] if r["drift_pct"] is not None else float("nan")
        print("  %-14s %4d %4.0f%% %7s %8.4f %8.4f %+9.2f %8s"
              % (r["arm"], r["n_covered"], r["window"] * 100, r["estimator"],
                 r["bf16"]["f1"], i4, d, str(r["ppl_fluctuation"])))

    print("\n  paper reference:")
    print("    coverage  drift +18.2% -> +1.3%,  F1 0.916 -> 0.923, FFN adds nothing")
    print("    window    <=20% F1<=0.921,  30% best,  >=50% F1 0.918")
    print("    estimator EMA pplΔ +0.18,  batch +1.4,  global -0.4 F1")
    print("\n  results -> %s" % (out / ("ovfreeze_%s.json" % args.ablation)))


if __name__ == "__main__":
    main()
