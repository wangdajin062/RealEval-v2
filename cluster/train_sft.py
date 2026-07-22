#!/usr/bin/env python
"""train_sft.py — Teacher-First SFT Strategy for Fraud Detection

Phase 1: Supervised fine-tune Qwen2.5-0.5B-Instruct (BF16) on TeleAntiFraud data
         until F1 breaks zero on the test split.
Phase 2: (optional) Run QAD distillation with the fine-tuned teacher.

Usage:
  /workspace/venv/bin/python cluster/train_sft.py                     # default
  /workspace/venv/bin/python cluster/train_sft.py --epochs 3 --lr 2e-5  # custom
  /workspace/venv/bin/python cluster/train_sft.py --dry-run            # data check only
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure /workspace is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sft] %(message)s")
logger = logging.getLogger("sft")


def parse_args():
    p = argparse.ArgumentParser(description="Teacher-First SFT for Fraud Detection")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs (1-3 recommended)")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset size (None = all)")
    p.add_argument("--bucket", type=str, default="hf://buckets/wangdajin062/TeleAntiFraud-bucket",
                   help="HF bucket/dataset name")
    p.add_argument("--output-dir", type=str, default="/workspace/models/sft-teacher",
                   help="Where to save the fine-tuned model")
    p.add_argument("--dry-run", action="store_true", help="Print data and exit")
    p.add_argument("--gradient-accumulation", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = bs × ga)")
    p.add_argument("--use-lora", action="store_true", default=True,
                   help="Use LoRA for memory-efficient fine-tuning (default: True)")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load data from HF bucket ──
    from realeval.data import prepare_sft_dataset
    logger.info("=== Loading data from %s ===", args.bucket)
    (train_texts, train_labels), (test_texts, test_labels) = prepare_sft_dataset(
        args.bucket, max_samples=args.max_samples)

    n_fraud_train = sum(train_labels)
    n_normal_train = len(train_labels) - n_fraud_train
    n_fraud_test = sum(test_labels)
    n_normal_test = len(test_labels) - n_fraud_test
    logger.info("Train: %d total (%d fraud / %d normal)", len(train_labels), n_fraud_train, n_normal_train)
    logger.info("Test:  %d total (%d fraud / %d normal)", len(test_labels), n_fraud_test, n_normal_test)

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    # ── 2. Load teacher model ──
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, TaskType

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info("=== Loading teacher: %s (BF16) ===", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto")

    # ── 3. Apply LoRA (memory-efficient SFT) ──
    if args.use_lora:
        logger.info("=== Applying LoRA (r=%d) ===", args.lora_r)
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=32, lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── 4. Tokenize dataset for causal LM ──
    PROMPT_TEMPLATE = (
        "Determine if the following text is fraud or normal.\n\n"
        "Text: {text}\n\nAnswer: {label}"
    )

    def tokenize_fn(examples):
        """Build (input_ids, labels) with an exact prompt/answer boundary.

        The previous implementation tokenised the prompt and the full text separately and
        masked `range(prompt_len)`. For this template both tokenise to the same length —
        the trailing space in "Answer: " is its own token and " fraud" occupies that same
        position — so the mask covered the answer too and every label became -100
        (loss 0.0, grad_norm 0.0, eval_loss nan).

        Here the prompt and the answer are tokenised independently with
        add_special_tokens=False and concatenated, so the boundary is known by
        construction and cannot drift with the tokeniser's merge rules.
        """
        PROMPT = ("Determine if the following text is fraud or normal.\n\n"
                  "Text: {text}\n\nAnswer:")
        input_ids_list, labels_list, attn_list = [], [], []

        for t, l in zip(examples["text"], examples["label"]):
            answer = " fraud" if l == 1 else " normal"
            p_ids = tokenizer(PROMPT.format(text=t), add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                a_ids = a_ids + [tokenizer.eos_token_id]

            # Truncate the PROMPT (never the answer) so supervision always survives.
            budget = 256 - len(a_ids)
            if len(p_ids) > budget:
                p_ids = p_ids[:budget]

            ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids[:]      # supervise the answer only
            input_ids_list.append(ids)
            labels_list.append(labels)
            attn_list.append([1] * len(ids))

        # Right-pad the batch; padding positions are masked out of the loss.
        maxlen = max(len(x) for x in input_ids_list)
        pad_id = tokenizer.pad_token_id
        for i in range(len(input_ids_list)):
            gap = maxlen - len(input_ids_list[i])
            input_ids_list[i] += [pad_id] * gap
            labels_list[i]    += [-100] * gap
            attn_list[i]      += [0] * gap

        n_sup = sum(1 for row in labels_list for x in row if x != -100)
        if n_sup == 0:
            raise RuntimeError(
                "tokenize_fn produced zero supervised tokens; training would report "
                "loss=0.0 and learn nothing. Check the prompt/answer construction.")

        return {"input_ids": input_ids_list, "labels": labels_list,
                "attention_mask": attn_list}

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_dict({"text": train_texts, "label": train_labels})
    test_ds = HFDataset.from_dict({"text": test_texts, "label": test_labels})

    logger.info("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=args.batch_size * 2,
                            remove_columns=["text", "label"])
    test_ds = test_ds.map(tokenize_fn, batched=True, batch_size=args.batch_size * 2,
                          remove_columns=["text", "label"])

    # ── 5. Train ──
    training_args = TrainingArguments(
        output_dir="/workspace/outputs/sft_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=None,  # tokenize_fn already pads; re-padding would double-pad
    )

    logger.info("=== Starting SFT training (%d epochs, lr=%s, bs=%d×%d) ===",
                args.epochs, args.lr, args.batch_size, args.gradient_accumulation)
    trainer.train()

    # ── 6. Evaluate on test set ──
    logger.info("=== Evaluating on test set ===")
    eval_results = trainer.evaluate()
    logger.info("Eval results: %s", eval_results)

    # ── 7. Save fine-tuned teacher ──
    logger.info("=== Saving fine-tuned teacher to %s ===", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("✅ SFT complete! Teacher saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
