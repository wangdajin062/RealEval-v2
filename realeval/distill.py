"""realeval/distill.py — Distillation Utilities

Knowledge distillation utilities for QAD training.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("distill")


def kl_divergence(student_logits, teacher_logits, temperature=1.0):
    """Compute KL divergence between student and teacher logits."""
    import torch
    import torch.nn.functional as F
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
