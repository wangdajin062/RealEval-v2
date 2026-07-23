"""
verify_paper_alignment.py — v2.0
论文数据集抓取、模型参数核对与模拟运行结果验证
=====================================================
对照 paper_v2.tex，验证 backend/ml/ 中的参数与论文一致性，
并下载三个测试数据集到 database/ 文件夹。

三个数据集:
  1. TAF-28k    — HuggingFace: JimmyMa99/TeleAntiFraud (公开)
  2. AdvFraud-3k — 论文自建对抗集 (非公开)
  3. ChiFraud    — GitHub: xuemingxxx/ChiFraud (公开)

用法:
  python scripts/verify_paper_alignment.py [--download] [--run-sim]
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# 将项目根目录和 backend 目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_paper")

# ============================================================
# 1. 论文数据（从 paper_v2.tex 提取）
# ============================================================
PAPER_VALUES = {
    # 主实验结果 (Table 3 / tab3)
    "taf28k": {
        "bf16_f1":          0.931,
        "bf16_precision":   0.928,
        "bf16_recall":      0.934,
        "bf16_fpr":         0.016,
        "nvfp4_qad_ovf_f1": 0.923,
        "nvfp4_qad_ovf_precision": 0.925,
        "nvfp4_qad_ovf_recall":    0.921,
        "nvfp4_qad_ovf_fpr":       0.018,
        "nvfp4_qad_ovf_recovery":  0.991,
        "nvfp4_qad_f1":     0.916,
        "nvfp4_qad_recovery": 0.984,
        "nvfp4_ptq_f1":     0.838,
        "nvfp4_qat_f1":     0.844,
        "bitdistiller_f1":  0.858,
        "awq_f1":           0.838,
        "gptq_f1":          0.840,
        "q4km_qad_ovf_f1":  0.917,
        "q4km_qad_ovf_recovery": 0.985,
        "q4km_qad_f1":      0.911,
        "q4km_ptq_f1":      0.851,
        "bert_fraud_f1":    0.876,
        "safe_qaq_f1":      0.918,
    },
    # PPL 值 (论文 Table IV 对应)
    "ppl": {
        "fp16":      8.43,
        "int4_ptq":  9.42,
        "int4_qad":  8.73,
        "int4_ov":   8.62,
    },
    # 跨数据集 (Table 4 / tab4)
    "cross_dataset": {
        "taf28k_bf16":   0.931,
        "taf28k_ovf":    0.923,
        "advfraud_bf16": 0.882,
        "advfraud_ovf":  0.875,
        "chifraud_bf16": 0.871,
        "chifraud_ovf":  0.860,
        "multisource_bf16": 0.889,
        "multisource_ovf":  0.881,
    },
    # 推测解码 (Table speculative_decoding)
    "spec_decoding": {
        "alpha_generic":    0.78,
        "alpha_tuned":      0.86,
        "gamma":            5,
        "speedup_h100":     3.49,
        "speedup_sd8g3":    3.32,
        "tokens_per_sec":   21.4,
    },
    # 融合权重
    "fusion_weights": {
        "w_text":  0.40,
        "w_audio": 0.30,
        "w_url":   0.20,
        "w_meta":  0.10,
    },
    # OV-Freeze 消融
    "ov_freeze": {
        "qad_baseline_f1":       0.916,
        "qad_baseline_ppl":      8.73,
        "qad_baseline_drift":    18.2,  # %
        "ovf_qkvo_f1":           0.923,
        "ovf_qkvo_ppl":          8.62,
        "ovf_qkvo_drift":        1.3,   # %
        "ema_095_f1":            0.923,
        "ema_095_ppl_amp":       0.18,
        "batch_level_f1":        0.918,
        "batch_level_ppl_amp":   1.4,
        "step_30_f1":            0.923,
        "step_30_ppl":           8.62,
    },
    # 损失函数消融 (Table 5)
    "loss_ablation": {
        "pure_kl_f1":         0.916,
        "pure_kl_kl_div":     0.005,
        "pure_kl_recovery":   98.4,
        "mse_f1":             0.901,
        "mse_kl_div":         0.082,
        "ce_f1":              0.844,
        "ce_kl_div":          0.311,
        "hybrid_f1":          0.879,
        "hybrid_kl_div":      0.124,
    },
    # 教师模型消融 (Table 6)
    "teacher_ablation": {
        "05b_fixed_f1":     0.916,
        "05b_conv_f1":      0.916,
        "05b_conv_tokens":  0.5,  # B
        "05b_conv_recovery": 98.4,
        "7b_fixed_f1":      0.892,
        "7b_conv_f1":       0.915,
        "7b_conv_tokens":   2.0,  # B
    },
    # 模型架构
    "model_arch": {
        "backbone":       "Qwen2.5-0.5B-Instruct",
        "params_million": 494,
        "size_fp16_mb":   960,
        "size_int4_mb":   240,
        "hidden_dim":     896,
        "n_layers":       24,
        "attn_heads_q":   14,
        "attn_heads_kv":  2,
        "ffn_dim":        4864,
        "vocab_size":     151936,
    },
    # 声学特征
    "acoustic": {
        "mfcc_dim":      64,
        "embedding_dim": 128,
        "sample_rate":   16000,
        "n_mels":        64,
        "hop_length_ms": 10,
        "n_fft_ms":      25,
    },
    # 隐私
    "privacy": {
        "glo_wer":               0.95,
        "blackbox_wer":          0.97,
        "speaker_id":            8.3,   # %
        "random_baseline_sid":   10.0,  # %
        "pesq_glo":              1.21,
        "mos_glo":               1.18,
    },
}

# ============================================================
# 2. 从 backend/ml/ 读取真实参数
# ============================================================
def load_ml_parameters() -> dict:
    """从真实生产代码中读取参数"""
    try:
        from ml.qad_pipeline import QADConfig
        qad_cfg = QADConfig()
    except Exception as e:
        logger.error("Failed to load QADConfig: %s", e)
        qad_cfg = None

    try:
        from ml.speculative_decoder import (
            ALPHA_TUNED, GAMMA, STUDENT_ARCH,
        )
    except Exception as e:
        logger.error("Failed to load speculative_decoder: %s", e)
        ALPHA_TUNED = GAMMA = None
        STUDENT_ARCH = {}

    try:
        from ml.multimodal_detector import (
            W_TEXT, W_AUDIO, W_URL, W_META,
            FUSION_BIAS, FUSION_SCALE,
        )
    except Exception as e:
        logger.error("Failed to load multimodal_detector: %s", e)
        W_TEXT = W_AUDIO = W_URL = W_META = None
        FUSION_BIAS = FUSION_SCALE = None

    try:
        from ml.acoustic_embedding import (
            MFCC_DIM, EMBEDDING_DIM, SAMPLE_RATE,
            N_MELS, HOP_LENGTH, N_FFT, SENSITIVITY_L2,
        )
    except Exception as e:
        logger.error("Failed to load acoustic_embedding: %s", e)
        MFCC_DIM = EMBEDDING_DIM = SAMPLE_RATE = None
        N_MELS = HOP_LENGTH = N_FFT = SENSITIVITY_L2 = None

    return {
        "qad_config": {
            "alpha":       qad_cfg.alpha if qad_cfg else None,
            "beta":        qad_cfg.beta if qad_cfg else None,
            "gamma_coeff": qad_cfg.gamma_coeff if qad_cfg else None,
            "temperature": qad_cfg.temperature if qad_cfg else None,
            "bits":        qad_cfg.bits if qad_cfg else None,
            "group_size":  qad_cfg.group_size if qad_cfg else None,
            "quant_scheme": qad_cfg.quant_scheme if qad_cfg else None,
            "freeze_ov":   qad_cfg.freeze_ov if qad_cfg else None,
            "ov_freeze_ratio": qad_cfg.ov_freeze_ratio if qad_cfg else None,
            "learning_rate": qad_cfg.learning_rate if qad_cfg else None,
            "batch_size":    qad_cfg.batch_size if qad_cfg else None,
            "max_steps":     qad_cfg.max_steps if qad_cfg else None,
            # PPL 指标
            "fp16_ppl":      qad_cfg.fp16_ppl if qad_cfg else None,
            "int4_ptq_ppl":  qad_cfg.int4_ptq_ppl if qad_cfg else None,
            "int4_qad_ppl":  qad_cfg.int4_qad_ppl if qad_cfg else None,
            "int4_ov_ppl":   qad_cfg.int4_ov_ppl if qad_cfg else None,
            "fp16_size_mb":  qad_cfg.fp16_size_mb if qad_cfg else None,
            "int4_size_mb":  qad_cfg.int4_size_mb if qad_cfg else None,
            "tokens_per_sec": qad_cfg.tokens_per_sec_sd8g3 if qad_cfg else None,
        },
        "spec_decoding": {
            "ALPHA_TUNED": ALPHA_TUNED,
            "GAMMA":       GAMMA,
        },
        "student_arch": STUDENT_ARCH,
        "fusion": {
            "W_TEXT":       W_TEXT,
            "W_AUDIO":      W_AUDIO,
            "W_URL":        W_URL,
            "W_META":       W_META,
            "FUSION_BIAS":  FUSION_BIAS,
            "FUSION_SCALE": FUSION_SCALE,
        },
        "acoustic": {
            "MFCC_DIM":      MFCC_DIM,
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "SAMPLE_RATE":   SAMPLE_RATE,
            "N_MELS":        N_MELS,
            "HOP_LENGTH":    HOP_LENGTH,
            "N_FFT":         N_FFT,
            "SENSITIVITY_L2": SENSITIVITY_L2,
        },
    }

# ============================================================
# 3. 参数一致性校验
# ============================================================
@dataclass
class AlignmentCheck:
    name:        str
    paper_value: float | str | None
    code_value:  float | str | None
    tolerance:   float = 0.01
    status:      str = "pending"

    def verify(self) -> str:
        if self.paper_value is None or self.code_value is None:
            self.status = "⚠️ MISSING"
            return self.status

        if isinstance(self.paper_value, str):
            if str(self.paper_value) == str(self.code_value):
                self.status = "✅ MATCH"
            else:
                self.status = f"❌ MISMATCH: paper={self.paper_value}, code={self.code_value}"
            return self.status

        diff = abs(float(self.paper_value) - float(self.code_value))
        if diff <= self.tolerance:
            self.status = "✅ MATCH"
        elif diff <= self.tolerance * 5:
            self.status = f"⚠️ CLOSE (diff={diff:.4f})"
        else:
            self.status = f"❌ MISMATCH (diff={diff:.4f})"
        return self.status


def run_parameter_alignment() -> list[AlignmentCheck]:
    """运行全部参数一致性校验"""
    ml_params = load_ml_parameters()
    checks = []

    # ── QAD 配置 ──
    qad = ml_params["qad_config"]
    # QADConfig.alpha vs 论文公式 (1) 中的 α
    checks.append(AlignmentCheck(
        "QAD α (L_task 系数)", 0.4, qad["alpha"]))
    checks.append(AlignmentCheck(
        "QAD β (L_KD 系数)", 0.5, qad["beta"]))
    checks.append(AlignmentCheck(
        "QAD γ (L_quant 系数)", 0.1, qad["gamma_coeff"]))
    checks.append(AlignmentCheck(
        "QAD τ (蒸馏温度)", 3.0, qad["temperature"]))
    # PPL 指标
    checks.append(AlignmentCheck(
        "PPL FP16 Student", PAPER_VALUES["ppl"]["fp16"],
        qad["fp16_ppl"]))
    checks.append(AlignmentCheck(
        "PPL INT4 PTQ", PAPER_VALUES["ppl"]["int4_ptq"],
        qad["int4_ptq_ppl"]))
    checks.append(AlignmentCheck(
        "PPL INT4 QAD", PAPER_VALUES["ppl"]["int4_qad"],
        qad["int4_qad_ppl"]))
    checks.append(AlignmentCheck(
        "PPL INT4 QAD+OVF", PAPER_VALUES["ppl"]["int4_ov"],
        qad["int4_ov_ppl"]))
    # 模型尺寸
    checks.append(AlignmentCheck(
        "FP16 模型体积 (MB)", PAPER_VALUES["model_arch"]["size_fp16_mb"],
        qad["fp16_size_mb"]))
    checks.append(AlignmentCheck(
        "INT4 模型体积 (MB)", PAPER_VALUES["model_arch"]["size_int4_mb"],
        qad["int4_size_mb"]))
    # 量化方案
    checks.append(AlignmentCheck(
        "量化方案", "Q4_K_M", qad["quant_scheme"]))
    # OV-Freeze
    checks.append(AlignmentCheck(
        "OV-Freeze 激活比例", 0.30, qad["ov_freeze_ratio"]))
    # 训练配置
    checks.append(AlignmentCheck(
        "训练批大小", 8, qad["batch_size"]))
    checks.append(AlignmentCheck(
        "最大训练步数", 2000, qad["max_steps"]))
    # 吞吐量
    checks.append(AlignmentCheck(
        "SD8G3 吞吐 (tok/s)", PAPER_VALUES["spec_decoding"]["tokens_per_sec"],
        qad["tokens_per_sec"]))

    # ── 推测解码 ──
    spec = ml_params["spec_decoding"]
    checks.append(AlignmentCheck(
        "α (领域调优接受率)", PAPER_VALUES["spec_decoding"]["alpha_tuned"],
        spec["ALPHA_TUNED"]))
    checks.append(AlignmentCheck(
        "γ (推测窗口)", PAPER_VALUES["spec_decoding"]["gamma"],
        spec["GAMMA"]))

    # ── 学生模型架构 ──
    arch = ml_params["student_arch"]
    if arch:
        checks.append(AlignmentCheck(
            "骨干网络", PAPER_VALUES["model_arch"]["backbone"],
            arch.get("backbone", "")))
        checks.append(AlignmentCheck(
            "参数规模 (M)", PAPER_VALUES["model_arch"]["params_million"],
            arch.get("params_fp16_M", 0)))
        checks.append(AlignmentCheck(
            "隐藏维度", PAPER_VALUES["model_arch"]["hidden_dim"],
            arch.get("hidden_dim", 0)))
        checks.append(AlignmentCheck(
            "Transformer 层数", PAPER_VALUES["model_arch"]["n_layers"],
            arch.get("n_layers", 0)))
        checks.append(AlignmentCheck(
            "注意力头数 Q", PAPER_VALUES["model_arch"]["attn_heads_q"],
            arch.get("attn_heads_Q", 0)))
        checks.append(AlignmentCheck(
            "注意力头数 KV", PAPER_VALUES["model_arch"]["attn_heads_kv"],
            arch.get("attn_heads_KV", 0)))
        checks.append(AlignmentCheck(
            "FFN 维度", PAPER_VALUES["model_arch"]["ffn_dim"],
            arch.get("ffn_dim", 0)))
        checks.append(AlignmentCheck(
            "词表大小", PAPER_VALUES["model_arch"]["vocab_size"],
            arch.get("vocab_size", 0)))

    # ── 融合权重 ──
    fusion = ml_params["fusion"]
    checks.append(AlignmentCheck(
        "W_TEXT", PAPER_VALUES["fusion_weights"]["w_text"],
        fusion["W_TEXT"]))
    checks.append(AlignmentCheck(
        "W_AUDIO", PAPER_VALUES["fusion_weights"]["w_audio"],
        fusion["W_AUDIO"]))
    checks.append(AlignmentCheck(
        "W_URL", PAPER_VALUES["fusion_weights"]["w_url"],
        fusion["W_URL"]))
    checks.append(AlignmentCheck(
        "W_META", PAPER_VALUES["fusion_weights"]["w_meta"],
        fusion["W_META"]))

    # ── 声学特征 ──
    ac = ml_params["acoustic"]
    checks.append(AlignmentCheck(
        "MFCC 维度", PAPER_VALUES["acoustic"]["mfcc_dim"],
        ac["MFCC_DIM"]))
    checks.append(AlignmentCheck(
        "嵌入总维度", PAPER_VALUES["acoustic"]["embedding_dim"],
        ac["EMBEDDING_DIM"]))
    checks.append(AlignmentCheck(
        "采样率", PAPER_VALUES["acoustic"]["sample_rate"],
        ac["SAMPLE_RATE"]))
    checks.append(AlignmentCheck(
        "Mel 滤波器组数", PAPER_VALUES["acoustic"]["n_mels"],
        ac["N_MELS"]))
    checks.append(AlignmentCheck(
        "帧移 (samples)", 160, ac["HOP_LENGTH"]))  # 10ms @ 16kHz = 160
    checks.append(AlignmentCheck(
        "FFT 窗长 (samples)", 400, ac["N_FFT"]))   # 25ms @ 16kHz = 400

    # 执行验证
    for c in checks:
        c.verify()
    return checks


# ============================================================
# 4. 数据集下载
# ============================================================
def download_taf28k(output_dir: Path) -> bool:
    """从 HuggingFace 下载 TAF-28k 数据集"""
    logger.info("=" * 60)
    logger.info("📥 下载 TAF-28k (HuggingFace: JimmyMa99/TeleAntiFraud)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset

        for split in ["train", "test"]:
            logger.info("加载 split=%s ...", split)
            ds = load_dataset("JimmyMa99/TeleAntiFraud", split=split, streaming=False)
            n = len(ds)
            logger.info("  split=%s: %d 条", split, n)

            # 保存为 JSONL
            out_path = output_dir / f"TAF28k_{split}.jsonl"
            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in ds:
                    f.write(json.dumps(dict(sample), ensure_ascii=False) + "\n")
                    count += 1
            logger.info("  已保存 %d 条 → %s", count, out_path)

        # 同时保存元数据
        meta = {
            "dataset": "TeleAntiFraud-28k",
            "source": "HuggingFace: JimmyMa99/TeleAntiFraud",
            "paper": "arXiv:2503.24115",
            "citation": "Ma et al., ACM MM 2025",
            "total_samples": 28511,
            "audio_hours": 307,
            "tasks": ["scene_classification", "fraud_detection", "fraud_type_classification"],
            "splits": {"train": "8/10", "val": "1/10", "test": "1/10"},
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_dir / "TAF28k_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("✅ TAF-28k 下载完成")
        return True

    except Exception as e:
        logger.error("❌ TAF-28k 下载失败: %s", e)
        # 创建占位元数据
        meta = {
            "dataset": "TeleAntiFraud-28k",
            "source": "HuggingFace: JimmyMa99/TeleAntiFraud",
            "status": f"DOWNLOAD_FAILED: {e}",
            "note": "如需手动下载: huggingface-cli download JimmyMa99/TeleAntiFraud",
        }
        with open(output_dir / "TAF28k_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return False


def download_chifraud(output_dir: Path) -> bool:
    """从 GitHub 克隆 ChiFraud 数据集"""
    logger.info("=" * 60)
    logger.info("📥 下载 ChiFraud (GitHub: xuemingxxx/ChiFraud)")
    logger.info("=" * 60)

    chifraud_dir = output_dir / "ChiFraud"
    if chifraud_dir.exists():
        logger.info("ChiFraud 目录已存在: %s", chifraud_dir)
        # 列出内容
        for item in sorted(chifraud_dir.iterdir()):
            logger.info("  %s", item.name)
        return True

    try:
        import subprocess
        result = subprocess.run(
            ["git", "clone", "https://github.com/xuemingxxx/ChiFraud.git",
             str(chifraud_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("✅ ChiFraud 克隆成功")
            return True
        else:
            logger.warning("⚠️ git clone 失败: %s", result.stderr)
    except Exception as e:
        logger.warning("⚠️ git clone 异常: %s", e)

    # 创建元数据文件
    meta = {
        "dataset": "ChiFraud (SmsSpam-CN)",
        "source": "GitHub: xuemingxxx/ChiFraud",
        "paper": "Tang et al., COLING 2025",
        "citation": "ACL Anthology: 2025.coling-main.398",
        "total_texts": 411934,
        "fraud_texts": 59106,
        "normal_texts": 352328,
        "fraud_categories": 11,
        "collection_period": "2022-2023 (12 months)",
        "status": "METADATA_ONLY (git clone unavailable, see README)",
        "note": "完整数据集需从 GitHub 下载: git clone https://github.com/xuemingxxx/ChiFraud.git",
    }
    with open(output_dir / "ChiFraud_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return False


def create_advfraud3k_documentation(output_dir: Path):
    """AdvFraud-3k 是非公开数据集，创建文档说明"""
    logger.info("=" * 60)
    logger.info("📝 记录 AdvFraud-3k (非公开自建数据集)")
    logger.info("=" * 60)

    doc = {
        "dataset": "AdvFraud-3k",
        "availability": "NON-PUBLIC (in-house adversarial dataset)",
        "description": (
            "通过对 TAF-28k 测试集中的 1,000 条真实欺诈文本样本进行人工对抗式改写，"
            "引入同义词扰动、句式拓扑重排、方言特征转换及隐喻表达等 8 种典型对抗策略；"
            "并由领域专家撰写 2,000 条新型欺诈话术。"
        ),
        "construction": {
            "source_samples": 1000,  # from TAF-28k test
            "newly_written": 2000,  # by domain experts
            "annotators": 3,  # independent
            "reviewers": 1,  # senior
            "inter_annotator_agreement": "Cohen's κ = 0.87",
            "adversarial_strategies": [
                "同义词扰动 (Synonym Perturbation)",
                "句式拓扑重排 (Syntactic Reordering)",
                "方言特征转换 (Dialect Feature Transformation)",
                "隐喻表达 (Metaphorical Expression)",
                "以及另外 4 种策略",
            ],
        },
        "evaluation_only": True,
        "note": (
            "该数据集仅用于测试评估（不参与训练），由所在机构数据合规审查。"
            "如需复现 AdvFraud-3k 上的结果，请参考论文 §4.1 的构建方法自行构建。"
        ),
        "paper_results": {
            "bf16_f1": 0.882,
            "nvfp4_qad_ovf_f1": 0.875,
            "absolute_drop_vs_bf16": 0.007,
            "relative_drop_vs_taf28k_bf16": 0.052,  # 5.2%
            "within_threat_constraint": "C₂(≤6%) ✅",
        },
    }
    with open(output_dir / "AdvFraud3k_metadata.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info("✅ AdvFraud-3k 文档已创建")


# ============================================================
# 5. 模拟运行验证
# ============================================================
def simulate_qad_pipeline() -> dict:
    """模拟 QAD 流水线并验证 PPL 和 F1"""
    logger.info("=" * 60)
    logger.info("🧪 模拟 QAD 流水线运行验证")
    logger.info("=" * 60)

    try:
        from ml.qad_pipeline import QADPipeline, QADConfig
    except Exception as e:
        logger.error("无法加载 QADPipeline: %s", e)
        return {"status": "FAILED", "error": str(e)}

    cfg = QADConfig()
    pipeline = QADPipeline(cfg)

    # 用模拟欺诈文本运行蒸馏
    fraud_texts = [
        "您的账户涉嫌洗钱，请立即转账到安全账户",
        "公安局通知：您涉及洗钱案件，配合调查",
        "恭喜中奖！点击链接领取奖品",
        "刷单兼职日入500元，联系客服报名",
        "您的贷款已审批，缴纳保证金即可放款",
    ] * 20  # 100 条

    result = pipeline.run_distillation(fraud_texts, max_steps=200)
    logger.info("流水线完成: %s", json.dumps(result, indent=2, ensure_ascii=False))

    # 验证
    checks = {}
    # PPL 检查
    for key, paper_val in PAPER_VALUES["ppl"].items():
        code_key = {
            "fp16": "fp16_ppl",
            "int4_ptq": "int4_ptq_ppl",
            "int4_qad": "int4_qad_ppl",
            "int4_ov": "int4_ov_ppl",
        }.get(key, "")
        code_val = result.get(code_key, None)
        if code_val is not None:
            diff = abs(code_val - paper_val)
            checks[f"PPL_{key}"] = {
                "paper": paper_val,
                "code": code_val,
                "diff": diff,
                "match": diff < 0.01,
            }

    result["ppl_checks"] = checks
    return result


def simulate_spec_decoding() -> dict:
    """模拟推测解码并验证加速比"""
    logger.info("=" * 60)
    logger.info("🧪 模拟推测解码验证")
    logger.info("=" * 60)

    try:
        from ml.speculative_decoder import (
            SpeculativeDecoder, ALPHA_TUNED, GAMMA, STUDENT_ARCH,
        )
    except Exception as e:
        logger.error("无法加载 SpeculativeDecoder: %s", e)
        return {"status": "FAILED", "error": str(e)}

    decoder = SpeculativeDecoder()
    stats = decoder.stats

    # 模拟接受率计算
    # 论文公式: speedup = (1 - α^(γ+1)) / (1 - α)
    alpha = ALPHA_TUNED
    gamma = GAMMA
    theoretical = (1 - alpha**(gamma+1)) / (1 - alpha)

    results = {
        "alpha": alpha,
        "gamma": gamma,
        "theoretical_speedup": round(theoretical, 2),
        "paper_theoretical_speedup": 4.25,
        "paper_h100_measured": PAPER_VALUES["spec_decoding"]["speedup_h100"],
        "paper_sd8g3_measured": PAPER_VALUES["spec_decoding"]["speedup_sd8g3"],
        "generic_alpha": PAPER_VALUES["spec_decoding"]["alpha_generic"],
        "generic_theoretical": round(
            (1 - 0.78**(gamma+1)) / (1 - 0.78), 2
        ),
        "student_arch": STUDENT_ARCH,
    }

    # 验证理论与论文一致性
    theo_diff = abs(theoretical - 4.25)
    results["theoretical_match"] = theo_diff < 0.01

    logger.info("推测解码验证: %s", json.dumps(results, indent=2, ensure_ascii=False))
    return results


def simulate_multimodal_fusion() -> dict:
    """模拟多模态融合并验证权重"""
    logger.info("=" * 60)
    logger.info("🧪 模拟多模态融合验证")
    logger.info("=" * 60)

    try:
        from ml.multimodal_detector import (
            W_TEXT, W_AUDIO, W_URL, W_META, MultimodalDetector,
        )
    except Exception as e:
        logger.error("无法加载 MultimodalDetector: %s", e)
        return {"status": "FAILED", "error": str(e)}

    detector = MultimodalDetector()

    results = {
        "fusion_weights": {
            "text":  W_TEXT,
            "audio": W_AUDIO,
            "url":   W_URL,
            "meta":  W_META,
        },
        "paper_weights": PAPER_VALUES["fusion_weights"],
        "checks": {}
    }

    for key in ["text", "audio", "url", "meta"]:
        code_val = results["fusion_weights"][key]
        paper_val = results["paper_weights"][f"w_{key}"]
        results["checks"][key] = abs(code_val - paper_val) < 0.001

    logger.info("融合验证: %s", json.dumps(results, indent=2, ensure_ascii=False))
    return results


# ============================================================
# 6. 生成对齐报告
# ============================================================
def generate_report(
    checks: list[AlignmentCheck],
    qad_result: dict,
    spec_result: dict,
    fusion_result: dict,
    download_status: dict,
    output_dir: Path,
) -> Path:
    """生成完整的对齐报告"""
    report_path = output_dir / "paper_alignment_report.md"

    lines = []
    lines.append("# 📊 QAD-MultiGuard 论文对齐验证报告")
    lines.append(f"\n> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 论文文件: `docs/paper_v2 .tex`")
    lines.append(f"> 代码路径: `backend/ml/`\n")

    # ── 数据集状态 ──
    lines.append("## 1. 数据集下载状态\n")
    lines.append("| 数据集 | 状态 | 来源 | 样本数 |")
    lines.append("|--------|------|------|--------|")
    for ds_name, status in download_status.items():
        ds_icon = "✅" if status.get("success") else ("⚠️" if "partial" in str(status) else "❌")
        lines.append(f"| {ds_icon} {ds_name} | {status.get('status', 'unknown')} | {status.get('source', 'N/A')} | {status.get('samples', 'N/A')} |")
    lines.append("")

    # ── 参数对齐 ──
    lines.append("## 2. 参数对齐检查\n")
    lines.append("| # | 参数 | 论文值 | 代码值 | 状态 |")
    lines.append("|---|------|--------|--------|------|")

    match_count = 0
    mismatch_count = 0
    close_count = 0
    missing_count = 0

    for i, c in enumerate(checks, 1):
        paper_str = str(c.paper_value) if c.paper_value is not None else "N/A"
        code_str = str(c.code_value) if c.code_value is not None else "N/A"
        lines.append(f"| {i} | {c.name} | {paper_str} | {code_str} | {c.status} |")

        if "MATCH" in c.status:
            match_count += 1
        elif "MISMATCH" in c.status:
            mismatch_count += 1
        elif "CLOSE" in c.status:
            close_count += 1
        else:
            missing_count += 1

    lines.append("")
    total = len(checks)
    lines.append(f"**总计**: {total} 项检查")
    lines.append(f"- ✅ 匹配: {match_count} ({100*match_count/total:.1f}%)")
    lines.append(f"- ⚠️ 接近: {close_count} ({100*close_count/total:.1f}%)")
    lines.append(f"- ❌ 不匹配: {mismatch_count} ({100*mismatch_count/total:.1f}%)")
    lines.append(f"- ⚠️ 缺失: {missing_count} ({100*missing_count/total:.1f}%)\n")

    # ── PPL 验证 ──
    lines.append("## 3. PPL 模拟验证\n")
    if "ppl_checks" in qad_result:
        lines.append("| 指标 | 论文值 | 代码值 | 偏差 | 状态 |")
        lines.append("|------|--------|--------|------|------|")
        for key, info in qad_result["ppl_checks"].items():
            icon = "✅" if info["match"] else "❌"
            lines.append(f"| {key} | {info['paper']} | {info['code']} | {info['diff']:.4f} | {icon} |")
    lines.append("")

    # ── 推测解码 ──
    lines.append("## 4. 推测解码验证\n")
    lines.append(f"- α (领域调优): `{spec_result.get('alpha', 'N/A')}` (论文: 0.86)")
    lines.append(f"- γ (推测窗口): `{spec_result.get('gamma', 'N/A')}` (论文: 5)")
    lines.append(f"- 理论加速比: `{spec_result.get('theoretical_speedup', 'N/A')}×` (论文: 4.25×)")
    lines.append(f"- 理论值匹配: {'✅' if spec_result.get('theoretical_match') else '❌'}")
    lines.append("")

    # ── 融合 ──
    lines.append("## 5. 多模态融合验证\n")
    fw = fusion_result.get("fusion_weights", {})
    checks_f = fusion_result.get("checks", {})
    lines.append(f"- W_text: `{fw.get('text')}` {'✅' if checks_f.get('text') else '❌'} (论文: 0.40)")
    lines.append(f"- W_audio: `{fw.get('audio')}` {'✅' if checks_f.get('audio') else '❌'} (论文: 0.30)")
    lines.append(f"- W_url: `{fw.get('url')}` {'✅' if checks_f.get('url') else '❌'} (论文: 0.20)")
    lines.append(f"- W_meta: `{fw.get('meta')}` {'✅' if checks_f.get('meta') else '❌'} (论文: 0.10)")
    lines.append("")

    # ── 需要 tex 修改的事项 ──
    lines.append("## 6. 需要修改 tex 的事项\n")
    lines.append("以下是需要在 paper_v2.tex 中修正的内容:\n")

    tex_issues = []
    for c in checks:
        if "MISMATCH" in c.status or "MISSING" in c.status:
            tex_issues.append(f"- **{c.name}**: {c.status}")

    if tex_issues:
        for issue in tex_issues:
            lines.append(issue)
    else:
        lines.append("✅ 无需修改 — 代码参数与论文完全一致\n")

    # ── 数据源信息 ──
    lines.append("## 7. 数据源信息\n")
    lines.append("### TAF-28k ✅")
    lines.append("- HuggingFace: `JimmyMa99/TeleAntiFraud`")
    lines.append("- ArXiv: `2503.24115`")
    lines.append("- 许可: 公开 (ACM MM 2025)")
    lines.append("")
    lines.append("### AdvFraud-3k ⚠️ (非公开)")
    lines.append("- 自建数据集，含 8 种对抗策略")
    lines.append("- 1000 条改写 + 2000 条新撰写")
    lines.append("- 构建方法见论文 §4.1")
    lines.append("")
    lines.append("### ChiFraud ✅")
    lines.append("- GitHub: `xuemingxxx/ChiFraud`")
    lines.append("- 411,934 条中文文本（59,106 欺诈 + 352,328 正常）")
    lines.append("- 许可: 公开 (COLING 2025)")
    lines.append("")

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    logger.info("📄 报告已保存: %s", report_path)
    return report_path


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="论文对齐验证工具")
    parser.add_argument("--download", action="store_true", help="下载数据集")
    parser.add_argument("--run-sim", action="store_true", help="运行模拟验证")
    parser.add_argument("--output", type=str, default="database",
                        help="输出目录 (默认: database/)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🔍 QAD-MultiGuard 论文对齐验证")
    logger.info("  项目根目录: %s", PROJECT_ROOT)
    logger.info("  输出目录:   %s", output_dir)

    # ── Step 1: 参数对齐 ──
    logger.info("\n📋 Step 1: 参数对齐检查")
    checks = run_parameter_alignment()
    for c in checks:
        c.verify()
        logger.info("  %s %s", c.status.split(" ")[0] if " " in c.status else c.status, c.name)

    # ── Step 2: 数据集下载 ──
    download_status = {}
    if args.download:
        logger.info("\n📥 Step 2: 数据集下载")
        taf_ok = download_taf28k(output_dir)
        download_status["TAF-28k"] = {
            "success": taf_ok,
            "status": "downloaded" if taf_ok else "metadata_only",
            "source": "HuggingFace: JimmyMa99/TeleAntiFraud",
            "samples": "28,511" if taf_ok else "N/A",
        }

        chi_ok = download_chifraud(output_dir)
        download_status["ChiFraud"] = {
            "success": chi_ok,
            "status": "downloaded" if chi_ok else "metadata_only (手动下载)",
            "source": "GitHub: xuemingxxx/ChiFraud",
            "samples": "411,934",
        }

        create_advfraud3k_documentation(output_dir)
        download_status["AdvFraud-3k"] = {
            "success": False,
            "status": "non_public (已创建文档)",
            "source": "自建 (论文 §4.1)",
            "samples": "3,000",
            "note": "非公开数据集，详见 AdvFraud3k_metadata.json",
        }
    else:
        download_status = {
            "TAF-28k": {"status": "skipped (使用 --download 下载)", "source": "HuggingFace"},
            "ChiFraud": {"status": "skipped", "source": "GitHub: xuemingxxx/ChiFraud"},
            "AdvFraud-3k": {"status": "non_public", "source": "自建 (论文 §4.1)"},
        }

    # ── Step 3: 模拟验证 ──
    if args.run_sim:
        logger.info("\n🧪 Step 3: 模拟运行验证")
        qad_result = simulate_qad_pipeline()
        spec_result = simulate_spec_decoding()
        fusion_result = simulate_multimodal_fusion()
    else:
        qad_result = {"status": "skipped (使用 --run-sim 运行)"}
        spec_result = {"status": "skipped"}
        fusion_result = {"status": "skipped"}

    # ── Step 4: 生成报告 ──
    logger.info("\n📄 Step 4: 生成对齐报告")
    report_path = generate_report(
        checks, qad_result, spec_result, fusion_result,
        download_status, output_dir,
    )

    # ── 汇总 ──
    match_count = sum(1 for c in checks if "MATCH" in c.status)
    total = len(checks)
    logger.info("\n" + "=" * 60)
    logger.info("✅ 验证完成!")
    logger.info("  参数匹配: %d/%d (%.1f%%)", match_count, total,
                100*match_count/total)
    logger.info("  报告路径: %s", report_path)
    if args.download:
        logger.info("  数据集目录: %s", output_dir)
    logger.info("=" * 60)

    return 0 if match_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
