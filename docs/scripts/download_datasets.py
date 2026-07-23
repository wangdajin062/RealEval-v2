"""
download_datasets.py — 论文三个数据集下载工具
==============================================
支持多个下载源（HuggingFace / ModelScope / GitHub / hf-mirror）

数据集:
  1. TAF-28k     — HuggingFace: JimmyMa99/TeleAntiFraud, ModelScope 镜像
  2. ChiFraud     — GitHub: xuemingxxx/ChiFraud (仅文本，可直接 clone)
  3. AdvFraud-3k  — 非公开（自建数据集，此处生成构建指南）

用法:
  python scripts/download_datasets.py [--source hf|ms|mirror] [--dataset all|taf28k|chifraud]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_datasets")

DATABASE_DIR = PROJECT_ROOT / "database"


# ============================================================
# TAF-28k 下载
# ============================================================
def download_taf28k_hf(output_dir: Path) -> bool:
    """从 HuggingFace 下载 (需要外网访问)"""
    try:
        from datasets import load_dataset
        for split in ["train", "test"]:
            ds = load_dataset("JimmyMa99/TeleAntiFraud", split=split, streaming=False)
            out = output_dir / f"TAF28k_{split}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for s in ds:
                    f.write(json.dumps(dict(s), ensure_ascii=False) + "\n")
            logger.info("  ✅ %s: %d 条 → %s", split, len(ds), out)
        return True
    except Exception as e:
        logger.warning("HuggingFace 下载失败: %s", e)
        return False


def download_taf28k_mirror(output_dir: Path) -> bool:
    """从 hf-mirror.com 下载 (国内镜像)"""
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        return download_taf28k_hf(output_dir)
    except Exception as e:
        logger.warning("hf-mirror 下载失败: %s", e)
        return False


def download_taf28k_modelscope(output_dir: Path) -> bool:
    """从 ModelScope 下载 (国内可用)"""
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("JimmyMa99/TeleAntiFraud")
        for split in ["train", "test"]:
            subset = ds[split] if split in ds else ds
            out = output_dir / f"TAF28k_{split}.jsonl"
            count = 0
            with open(out, "w", encoding="utf-8") as f:
                for s in subset:
                    f.write(json.dumps(dict(s), ensure_ascii=False) + "\n")
                    count += 1
            logger.info("  ✅ ModelScope %s: %d 条 → %s", split, count, out)
        return True
    except ImportError:
        logger.warning("modelscope 未安装，请运行: pip install modelscope")
        return False
    except Exception as e:
        logger.warning("ModelScope 下载失败: %s", e)
        return False


# ============================================================
# ChiFraud 下载
# ============================================================
def download_chifraud(output_dir: Path) -> bool:
    """从 GitHub 克隆 ChiFraud (纯文本数据集)"""
    import subprocess
    target = output_dir / "ChiFraud"
    if target.exists():
        logger.info("ChiFraud 已存在: %s", target)
        return True

    # 多源尝试
    repos = [
        "https://github.com/xuemingxxx/ChiFraud.git",
        "https://ghproxy.com/https://github.com/xuemingxxx/ChiFraud.git",
        "https://hub.fastgit.xyz/xuemingxxx/ChiFraud.git",
    ]
    for repo in repos:
        try:
            logger.info("尝试克隆: %s", repo)
            r = subprocess.run(
                ["git", "clone", "--depth=1", repo, str(target)],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                logger.info("✅ ChiFraud 克隆成功: %s", target)
                return True
            logger.warning("克隆失败 (%s): %s", repo, r.stderr[:200])
        except Exception as e:
            logger.warning("异常 (%s): %s", repo, e)

    return False


# ============================================================
# AdvFraud-3k 构建指南
# ============================================================
def create_advfraud_guide(output_dir: Path):
    """创建 AdvFraud-3k 构建指南"""
    guide = {
        "dataset": "AdvFraud-3k",
        "availability": "非公开 (自建对抗数据集)",
        "paper_section": "§4.1 实验设置与数据基准",
        "construction_method": {
            "step1": "从 TAF-28k 测试集随机抽取 1,000 条欺诈样本",
            "step2": "由 3 名独立标注员进行对抗式改写",
            "step3": "第 4 名资深标注员复核 (Cohen's κ = 0.87)",
            "step4": "领域专家撰写 2,000 条新型欺诈话术",
        },
        "adversarial_strategies": [
            "同义词扰动 (Synonym Perturbation)",
            "句式拓扑重排 (Syntactic Reordering)",
            "方言特征转换 (Dialect Feature Transformation)",
            "隐喻表达 (Metaphorical Expression)",
            "语气弱化 (Tone Attenuation)",
            "关键信息替换 (Key Info Substitution)",
            "长句拆分 (Long Sentence Splitting)",
            "跨领域话术注入 (Cross-domain Jargon Injection)",
        ],
        "note": "该数据集仅用于测试评估，不参与训练。如需复现，请参照上述方法从 TAF-28k 构建。",
    }
    path = output_dir / "AdvFraud3k_construction_guide.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(guide, f, ensure_ascii=False, indent=2)
    logger.info("✅ AdvFraud-3k 构建指南: %s", path)


# ============================================================
# 数据集元数据
# ============================================================
def create_metadata(output_dir: Path):
    """创建完整数据集元数据文件"""
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            "TAF-28k": {
                "full_name": "TeleAntiFraud-28k",
                "description": "首个面向电信反诈的开源音频-文本双模态慢思考数据集",
                "samples": 28511,
                "audio_hours": 307,
                "tasks": ["场景分类", "欺诈检测", "欺诈类型分类"],
                "splits": "8:1:1 (train:val:test)",
                "sources": [
                    {"name": "HuggingFace", "url": "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud"},
                    {"name": "ModelScope", "url": "https://www.modelscope.cn/datasets/JimmyMa99/TeleAntiFraud"},
                    {"name": "hf-mirror (国内镜像)", "url": "https://hf-mirror.com/datasets/JimmyMa99/TeleAntiFraud"},
                    {"name": "GitHub", "url": "https://github.com/JimmyMa99/TeleAntiFraud"},
                ],
                "paper": "Ma et al., ACM MM 2025",
                "arxiv": "2503.24115",
                "bib_key": "1",
            },
            "ChiFraud": {
                "full_name": "ChiFraud (SmsSpam-CN)",
                "description": "长周期中文网络欺诈文本基准数据集",
                "samples": 411934,
                "fraud_texts": 59106,
                "normal_texts": 352328,
                "fraud_categories": 11,
                "collection_period": "2022-2023 (12 months)",
                "sources": [
                    {"name": "GitHub", "url": "https://github.com/xuemingxxx/ChiFraud"},
                    {"name": "ACL Anthology", "url": "https://aclanthology.org/2025.coling-main.398/"},
                ],
                "paper": "Tang et al., COLING 2025",
                "bib_key": "chifraud",
            },
            "AdvFraud-3k": {
                "full_name": "AdvFraud-3k",
                "description": "自建对抗攻击测试集",
                "samples": 3000,
                "rewritten_from_taf28k": 1000,
                "newly_written": 2000,
                "adversarial_strategies": 8,
                "annotators": "3 名独立标注 + 1 名资深复核",
                "agreement": "Cohen's κ = 0.87",
                "availability": "NON-PUBLIC",
                "reason": "包含对抗攻击样本，由所在机构数据合规审查",
                "note": "构建方法见 AdvFraud3k_construction_guide.json",
                "paper_section": "§4.1",
                "bib_key": "in-house",
            },
        },
        "download_commands": {
            "taf28k_hf": "huggingface-cli download JimmyMa99/TeleAntiFraud --local-dir database/TAF28k",
            "taf28k_python": "from datasets import load_dataset; ds = load_dataset('JimmyMa99/TeleAntiFraud')",
            "taf28k_modelscope": "from modelscope.msdatasets import MsDataset; ds = MsDataset.load('JimmyMa99/TeleAntiFraud')",
            "chifraud_git": "git clone https://github.com/xuemingxxx/ChiFraud.git database/ChiFraud",
        },
    }

    path = output_dir / "datasets_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("✅ 数据集元数据: %s", path)


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数据集下载工具")
    parser.add_argument("--source", choices=["hf", "ms", "mirror", "all"],
                        default="all", help="下载源 (默认: all)")
    parser.add_argument("--dataset", choices=["all", "taf28k", "chifraud", "advfraud"],
                        default="all", help="目标数据集 (默认: all)")
    parser.add_argument("--output", type=str, default="database",
                        help="输出目录 (默认: database/)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📥 论文数据集下载工具")
    logger.info("  输出目录: %s", output_dir)
    logger.info("  下载源: %s", args.source)
    logger.info("  数据集: %s", args.dataset)

    # ── 元数据 ──
    create_metadata(output_dir)
    create_advfraud_guide(output_dir)

    # ── TAF-28k ──
    if args.dataset in ("all", "taf28k"):
        logger.info("\n📥 TAF-28k (TeleAntiFraud-28k)")

        if args.source in ("all", "hf"):
            if download_taf28k_hf(output_dir):
                args.source = "done"

        if args.source in ("all", "mirror") and args.source != "done":
            if download_taf28k_mirror(output_dir):
                args.source = "done"

        if args.source in ("all", "ms") and args.source != "done":
            download_taf28k_modelscope(output_dir)

        if args.source == "done":
            pass
        elif args.source not in ("all",):
            logger.warning("⚠️ 所有源均不可用，请手动下载")

    # ── ChiFraud ──
    if args.dataset in ("all", "chifraud"):
        logger.info("\n📥 ChiFraud (SmsSpam-CN)")
        ok = download_chifraud(output_dir)
        if not ok:
            logger.warning("⚠️ GitHub 不可达，请手动克隆: "
                          "git clone https://github.com/xuemingxxx/ChiFraud.git database/ChiFraud")

    # ── AdvFraud-3k ──
    if args.dataset in ("all", "advfraud"):
        logger.info("\n📝 AdvFraud-3k (非公开，已生成构建指南)")

    logger.info("\n✅ 完成! 数据集目录: %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            logger.info("  %s (%d bytes)", f.name, f.stat().st_size)
        else:
            logger.info("  %s/ (目录)", f.name)


if __name__ == "__main__":
    main()
