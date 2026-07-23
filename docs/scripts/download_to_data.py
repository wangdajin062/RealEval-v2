"""
download_to_data.py — 下载论文三个数据集到 data/ 文件夹
=========================================================
多源回退策略: ModelScope → hf-mirror → HuggingFace → GitHub

数据集:
  1. TAF-28k    — ModelScope / HuggingFace: JimmyMa99/TeleAntiFraud
  2. ChiFraud    — GitHub: xuemingxxx/ChiFraud
  3. AdvFraud-3k — 非公开 (创建元数据 + 构建指南)

用法:
  python scripts/download_to_data.py [--dataset all|taf28k|chifraud]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_to_data")

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# TAF-28k 下载 (多源回退)
# ============================================================
def download_taf28k_modelscope(output_dir: Path) -> Optional[dict]:
    """从 ModelScope 下载 TAF-28k (国内首选)"""
    logger.info("  尝试 ModelScope: JimmyMa99/TeleAntiFraud ...")
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("JimmyMa99/TeleAntiFraud", subset_name="default")

        stats = {}
        # ModelScope 数据集通常有 train/validation/test splits
        if hasattr(ds, '_data_files'):
            logger.info("    data_files: %s", list(ds._data_files.keys())[:5])
            stats['splits_found'] = list(ds._data_files.keys())

        # 尝试按 split 遍历
        for split_name in ["train", "validation", "test"]:
            try:
                subset = ds[split_name] if hasattr(ds, '__getitem__') else ds
                out_path = output_dir / f"TAF28k_{split_name}.jsonl"
                count = 0
                with open(out_path, "w", encoding="utf-8") as f:
                    for sample in subset:
                        # 处理不同格式的 sample
                        if isinstance(sample, dict):
                            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        else:
                            f.write(json.dumps({"text": str(sample)}, ensure_ascii=False) + "\n")
                        count += 1
                if count > 0:
                    logger.info("    ✅ %s: %d 条 → %s", split_name, count, out_path.name)
                    stats[split_name] = count
            except Exception as e:
                logger.debug("    split %s: %s", split_name, e)

        if any(v > 0 for v in stats.values() if isinstance(v, int)):
            return {"source": "ModelScope", "splits": stats}
        else:
            # 尝试直接遍历整个数据集
            out_path = output_dir / "TAF28k_full.jsonl"
            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in ds:
                    if isinstance(sample, dict):
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
            if count > 0:
                logger.info("    ✅ full dataset: %d 条 → %s", count, out_path.name)
                return {"source": "ModelScope", "full_count": count}
            return None

    except Exception as e:
        logger.warning("  ModelScope 失败: %s", str(e)[:120])
        return None


def download_taf28k_hf_mirror(output_dir: Path) -> Optional[dict]:
    """从 hf-mirror.com 下载 (国内 HuggingFace 镜像)"""
    logger.info("  尝试 hf-mirror.com: JimmyMa99/TeleAntiFraud ...")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from datasets import load_dataset

        stats = {}
        for split in ["train", "test"]:
            try:
                ds = load_dataset(
                    "JimmyMa99/TeleAntiFraud",
                    split=split,
                    streaming=False,
                    trust_remote_code=True,
                )
                out_path = output_dir / f"TAF28k_{split}.jsonl"
                count = 0
                with open(out_path, "w", encoding="utf-8") as f:
                    for sample in ds:
                        f.write(json.dumps(dict(sample), ensure_ascii=False) + "\n")
                        count += 1
                logger.info("    ✅ %s: %d 条 → %s", split, count, out_path.name)
                stats[split] = count
            except Exception as e:
                logger.warning("    split %s: %s", split, str(e)[:120])
        return {"source": "hf-mirror", "splits": stats} if stats else None
    except Exception as e:
        logger.warning("  hf-mirror 失败: %s", str(e)[:120])
        return None


def download_taf28k_hf_direct(output_dir: Path) -> Optional[dict]:
    """从 HuggingFace 直接下载"""
    logger.info("  尝试 HuggingFace 直连 ...")
    try:
        os.environ.pop("HF_ENDPOINT", None)  # 清除镜像设置
        from datasets import load_dataset
        ds = load_dataset("JimmyMa99/TeleAntiFraud", streaming=False)
        stats = {}
        for split in ds.keys():
            subset = ds[split]
            out_path = output_dir / f"TAF28k_{split}.jsonl"
            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in subset:
                    f.write(json.dumps(dict(sample), ensure_ascii=False) + "\n")
                    count += 1
            logger.info("    ✅ %s: %d 条", split, count)
            stats[split] = count
        return {"source": "HuggingFace", "splits": stats}
    except Exception as e:
        logger.warning("  HuggingFace 直连失败: %s", str(e)[:120])
        return None


# ============================================================
# ChiFraud 下载
# ============================================================
def download_chifraud(output_dir: Path) -> Optional[dict]:
    """从 GitHub 克隆 ChiFraud (多源尝试)"""
    target = output_dir / "ChiFraud"
    if target.exists() and any(target.iterdir()):
        # 检查是否有效克隆
        if (target / "README.md").exists() or (target / "data").exists():
            logger.info("  ChiFraud 已存在且有效: %s", target)
            return {"source": "existing", "path": str(target)}

    repos = [
        ("GitHub 直连", "https://github.com/xuemingxxx/ChiFraud.git"),
        ("ghproxy 加速", "https://ghproxy.com/https://github.com/xuemingxxx/ChiFraud.git"),
    ]

    for name, repo in repos:
        logger.info("  尝试 %s: %s", name, repo)
        try:
            # 如果之前有失败的 clone，先清理
            if target.exists():
                import shutil
                shutil.rmtree(target, ignore_errors=True)

            result = subprocess.run(
                ["git", "clone", "--depth=1", "--single-branch", repo, str(target)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("  ✅ ChiFraud 克隆成功 (%s)", name)
                return {"source": name, "path": str(target)}
            else:
                logger.warning("  克隆失败: %s", result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("  超时: %s", name)
        except Exception as e:
            logger.warning("  异常: %s", str(e)[:120])

    return None


# ============================================================
# AdvFraud-3k
# ============================================================
def create_advfraud_doc(output_dir: Path):
    """创建 AdvFraud-3k 文档"""
    doc = {
        "dataset": "AdvFraud-3k",
        "availability": "NON-PUBLIC — 自建对抗数据集",
        "description": "论文 §4.1 中使用的对抗测试集",
        "samples": 3000,
        "construction": "1,000条 TAF-28k 改写 + 2,000条专家撰写",
        "note": "构建方法详见 database/AdvFraud3k_construction_guide.json",
    }
    path = output_dir / "AdvFraud3k_README.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info("  ✅ AdvFraud-3k 文档已创建 (非公开数据集)")


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载论文数据集到 data/")
    parser.add_argument("--dataset", choices=["all", "taf28k", "chifraud"],
                        default="all")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("📥 论文数据集下载 → %s", DATA_DIR)
    logger.info("")

    results = {}

    # ── TAF-28k ──
    if args.dataset in ("all", "taf28k"):
        logger.info("=" * 60)
        logger.info("📥 TAF-28k (TeleAntiFraud-28k)")
        logger.info("=" * 60)
        taf_dir = DATA_DIR / "TAF28k"
        taf_dir.mkdir(parents=True, exist_ok=True)

        result = None
        for download_fn in [download_taf28k_modelscope,
                            download_taf28k_hf_mirror,
                            download_taf28k_hf_direct]:
            if result is None:
                result = download_fn(taf_dir)

        if result:
            logger.info("✅ TAF-28k 下载成功: %s", result)
            results["TAF-28k"] = result
        else:
            logger.warning("⚠️ 所有源均不可用，创建元数据文件")
            meta = {
                "dataset": "TeleAntiFraud-28k",
                "samples": 28511,
                "sources": {
                    "ModelScope": "JimmyMa99/TeleAntiFraud",
                    "HuggingFace": "JimmyMa99/TeleAntiFraud",
                    "hf-mirror": "https://hf-mirror.com/datasets/JimmyMa99/TeleAntiFraud",
                    "GitHub": "https://github.com/JimmyMa99/TeleAntiFraud",
                },
                "download_commands": {
                    "modelscope": "python -c \"from modelscope.msdatasets import MsDataset; ds = MsDataset.load('JimmyMa99/TeleAntiFraud')\"",
                    "huggingface": "huggingface-cli download JimmyMa99/TeleAntiFraud --local-dir data/TAF28k",
                    "python": "from datasets import load_dataset; ds = load_dataset('JimmyMa99/TeleAntiFraud')",
                },
                "status": "download_failed_all_sources",
                "note": "请在有网络连接的环境中手动执行上述命令下载",
            }
            with open(taf_dir / "README.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            results["TAF-28k"] = {"status": "failed", "metadata_created": True}

    # ── ChiFraud ──
    if args.dataset in ("all", "chifraud"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("📥 ChiFraud (SmsSpam-CN)")
        logger.info("=" * 60)
        result = download_chifraud(DATA_DIR)

        if result:
            logger.info("✅ ChiFraud 下载成功: %s", result)
            results["ChiFraud"] = result
        else:
            logger.warning("⚠️ 所有源均不可用，创建元数据文件")
            meta = {
                "dataset": "ChiFraud (SmsSpam-CN)",
                "samples": 411934,
                "source": "https://github.com/xuemingxxx/ChiFraud",
                "paper": "Tang et al., COLING 2025",
                "download_command": "git clone --depth=1 https://github.com/xuemingxxx/ChiFraud.git data/ChiFraud",
                "status": "download_failed_all_sources",
                "note": "请在有网络连接的环境中手动执行上述命令下载",
            }
            chi_dir = DATA_DIR / "ChiFraud"
            chi_dir.mkdir(parents=True, exist_ok=True)
            with open(chi_dir / "README.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            results["ChiFraud"] = {"status": "failed", "metadata_created": True}

    # ── AdvFraud-3k ──
    if args.dataset in ("all", "advfraud"):
        create_advfraud_doc(DATA_DIR)
        results["AdvFraud-3k"] = {"status": "non_public", "doc_created": True}

    # ── 最终状态 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 下载结果")
    logger.info("=" * 60)
    for ds_name, info in results.items():
        status = info.get("source", info.get("status", "unknown"))
        logger.info("  %s: %s", ds_name, status)

    logger.info("")
    logger.info("📂 数据目录: %s", DATA_DIR)
    for item in sorted(DATA_DIR.iterdir()):
        if item.is_dir():
            file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            logger.info("  %s/ (%d 文件)", item.name, file_count)
        else:
            logger.info("  %s (%d bytes)", item.name, item.stat().st_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
