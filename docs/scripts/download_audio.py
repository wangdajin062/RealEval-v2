"""
TeleAntiFraud 音频下载与解压工具
================================
从 HuggingFace 仓库下载 audio.zip（约 12GB）并解压到本地目录。

支持断点续传，可随时中断后重新运行。

用法:
  # 下载完整 audio.zip 并解压（建议后台运行）
  python scripts/download_audio.py

  # 指定输出目录
  python scripts/download_audio.py --output data/teleantifraud/audio

  # 仅解压已缓存的 zip（不重新下载）
  python scripts/download_audio.py --extract-only

  # 仅提取特定子目录
  python scripts/download_audio.py --include "POS-imitate-4"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HF_REPO = "JimmyMa99/TeleAntiFraud"
ZIP_FILENAME = "audio.zip"


def download_audio_zip(cache_dir: str | Path | None = None) -> Path:
    """下载 audio.zip（支持断点续传），返回 zip 路径。"""
    logger.info("正在下载 audio.zip（约 12GB，支持断点续传）...")
    logger.info("随时可以 Ctrl+C 中断，重新运行会从断点继续。")

    t0 = time.time()
    zip_path = hf_hub_download(
        HF_REPO,
        ZIP_FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
        resume_download=True,
        force_download=False,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    logger.info(
        "下载完成: %s (%.0f MB, %.1f min)", zip_path, size_mb, elapsed / 60
    )
    return Path(zip_path)


def extract_audio(
    zip_path: Path,
    output_dir: str | Path,
    include_prefix: str | None = None,
    limit: int | None = None,
):
    """
    解压 audio.zip 到 output_dir。

    参数:
        include_prefix: 仅解压路径包含此前缀的文件（如 "POS-imitate-4"）
        limit: 最多解压文件数（用于测试）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    logger.info("解压到: %s", output_dir)
    t0 = time.time()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        audio_files = [n for n in names if n.startswith("audio/") and n.endswith(".mp3")]

        logger.info("audio.zip 中共 %d 个音频文件", len(audio_files))

        if include_prefix:
            audio_files = [n for n in audio_files if include_prefix in n]
            logger.info(
                "过滤 '%s': %d 个文件", include_prefix, len(audio_files)
            )

        if limit:
            audio_files = audio_files[:limit]
            logger.info("限制解压: %d 个文件", len(audio_files))

        for name in audio_files:
            target_path = output_dir / name

            if target_path.exists() and target_path.stat().st_size > 0:
                skipped += 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                zf.extract(name, output_dir)
                extracted += 1
            except Exception as e:
                logger.warning("解压失败 %s: %s", name, e)

            if extracted > 0 and extracted % 200 == 0:
                elapsed = time.time() - t0
                logger.info("  已解压 %d 个 (%.1f 文件/秒)", extracted, extracted / elapsed)

    elapsed = time.time() - t0
    total = len(audio_files) if not limit else min(limit, len(audio_files))

    logger.info(
        "解压完成: %d 个 (新增 %d, 跳过 %d / %d 总), %.1f 秒",
        extracted + skipped, extracted, skipped, total, elapsed,
    )

    # 写一个 manifest 文件
    manifest_path = output_dir / "audio_manifest.json"
    import json
    manifest = {
        "source": f"{HF_REPO}/{ZIP_FILENAME}",
        "total_audio_files": total,
        "extracted": extracted,
        "skipped": skipped,
        "output_dir": str(output_dir.resolve()),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("清单已写入: %s", manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="下载并解压 TeleAntiFraud 音频数据"
    )
    parser.add_argument(
        "--output", default="data/teleantifraud/audio",
        help="音频输出目录（默认: data/teleantifraud/audio）",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HF 缓存目录（默认: ~/.cache/huggingface）",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="仅解压已缓存的 zip，不重新下载",
    )
    parser.add_argument(
        "--include", default=None,
        help="仅解压路径包含此前缀的音频（如 POS-imitate-4）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="限制解压文件数（测试用）",
    )
    args = parser.parse_args()

    # ── 下载 ───────────────────────────────────────────────
    if args.extract_only:
        # 尝试从缓存找已有 zip
        from huggingface_hub import HfApi, hf_hub_url
        from pathlib import Path as _P
        import re
        try:
            api = HfApi()
            meta = api.get_paths_info(HF_REPO, [ZIP_FILENAME], repo_type="dataset")
            if meta:
                commit_hash = meta[0].last_commit.split(":")[0] if ":" in str(meta[0].last_commit) else None
        except Exception:
            pass

        # 直接在缓存目录搜索
        cache_base = _P(args.cache_dir or
                        _P.home() / ".cache" / "huggingface" / "hub")
        blobs_dir = cache_base / "datasets--JimmyMa99--TeleAntiFraud" / "blobs"
        if blobs_dir.exists():
            zips = [f for f in blobs_dir.glob("*") if not f.name.endswith(".incomplete")]
            if zips:
                zip_path = zips[0]
                logger.info("使用缓存 zip: %s", zip_path)
            else:
                logger.error("未找到完整的缓存 zip 文件（存在未完成的下载）")
                logger.error("请先运行完整下载: python scripts/download_audio.py")
                sys.exit(1)
        else:
            logger.error("未找到 HF 缓存目录，请先运行下载")
            sys.exit(1)
    else:
        zip_path = download_audio_zip(args.cache_dir)

    # ── 解压 ───────────────────────────────────────────────
    extract_audio(
        zip_path, args.output,
        include_prefix=args.include,
        limit=args.limit,
    )

    logger.info("全部完成！音频目录: %s", Path(args.output).resolve())
    logger.info("在 SafeData-QAQ.py 中使用: --audio-dir %s", args.output)


if __name__ == "__main__":
    main()
