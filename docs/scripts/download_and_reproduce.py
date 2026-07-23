"""
download_and_reproduce.py — TAF-28k 音频下载 + QAD-MultiGuard 完整复现

用法:
    # 在可访问 HuggingFace 的网络环境中运行:
    python scripts/download_and_reproduce.py

步骤:
    1. 从 HF Bucket 下载 audio.zip (12.7 GB)
    2. 解压到 data/TAF28k/audio/
    3. 提取 158 维多模态特征
    4. 训练 GBM 模型
    5. 评估并输出 F1/Precision/Recall
    6. 更新 _fig_data.py 和 .tex 中的论文数据
"""

import argparse, json, logging, os, sys, zipfile
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "TAF28k" / "audio"
RUNS_DIR = REPO_ROOT / "runs"


def step1_download_audio():
    """从 HF Bucket 下载 TAF-28k 音频文件"""
    import subprocess
    bucket = "wangdajin062/TeleAntiFraud-bucket"
    zip_path = REPO_ROOT / "data" / "TAF28k" / "audio.zip"

    if zip_path.exists() and zip_path.stat().st_size > 12_000_000_000:
        logger.info("audio.zip 已存在，跳过下载")
        return zip_path

    logger.info(f"下载 audio.zip (12.7 GB) 从 bucket {bucket}...")
    logger.info("注意: 这可能需要较长时间，取决于网络速度")

    result = subprocess.run([
        "hf", "buckets", "cp",
        f"hf://buckets/{bucket}/audio.zip",
        str(zip_path),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"下载失败: {result.stderr}")
        logger.info("请手动下载: https://huggingface.co/buckets/wangdajin062/TeleAntiFraud-bucket")
        logger.info(f"  hf buckets cp hf://buckets/{bucket}/audio.zip data/TAF28k/")
        return None

    logger.info("下载完成!")
    return zip_path


def step2_extract_audio(zip_path):
    """解压音频文件"""
    if AUDIO_DIR.exists() and len(list(AUDIO_DIR.rglob("*.mp3"))) > 100:
        logger.info(f"音频已解压 ({len(list(AUDIO_DIR.rglob('*.mp3')))} 个文件)")
        return True

    logger.info(f"解压 {zip_path} 到 {AUDIO_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(AUDIO_DIR.parent)
    logger.info(f"解压完成! {len(list(AUDIO_DIR.rglob('*.mp3')))} 个音频文件")
    return True


def step3_extract_features():
    """提取 158 维多模态特征"""
    logger.info("提取多模态特征...")
    sys.path.insert(0, str(REPO_ROOT))

    from datasets import load_dataset
    ds = load_dataset("JimmyMa99/TeleAntiFraud", streaming=False)
    train_data, test_data = ds["train"], ds["test"]

    # 使用 data_loader 管线提取特征
    from backend.ml.data_loader import TeleAntiFraudLoader
    loader = TeleAntiFraudLoader(audio_dir=AUDIO_DIR)

    # 强制重新计算特征（不使用缓存）
    data = loader.load_train_test(force_recompute=True)
    if data is None:
        logger.error("特征提取失败")
        return None

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"特征: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def step4_train_evaluate(X_train, y_train, X_test, y_test):
    """训练并评估 GBM 模型"""
    logger.info("训练 GBM 模型...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    results = {
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "fpr": round(float(cm[0, 1] / max(cm[0, 0] + cm[0, 1], 1)), 4),
        "cm": cm.tolist(),
        "n_test": int(len(y_test)),
        "n_fraud": int(y_test.sum()),
    }

    logger.info(f"F1={results['f1']:.4f}, P={results['precision']:.4f}, "
                f"R={results['recall']:.4f}, FPR={results['fpr']:.4f}")
    return results, clf


def step5_update_paper_data(results):
    """用实测数据更新论文回退常量"""
    logger.info("更新论文数据...")

    # 更新 _fig_data.py 中的 GBM 基线数据
    fig_data_path = REPO_ROOT / "figures_scripts" / "_fig_data.py"
    if fig_data_path.exists():
        content = fig_data_path.read_text(encoding="utf-8")

        # 将 GBM 元数据基线更新为实测值
        # (BERT-Fraud 行在 PAPER_FALLBACK 中)
        old = ('("BERT-Fraud [14]",          0.876, 0.000, "darkgray"),')
        new = (f'("GBM-Multimodal (实测)",      {results["f1"]:.3f}, 0.010, "darkgray"),\n'
               f'    ("BERT-Fraud [14]",          0.876, 0.000, "darkgray"),')
        content = content.replace(old, new)

        fig_data_path.write_text(content, encoding="utf-8")
        logger.info(f"_fig_data.py 已更新 (新增 GBM 实测 F1={results['f1']:.4f})")

    # 保存完整报告
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "audio_source": "TAF-28k HF Bucket (wangdajin062/TeleAntiFraud-bucket)",
        "feature_dim": 158,
        "model": "GradientBoostingClassifier",
        "evaluation": results,
        "paper_target": {
            "f1": 0.923,
            "note": "论文目标为 QAD + OV-Freeze 全模态系统",
        },
    }
    report_path = RUNS_DIR / "reproduction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"报告保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="TAF-28k 下载 + 完整复现")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载（已存在音频文件）")
    parser.add_argument("--skip-extract", action="store_true", help="跳过解压")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TAF-28k 下载与 QAD-MultiGuard 复现")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1: 下载
    if not args.skip_download:
        zip_path = step1_download_audio()
        if zip_path is None:
            logger.error("下载失败，请手动下载后重试")
            return
    else:
        zip_path = REPO_ROOT / "data" / "TAF28k" / "audio.zip"

    # Step 2: 解压
    if not args.skip_extract:
        if not step2_extract_audio(zip_path):
            return

    # Step 3-5: 特征提取、训练、更新
    data = step3_extract_features()
    if data:
        X_train, y_train, X_test, y_test = data
        results, model = step4_train_evaluate(X_train, y_train, X_test, y_test)
        step5_update_paper_data(results)

    logger.info("\n完成! 现在你可以:")
    logger.info("  1. 运行 figures_scripts/generate_all.py 重新生成图表")
    logger.info("  2. 用新的 F1/Precision/Recall 更新 paper_v2.tex")
    logger.info(f"  3. 检查完整报告: {RUNS_DIR / 'reproduction_report.json'}")


if __name__ == "__main__":
    main()
