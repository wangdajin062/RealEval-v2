"""test_hf_data.py — 验证 HuggingFace datasets 集成与数据降级路径

用法:
    pytest tests/test_hf_data.py -v           # 运行所有测试
    pytest tests/test_hf_data.py -v -k hf     # 只运行 HF 相关测试
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import os


def test_local_data_first(tmp_path: Path) -> bool:
    """真实文件优先级高于 HF：若有本地文件，不会去加载 HF。"""
    # 在项目 data/ 下创建临时测试文件
    project_data = Path(__file__).resolve().parent.parent / "data"
    taf_dir = project_data / "TAF28k"
    taf_dir.mkdir(parents=True, exist_ok=True)
    import json
    test_file = taf_dir / "taf28k.jsonl"
    original_content = None
    if test_file.exists():
        original_content = test_file.read_text(encoding="utf-8")
    try:
        test_file.write_text(
            json.dumps({"text": "Local test fraud msg", "label": 1}) + "\n",
            encoding="utf-8")
        from realeval.data import load_taf28k
        ds = load_taf28k(max_samples=10)
        assert len(ds["texts"]) == 1 and ds["labels"] == [1], f"Unexpected: {ds}"
        print(f"[PASS] test_local_data_first: 本地文件优先 ({len(ds['texts'])} 条)")
        return True
    finally:
        if original_content is not None:
            test_file.write_text(original_content, encoding="utf-8")
        else:
            test_file.unlink(missing_ok=True)


def test_hf_loads() -> bool:
    """验证 HF datasets 加载。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("JimmyMa99/TeleAntiFraud", split="train")
        print(f"[INFO] HF TeleAntiFraud: {len(ds)} 条, 字段: {ds.column_names}")
        assert len(ds) > 0
        sample = ds[0]
        print(f"  样例: instruction={sample.get('instruction','')[:60]}... "
              f"label={sample.get('label')}")
        print("[PASS] test_hf_loads: HF 数据集加载成功")
        return True
    except Exception as e:
        print(f"[SKIP] test_hf_loads: HF 加载失败 ({e}) — 不联网或 datasets 未安装")
        return False


def test_fallback_synthetic() -> bool:
    """无真实数据时，必须降级到合成数据。"""
    from realeval.data import load_synthetic
    ds = load_synthetic(n=20)
    assert len(ds["texts"]) == 20 and len(ds["labels"]) == 20
    assert sum(ds["labels"]) >= 5  # 大约一半是欺诈
    print(f"[PASS] test_fallback_synthetic: 降级 OK ({len(ds['texts'])} 条, "
          f"{sum(ds['labels'])} 欺诈)")
    return True


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)

    results = []
    with tempfile.TemporaryDirectory() as td:
        results.append(test_local_data_first(Path(td)))

    results.append(test_fallback_synthetic())
    results.append(test_hf_loads())

    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'='*40}\n{passed}/{total} 测试通过")
    sys.exit(0 if passed == total else 1)
