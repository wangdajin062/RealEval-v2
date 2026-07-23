"""
TeleAntiFraud 数据集下载脚本
==============================
通过 HuggingFace 邮箱登录获取 Token，下载 binary_classification 数据。

用法:
  # 交互式输入邮箱密码
  python scripts/download_teleantifraud.py

  # 或通过环境变量传入（更安全）
  set HF_EMAIL=your@email.com
  set HF_PASSWORD=yourpassword
  python scripts/download_teleantifraud.py

  # 或直接提供 Token
  python scripts/download_teleantifraud.py --token hf_xxxxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests


def login_get_token(email: str, password: str) -> str | None:
    """通过邮箱密码登录 HF 并获取访问令牌"""
    session = requests.Session()

    # Step 1: 获取登录页面和 CSRF token
    print("[1/4] 获取登录页面...")
    r = session.get("https://huggingface.co/login")
    if r.status_code != 200:
        print(f"  获取登录页面失败: {r.status_code}")
        return None

    import re
    csrf_match = re.search(r'name="csrf"[^>]*value="([^"]+)"', r.text)
    if not csrf_match:
        print("  无法获取 CSRF token")
        return None
    csrf = csrf_match.group(1)

    # Step 2: 提交登录
    print("[2/4] 登录中...")
    r2 = session.post(
        "https://huggingface.co/login",
        data={"csrf": csrf, "username": email, "password": password},
        allow_redirects=True,
    )
    if r2.status_code != 200 or "login" in r2.url.lower():
        print("  登录失败，请检查邮箱和密码")
        return None
    print("  登录成功!")

    # Step 3: 获取 Token
    print("[3/4] 获取 Access Token...")
    r3 = session.get("https://huggingface.co/settings/tokens")
    if r3.status_code != 200:
        print(f"  获取 Token 页面失败: {r3.status_code}")
        return None

    # 从页面提取已有 token 或创建新 token
    tokens = re.findall(r'value="(hf_[a-zA-Z0-9]{10,})"', r3.text)
    if tokens:
        token = tokens[0]
        print(f"  使用已有 Token: {token[:8]}...{token[-4:]}")
        return token

    # Step 4: 创建新 Token（通过 API）
    print("  创建新 Token...")
    # 获取创建 token 的 csrf
    csrf_match2 = re.search(r'name="csrf"[^>]*value="([^"]+)"', r3.text)
    if not csrf_match2:
        print("  无法获取创建 token 的 CSRF")
        return None
    csrf2 = csrf_match2.group(1)

    r4 = session.post(
        "https://huggingface.co/settings/tokens",
        data={
            "csrf": csrf2,
            "name": "teleantifraud_download",
            "role": "write",
        },
    )
    if r4.status_code != 200:
        print(f"  创建 Token 失败: {r4.status_code}")
        return None

    tokens2 = re.findall(r'value="(hf_[a-zA-Z0-9]{10,})"', r4.text)
    if tokens2:
        token = tokens2[0]
        print(f"  创建新 Token: {token[:8]}...{token[-4:]}")
        return token

    print("  无法获取 Token")
    return None


def download_dataset(token: str, output_dir: str | Path) -> Path | None:
    """使用 Token 下载 TeleAntiFraud binary_classification.zip"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {token}"}
    url = ("https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/"
           "resolve/main/binary_classification.zip")

    zip_path = output_dir / "binary_classification.zip"

    print(f"[4/4] 下载 binary_classification.zip...")
    r = requests.get(url, headers=headers, stream=True)
    if r.status_code != 200:
        print(f"  下载失败: HTTP {r.status_code}")
        if r.status_code == 401:
            print("  Token 无权访问此数据集。需要先在 HF 上申请访问:")
            print("  https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud")
        return None

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  进度: {pct}% ({downloaded//1024**2}MB/{total//1024**2}MB)", end="")

    print(f"\n  下载完成: {zip_path} ({downloaded//1024**2}MB)")

    # 解压
    extract_dir = output_dir / "binary_classification"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"  解压到: {extract_dir}")

    # 检查文件
    train_json = extract_dir / "train.json"
    test_json = extract_dir / "test.json"
    if train_json.exists():
        with open(train_json, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  train.json: {len(data)} 条样本")
    if test_json.exists():
        with open(test_json, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  test.json: {len(data)} 条样本")

    return extract_dir


def main():
    parser = argparse.ArgumentParser(description="下载 TeleAntiFraud 数据集")
    parser.add_argument("--token", help="HF Access Token（直接提供 Token 则跳过登录）")
    parser.add_argument("--output", default="data/teleantifraud", help="输出目录")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

    if not token:
        email = os.environ.get("HF_EMAIL") or input("HF 邮箱: ")
        password = os.environ.get("HF_PASSWORD")
        if not password:
            import getpass
            password = getpass.getpass("HF 密码: ")

        token = login_get_token(email, password)
        if not token:
            print("\n登录失败。你也可以直接提供 Token:")
            print("  python scripts/download_teleantifraud.py --token hf_xxxxx")
            print("Token 获取地址: https://huggingface.co/settings/tokens")
            sys.exit(1)

    extract_dir = download_dataset(token, args.output)
    if extract_dir:
        print(f"\n{'='*60}")
        print(f"数据已下载到: {extract_dir}")
        print(f"使用方式: python SafeData-QAQ.py --data-dir {extract_dir} --samples 4000")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
