from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_generate_all_from_repo_root_runs_without_cwd_path_error():
    result = subprocess.run(
        [sys.executable, "docs/figure_scripts/generate_all.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "paper_data.py — all consistency self-checks pass" in result.stdout
