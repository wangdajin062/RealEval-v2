"""H100 RealEval API Server — FastAPI"""
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="H100 RealEval API",
    version="1.0.0",
    description="Real computation evaluation suite API for H100 clusters",
)

WORKSPACE = Path(os.environ.get("REALEVAL_OUTPUT_ROOT", "/workspace/outputs"))
REPO = Path("/workspace/repo")


# ─── Models ───

class ExperimentRequest(BaseModel):
    experiments: str = "all"          # e.g. "1,3,6" or "all"
    mode: str = "paper"               # paper | smoke
    distributed: bool = False
    benchmark: bool = True
    resume: bool = True


class RunResponse(BaseModel):
    run_id: str
    status: str
    started_at: str


class StatusResponse(BaseModel):
    run_id: str
    status: str
    pid: Optional[int] = None
    started_at: Optional[str] = None


# ─── Endpoints ───

@app.get("/")
async def root():
    return {
        "service": "H100 RealEval API",
        "version": "1.0.0",
        "endpoints": ["/health", "/gpu", "/experiments", "/results", "/docs"],
    }


@app.get("/health")
async def health():
    """System health check with GPU status."""
    gpu_info = {}
    try:
        import torch
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
        }
    except ImportError:
        gpu_info = {"error": "torch not installed"}

    disk = {}
    try:
        stat = os.statvfs("/workspace")
        disk["total_gb"] = round(stat.f_frsize * stat.f_blocks / 1024**3, 1)
        disk["free_gb"] = round(stat.f_frsize * stat.f_bavail / 1024**3, 1)
    except Exception:
        disk = {"error": "unable to query"}

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_info,
        "disk": disk,
    }


@app.get("/gpu")
async def gpu_status():
    """Detailed GPU status using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=30,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temp_c": float(parts[2]) if parts[2] != "[Not Supported]" else None,
                    "utilization_pct": float(parts[3]) if parts[3] != "[Not Supported]" else None,
                    "memory_used_mib": float(parts[4]),
                    "memory_total_mib": float(parts[5]),
                    "power_w": float(parts[6]) if parts[6] != "[Not Supported]" else None,
                })
        return {"gpus": gpus, "count": len(gpus)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiments/run")
async def run_experiments(req: ExperimentRequest):
    """Trigger an experiment run."""
    if not (REPO / "experiments" / "runner.py").exists():
        raise HTTPException(status_code=400, detail="RealEval repo not found at /workspace/repo")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    args = [
        "python", "-m", "experiments.runner",
        "--exp", req.experiments,
        f"--{req.mode}",
    ]
    if req.benchmark:
        args.append("--benchmark")
    if req.resume:
        args.append("--resume")

    env = {**os.environ, "REALEVAL_OUTPUT_ROOT": str(WORKSPACE)}
    log_file = WORKSPACE / f"run_{run_id}.log"

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            args, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT,
        )

    return RunResponse(run_id=run_id, status="running", started_at=datetime.utcnow().isoformat())


@app.get("/experiments/status/{run_id}")
async def experiment_status(run_id: str):
    """Check experiment run status."""
    log_file = WORKSPACE / f"run_{run_id}.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return StatusResponse(run_id=run_id, status="check_log", started_at=None)


@app.get("/results")
async def list_results():
    """List experiment output files."""
    results = []
    for f in WORKSPACE.rglob("*"):
        if f.is_file():
            results.append({
                "name": str(f.relative_to(WORKSPACE)),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return {"results": sorted(results, key=lambda r: r["name"])}


@app.get("/results/{filename:path}")
async def download_result(filename: str):
    """Download a specific result file."""
    filepath = WORKSPACE / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(filepath)


if __name__ == "__main__":
    import uvicorn
    workers = int(os.environ.get("API_WORKERS", "4"))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=workers, log_level="info")
