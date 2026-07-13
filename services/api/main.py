"""H100 RealEval API — Experiment control and status endpoint."""
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, os, json
from datetime import datetime

app = FastAPI(title="H100 RealEval", version="1.0.0")

class RunRequest(BaseModel):
    mode: str = "smoke"  # smoke | paper | sft

@app.get("/")
def root():
    return {"service": "H100 RealEval API", "status": "running"}

@app.get("/gpu")
def gpu_status():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                        "--format=csv,noheader,nounits"], timeout=5).decode()
        return {"gpus": [dict(zip(["name","mem_used_mb","mem_total_mb","util_pct"],
                                   [x.strip() for x in line.split(",")]))
                         for line in out.strip().splitlines()]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def experiment_status():
    results_dir = "/workspace/outputs/results"
    if os.path.isdir(results_dir):
        files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])
        return {"completed": len(files), "experiments": [f.replace(".json","") for f in files]}
    return {"completed": 0, "experiments": []}

ALLOWED_MODES = {"smoke", "paper"}
API_TOKEN = os.environ.get("REALEVAL_API_TOKEN", "")

@app.post("/run")
def run_experiment(req: RunRequest):
    # Validate mode against whitelist (prevents command injection)
    mode = req.mode.strip()
    if mode not in ALLOWED_MODES:
        return {"status": "error", "error": f"Invalid mode: {mode}. Allowed: {ALLOWED_MODES}"}
    try:
        # Use args list (no shell=True) for safe subprocess execution
        subprocess.Popen(
            ["python", "-m", "experiments.runner", f"--{mode}"],
            cwd="/workspace",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return {"status": "started", "mode": mode, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
