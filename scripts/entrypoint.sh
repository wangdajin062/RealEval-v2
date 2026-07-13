#!/bin/bash
set -e
echo "╔══════════════════════════════════════════════════╗"
echo "║       H100 RealEval — Services Starting          ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  SSH      → port 22                              ║"
echo "║  Jupyter  → http://____:8888                     ║"
echo "║  VSCode   → http://____:3000                     ║"
echo "║  Ollama   → http://____:11434                    ║"
echo "║  API      → http://____:8000/docs                ║"
echo "╚══════════════════════════════════════════════════╝"

# SSH
/usr/sbin/sshd -D &

# Jupyter Lab
jupyter-lab --config=/etc/jupyter/jupyter_notebook_config.py &

# code-server (VSCode)
code-server --config /root/.config/code-server/config.yaml &

# Ollama (if available)
command -v ollama && ollama serve &

# FastAPI
python /workspace/api/main.py &

# Start user command or keep alive
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "[entrypoint] Services running. Connect via Jupyter/VSCode/SSH."
    exec tail -f /dev/null
fi
