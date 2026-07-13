#!/bin/bash
# Returns 0 if all critical services are healthy
FAIL=0

# GPU check
nvidia-smi &>/dev/null || { echo "GPU not available"; FAIL=1; }

# Jupyter check
curl -s http://localhost:8888/api/status &>/dev/null || { echo "Jupyter not responding"; FAIL=1; }

# API check
curl -s http://localhost:8000/ &>/dev/null || { echo "API not responding"; FAIL=1; }

# Disk check
[ "$(df /workspace --output=pcent | tail -1 | tr -d ' %')" -lt 95 ] || { echo "Disk nearly full"; FAIL=1; }

exit $FAIL
