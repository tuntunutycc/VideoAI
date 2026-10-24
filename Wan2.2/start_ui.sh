#!/usr/bin/env bash
# Prefer the always-on systemd user service. This script is a manual fallback.
set -euo pipefail
cd /home/rangoondiscount/VideoAI/Wan2.2
mkdir -p /home/rangoondiscount/VideoAI/logs
exec .venv/bin/streamlit run app.py \
  --server.port 8502 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false
