#!/usr/bin/env bash
# Finishes the Wan 2.2 Python 3.12 environment after a disconnect or reboot.
# Does not touch the model download and does not run generate.py.
set -u

export PATH="/home/rangoondiscount/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export UV_CACHE_DIR="/home/rangoondiscount/.cache/uv"
export UV_LINK_MODE=copy

ROOT="/home/rangoondiscount/VideoAI"
WAN="$ROOT/Wan2.2"
PY="$WAN/.venv/bin/python"
LOG_DIR="$ROOT/logs"
LOCK="$LOG_DIR/setup-wan.lock"
LOG="$LOG_DIR/setup-wan.log"
STATUS="$LOG_DIR/setup-wan-STATUS.txt"
DONE="$LOG_DIR/setup-wan.DONE"
REQ_FILTERED="$LOG_DIR/requirements-runtime.txt"

mkdir -p "$LOG_DIR" "$UV_CACHE_DIR" "$ROOT/outputs" "$ROOT/uploads"

if [[ "${WAN_SETUP_LOCKED:-}" != 1 ]]; then
  if ! flock -n -E 75 -o "$LOCK" env WAN_SETUP_LOCKED=1 "$0"; then
    rc=$?
    if [[ $rc -eq 75 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] skipped: setup lock held" >>"$LOG_DIR/setup-wan-cron.log"
      exit 0
    fi
    exit "$rc"
  fi
  exit $?
fi

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line" >>"$LOG"
  echo "$line"
}

write_status() {
  local state="$1"
  {
    echo "Wan 2.2 environment setup"
    echo "Updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "State: $state"
    echo "Log: $LOG"
    echo
    echo "This keeps running if the Mac session disconnects."
    echo "Cron retries every 15 minutes until this file says DONE."
  } >"$STATUS"
}

imports_ok() {
  (
    cd "$WAN" || exit 1
    "$PY" - <<'PY'
import torch, flash_attn, wan, streamlit
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
print("streamlit", streamlit.__version__)
print("wan", wan.__file__)
PY
  )
}

if [[ -f "$DONE" ]] && imports_ok >>"$LOG" 2>&1; then
  write_status "DONE"
  exit 0
fi

log "Environment setup starting (pid $$)"
write_status "running"

if [[ ! -x "$PY" ]]; then
  log "Python 3.12 venv is missing at $PY"
  write_status "failed — Python 3.12 venv is missing"
  exit 1
fi

torch_install_running() {
  pgrep -f 'uv pip install --python .venv/bin/python torch ' >/dev/null 2>&1
}

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  if torch_install_running; then
    log "Waiting for the PyTorch install that is already running"
    write_status "waiting for the in-flight PyTorch install"
    while torch_install_running; do
      sleep 30
    done
  fi
  if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
    log "Installing PyTorch cu128"
    write_status "installing PyTorch"
    if ! uv pip install --python "$PY" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 >>"$LOG" 2>&1; then
      log "PyTorch install failed"
      write_status "failed — PyTorch install failed, cron will retry"
      exit 1
    fi
  fi
fi
log "PyTorch is importable: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"

if ! "$PY" -c 'import flash_attn' >/dev/null 2>&1; then
  log "Trying flash-attn source build with MAX_JOBS=4"
  write_status "building flash-attn"
  if ! MAX_JOBS=4 uv pip install --python "$PY" flash-attn --no-build-isolation >>"$LOG" 2>&1; then
    torch_mm="$("$PY" -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(v[0]+"."+v[1])')"
    cuda_tag="$("$PY" -c 'import torch; print(torch.version.cuda.replace(".", ""))')"
    wheel="flash_attn-2.8.3+cu${cuda_tag}torch${torch_mm}-cp312-cp312-linux_x86_64.whl"
    url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/${wheel}"
    log "Source build failed. Installing prebuilt wheel $wheel"
    write_status "installing prebuilt flash-attn"
    if ! uv pip install --python "$PY" "$url" >>"$LOG" 2>&1; then
      log "Prebuilt flash-attn wheel failed: $url"
      write_status "failed — flash-attn install failed, cron will retry"
      exit 1
    fi
  fi
fi
log "flash_attn is importable"

grep -v -E '^(flash_attn|numpy)([[:space:]]|$)' "$WAN/requirements.txt" >"$REQ_FILTERED"
log "Installing Wan runtime packages"
write_status "installing Wan requirements and Streamlit"
if ! uv pip install --python "$PY" -r "$REQ_FILTERED" >>"$LOG" 2>&1; then
  log "Pinned requirements failed. Retrying without the transformers upper bound."
  sed -E 's/^transformers>=4.49.0,<=4.51.3/transformers>=4.49.0/' "$REQ_FILTERED" >"${REQ_FILTERED}.relaxed"
  if ! uv pip install --python "$PY" -r "${REQ_FILTERED}.relaxed" >>"$LOG" 2>&1; then
    log "Runtime package install failed"
    write_status "failed — requirements install failed, cron will retry"
    exit 1
  fi
fi
if ! uv pip install --python "$PY" streamlit psutil >>"$LOG" 2>&1; then
  log "Streamlit install failed"
  write_status "failed — Streamlit install failed, cron will retry"
  exit 1
fi

# generate.py imports the whole wan package, so speech/animate extras are required even for TI2V.
log "Installing Wan speech/animate import extras"
write_status "installing Wan speech/animate extras"
if ! uv pip install --python "$PY" -r "$WAN/requirements_s2v.txt" peft einops loguru sentencepiece >>"$LOG" 2>&1; then
  log "Speech/animate extras install failed"
  write_status "failed — speech/animate extras install failed, cron will retry"
  exit 1
fi

log "Checking imports"
if ! imports_ok >>"$LOG" 2>&1; then
  log "Import check failed"
  write_status "failed — import check failed, see the log"
  exit 1
fi

date '+%Y-%m-%d %H:%M:%S %Z' >"$DONE"
write_status "DONE"
log "DONE — Python environment is ready. Generation stays off until the weight download says DONE."
exit 0
