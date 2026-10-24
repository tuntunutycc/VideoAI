#!/usr/bin/env bash
# Downloads Wan-AI/Wan2.2-TI2V-5B (Apache 2.0) into the VideoAI project.
# Safe to re-run from cron after an SSH drop or a reboot. Partial files resume.
set -u

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROOT="/home/rangoondiscount/VideoAI"
DEST="$ROOT/models/Wan2.2-TI2V-5B"
LOG_DIR="$ROOT/logs"
LOCK="$LOG_DIR/wan22.lock"
STATUS="$LOG_DIR/wan22-STATUS.txt"
LOG="$LOG_DIR/wan22.log"
CRON_LOG="$LOG_DIR/wan22-cron.log"
DONE="$LOG_DIR/wan22.DONE"
BASE="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main"
TOTAL_EXPECTED=34203123497

# path<TAB>expected bytes, from the Hugging Face tree on 2026-10-12
MANIFEST=$(cat <<'EOF'
.gitattributes	1866
.msc	507
.mv	36
README.md	15935
config.json	251
configuration.json	43
diffusion_pytorch_model.safetensors.index.json	72865
google/umt5-xxl/special_tokens_map.json	6623
google/umt5-xxl/tokenizer_config.json	61728
google/umt5-xxl/spiece.model	4548313
google/umt5-xxl/tokenizer.json	16837417
examples/i2v_input.JPG	250628
assets/logo.png	56322
assets/moe_arch.png	74900
assets/vae.png	165486
assets/comp_effic.png	202156
assets/performance.png	306535
assets/moe_2.png	527914
diffusion_pytorch_model-00003-of-00003.safetensors	178558176
Wan2.2_VAE.pth	2818839170
diffusion_pytorch_model-00001-of-00003.safetensors	9825014472
diffusion_pytorch_model-00002-of-00003.safetensors	9995661736
models_t5_umt5-xxl-enc-bf16.pth	11361920418
EOF
)

mkdir -p "$DEST" "$LOG_DIR"

# flock itself keeps the lock. -o closes that fd in the download process so
# curl/sleep cannot hold it after this script exits.
if [[ "${WAN22_LOCKED:-}" != 1 ]]; then
  if ! flock -n -E 75 -o "$LOCK" env WAN22_LOCKED=1 "$0"; then
    rc=$?
    if [[ $rc -eq 75 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] skipped: wan22 download lock held" >>"$CRON_LOG"
      exit 0
    fi
    exit "$rc"
  fi
  exit $?
fi

if [[ -f "$DONE" ]]; then
  exit 0
fi

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line" >>"$LOG"
  echo "$line"
}

file_bytes() {
  local path="$1"
  if [[ -f "$path" ]]; then
    stat -c %s "$path"
  else
    echo 0
  fi
}

write_status() {
  local state="$1"
  local got_total=0
  local rel expected got pct
  {
    echo "Wan 2.2 TI2V-5B download"
    echo "Updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "State: $state"
    echo "Destination: $DEST"
    echo "Log: $LOG"
    echo
    while IFS=$'\t' read -r rel expected; do
      [[ -z "$rel" ]] && continue
      got="$(file_bytes "$DEST/$rel")"
      got_total=$((got_total + got))
      if [[ "$expected" -gt 0 ]]; then
        pct=$((got * 100 / expected))
      else
        pct=0
      fi
      printf '  %3s%%  %15s / %15s  %s\n' "$pct" "$got" "$expected" "$rel"
    done <<<"$MANIFEST"
    echo
    if [[ "$TOTAL_EXPECTED" -gt 0 ]]; then
      printf 'Total: %s / %s bytes (%.1f%%)\n' "$got_total" "$TOTAL_EXPECTED" \
        "$(awk -v g="$got_total" -v t="$TOTAL_EXPECTED" 'BEGIN { printf "%.1f", (100 * g) / t }')"
    fi
    echo
    echo "This keeps running if the connection drops."
    echo "After a reboot, cron starts it again within 15 minutes and resumes partial files."
    echo "Watch live: tail -f $LOG"
  } >"$STATUS"
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && ! -f "$DONE" ]]; then
    write_status "interrupted (exit $rc) — cron will resume"
    log "Stopped with exit $rc; lock released so cron can resume"
  fi
}
trap on_exit EXIT

all_complete() {
  local rel expected got
  while IFS=$'\t' read -r rel expected; do
    [[ -z "$rel" ]] && continue
    got="$(file_bytes "$DEST/$rel")"
    if [[ "$got" -ne "$expected" ]]; then
      return 1
    fi
  done <<<"$MANIFEST"
  return 0
}

download_one() {
  local rel="$1"
  local expected="$2"
  local out="$DEST/$rel"
  local got rc=0
  mkdir -p "$(dirname "$out")"
  got="$(file_bytes "$out")"
  if [[ "$got" -eq "$expected" ]]; then
    log "complete  $rel"
    return 0
  fi
  if [[ "$got" -gt "$expected" ]]; then
    log "removing oversized file $rel ($got > $expected)"
    rm -f "$out"
    got=0
  fi
  log "downloading $rel ($got / $expected bytes)"
  write_status "downloading $rel"
  if [[ "$got" -gt 0 ]]; then
    curl -4 --http1.1 -C - -fL --retry 50 --retry-all-errors --retry-delay 15 \
      --connect-timeout 30 --speed-time 120 --speed-limit 10000 \
      -A "videoai-wan22-download/1.0" \
      -o "$out" \
      "$BASE/$rel" >>"$LOG" 2>&1 &
  else
    curl -4 --http1.1 -fL --retry 50 --retry-all-errors --retry-delay 15 \
      --connect-timeout 30 --speed-time 120 --speed-limit 10000 \
      -A "videoai-wan22-download/1.0" \
      -o "$out" \
      "$BASE/$rel" >>"$LOG" 2>&1 &
  fi
  local cpid=$!
  local ticks=0
  while kill -0 "$cpid" 2>/dev/null; do
    if (( ticks % 10 == 0 )); then
      write_status "downloading $rel"
    fi
    ticks=$((ticks + 1))
    sleep 2
  done
  wait "$cpid" || rc=$?
  got="$(file_bytes "$out")"
  if [[ $rc -ne 0 || "$got" -ne "$expected" ]]; then
    log "incomplete $rel (curl exit $rc, have $got, want $expected)"
    return 1
  fi
  log "finished  $rel"
  return 0
}

log "Wan 2.2 TI2V-5B download starting (pid $$)"
write_status "running"

if all_complete; then
  date '+%Y-%m-%d %H:%M:%S %Z' >"$DONE"
  write_status "DONE"
  log "All files already complete"
  exit 0
fi

failed=0
while IFS=$'\t' read -r rel expected; do
  [[ -z "$rel" ]] && continue
  if ! download_one "$rel" "$expected"; then
    failed=1
    break
  fi
done <<<"$MANIFEST"

if [[ "$failed" -ne 0 ]]; then
  write_status "incomplete — cron will resume"
  exit 1
fi

if ! all_complete; then
  write_status "incomplete — cron will resume"
  log "Manifest check failed after downloads"
  exit 1
fi

date '+%Y-%m-%d %H:%M:%S %Z' >"$DONE"
write_status "DONE"
log "DONE — Wan 2.2 TI2V-5B is on this machine"
exit 0
