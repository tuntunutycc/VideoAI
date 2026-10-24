#!/usr/bin/env bash
# Dual-GPU TI2V on 29 GB hosts: encode T5 in one process, then FSDP/Ulysses DiT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
TORCHRUN="$ROOT/.venv/bin/torchrun"
CKPT="${CKPT_DIR:-/home/rangoondiscount/VideoAI/models/Wan2.2-TI2V-5B}"
CTX="$(mktemp /tmp/wan_ti2v_ctx_XXXXXX.pt)"
cleanup() { rm -f "$CTX"; }
trap cleanup EXIT

PROMPT=""
SIZE="1280*704"
FRAMES=49
STEPS=20
GUIDE=5.0
SEED=42
SAVE=""
IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --frame_num) FRAMES="$2"; shift 2 ;;
    --sample_steps) STEPS="$2"; shift 2 ;;
    --sample_guide_scale) GUIDE="$2"; shift 2 ;;
    --base_seed) SEED="$2"; shift 2 ;;
    --save_file) SAVE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --ckpt_dir) CKPT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPT" || -z "$SAVE" ]]; then
  echo "Usage: $0 --prompt TEXT --save_file OUT.mp4 [options]" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# Peer-to-peer DMA hangs between these two Ada cards; force NCCL over SHM/sockets.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "[dual] Stage 1/2: T5 encode (single process)"
"$PY" "$ROOT/encode_context.py" \
  --task ti2v-5B \
  --ckpt_dir "$CKPT" \
  --prompt "$PROMPT" \
  --save_context "$CTX"

echo "[dual] Stage 2/2: DiT sample on 2 GPUs (FSDP + Ulysses)"
if [[ -n "$IMAGE" ]]; then
  "$TORCHRUN" --nproc_per_node=2 "$ROOT/generate.py" \
    --task ti2v-5B \
    --size "$SIZE" \
    --frame_num "$FRAMES" \
    --ckpt_dir "$CKPT" \
    --convert_model_dtype \
    --prompt "$PROMPT" \
    --base_seed "$SEED" \
    --sample_steps "$STEPS" \
    --sample_guide_scale "$GUIDE" \
    --save_file "$SAVE" \
    --dit_fsdp \
    --ulysses_size 2 \
    --offload_model False \
    --context_file "$CTX" \
    --image "$IMAGE"
else
  "$TORCHRUN" --nproc_per_node=2 "$ROOT/generate.py" \
    --task ti2v-5B \
    --size "$SIZE" \
    --frame_num "$FRAMES" \
    --ckpt_dir "$CKPT" \
    --convert_model_dtype \
    --prompt "$PROMPT" \
    --base_seed "$SEED" \
    --sample_steps "$STEPS" \
    --sample_guide_scale "$GUIDE" \
    --save_file "$SAVE" \
    --dit_fsdp \
    --ulysses_size 2 \
    --offload_model False \
    --context_file "$CTX"
fi

echo "[dual] Done: $SAVE"
