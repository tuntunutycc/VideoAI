#!/usr/bin/env python3
"""VideoAI web UI for Wan2.2 TI2V-5B on two RTX 4000 Ada GPUs."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path("/home/rangoondiscount/VideoAI")
WAN_ROOT = ROOT / "Wan2.2"
CKPT_DIR = ROOT / "models" / "Wan2.2-TI2V-5B"
STATUS_FILE = ROOT / "logs" / "wan22-STATUS.txt"
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR = ROOT / "uploads"
LEGACY_LOCK = ROOT / "gpu.lock"
LOCK_FILES = {0: ROOT / "gpu0.lock", 1: ROOT / "gpu1.lock"}
GEN_LOGS = {0: ROOT / "logs" / "generate-gpu0.log", 1: ROOT / "logs" / "generate-gpu1.log"}
GEN_LOG = ROOT / "logs" / "generate.log"  # latest-started job (compat)
PYTHON = WAN_ROOT / ".venv" / "bin" / "python"
TORCHRUN = WAN_ROOT / ".venv" / "bin" / "torchrun"
GENERATE = WAN_ROOT / "generate.py"
ADD_AUDIO = WAN_ROOT / "add_audio.py"
DUAL_RUNNER = WAN_ROOT / "run_dual_ti2v.sh"

# Parallel jobs need ~10–12 GB host RAM each; refuse a 2nd job below this.
MIN_HOST_RAM_GB_FOR_PARALLEL = 12.0

DEFAULT_PROMPT = (
    "A red paper boat floats down a quiet canal at dusk, "
    "lantern light reflecting on the water"
)
PRESETS = [
    ("Paper boat", DEFAULT_PROMPT),
    (
        "Boxing cats",
        "Two anthropomorphic cats in comfy boxing gear and bright gloves "
        "fight intensely on a spotlighted stage",
    ),
    (
        "Night stall",
        "A street food stall at night, steam rising from the grill, "
        "warm lights, the camera slowly dollies closer",
    ),
    (
        "Sunrise surf",
        "Ocean waves roll onto a black sand beach at sunrise, "
        "slow cinematic motion, spray catching the light",
    ),
]
FRAME_CHOICES = [33, 49, 65, 81, 97, 121]
DURATION_OPTIONS = {
    "~2 seconds (49 frames, fast test)": 49,
    "~3.5 seconds (81 frames)": 81,
    "~5 seconds (121 frames, full clip)": 121,
    "Custom 4n+1": 0,
}
VIDEO_GPU_CHOICES = {
    "GPU 0": "gpu0",
    "GPU 1": "gpu1",
    "Auto Load-Balance": "auto",
}
AUDIO_GPU_CHOICES = ["GPU 1", "GPU 0", "Auto Load-Balance"]


def folder_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(out.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def read_status() -> dict:
    text = STATUS_FILE.read_text(errors="replace") if STATUS_FILE.exists() else ""
    state = "unknown"
    total = ""
    for line in text.splitlines():
        if line.startswith("State:"):
            state = line.split(":", 1)[1].strip()
        elif line.startswith("Total:"):
            total = line.strip()
    return {
        "text": text.strip(),
        "state": state,
        "total": total,
        "ready": state == "DONE",
        "bytes": folder_bytes(CKPT_DIR),
    }


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace")


def host_ram_available_gb() -> float:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return 0.0
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            # kB
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def lock_path(gpu: int) -> Path:
    return LOCK_FILES[int(gpu)]


def read_lock(gpu: int | None = None) -> dict | None:
    """Read one GPU lock, or the newest of either (gpu=None)."""
    if gpu is not None:
        path = lock_path(gpu)
        if not path.exists():
            # Migrate legacy single lock onto gpu0 if it looks like gpu0.
            if gpu == 0 and LEGACY_LOCK.exists():
                path = LEGACY_LOCK
            else:
                return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "pid" not in data:
            return None
        data.setdefault("gpu", gpu)
        return data

    infos = []
    for g in (0, 1):
        info = read_lock(g)
        if info:
            infos.append(info)
    if not infos and LEGACY_LOCK.exists():
        try:
            data = json.loads(LEGACY_LOCK.read_text())
            if isinstance(data, dict) and "pid" in data:
                data.setdefault("gpu", 0)
                infos.append(data)
        except json.JSONDecodeError:
            pass
    if not infos:
        return None
    # Prefer a still-running job; else most recently started.
    running = [i for i in infos if i.get("status") == "running" and pid_alive(int(i["pid"]))]
    pool = running or infos
    return max(pool, key=lambda i: i.get("started", ""))


def write_lock(gpu: int, data: dict) -> None:
    data = dict(data)
    data["gpu"] = int(gpu)
    lock_path(gpu).write_text(json.dumps(data, indent=2))
    # Keep legacy path in sync for any external watchers.
    if gpu == 0:
        LEGACY_LOCK.write_text(json.dumps(data, indent=2))


def gpu_busy(gpu: int) -> bool:
    info = read_lock(gpu)
    return bool(info and info.get("status") == "running" and pid_alive(int(info["pid"])))


def any_job_running() -> bool:
    return gpu_busy(0) or gpu_busy(1)


def running_job_count() -> int:
    return int(gpu_busy(0)) + int(gpu_busy(1))


def safe_to_kill(pid: int) -> bool:
    command = cmdline(pid)
    blocked = ("download-wan22", "Wan2.2-TI2V-5B/resolve", "wan22.lock")
    return not any(token in command for token in blocked)


def kill_job(gpu: int | None = None) -> str:
    targets = [gpu] if gpu is not None else [0, 1]
    messages = []
    for g in targets:
        info = read_lock(g)
        if not info or info.get("status") != "running":
            continue
        pid = int(info.get("pid", 0))
        if not pid_alive(pid):
            info["status"] = "killed"
            info["ended"] = datetime.now().isoformat(timespec="seconds")
            write_lock(g, info)
            messages.append(f"GPU {g}: locked process had already exited.")
            continue
        if not safe_to_kill(pid):
            messages.append(f"GPU {g}: refused to kill non-generation process.")
            continue
        pgid = int(info.get("pgid") or pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                messages.append(f"GPU {g}: could not stop pid {pid}: {exc}")
                continue
        for _ in range(20):
            if not pid_alive(pid):
                break
            time.sleep(0.25)
        if pid_alive(pid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        info["status"] = "killed"
        info["ended"] = datetime.now().isoformat(timespec="seconds")
        write_lock(g, info)
        messages.append(f"GPU {g}: stopped generation job {pid}.")
    return " ".join(messages) if messages else "No generation job is locked."


def query_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "temp": parts[2],
                "util": parts[3],
                "used": int(float(parts[4])),
                "total": int(float(parts[5])),
            }
        )
    return gpus


def pick_free_gpu() -> int | None:
    """Pick the unlocked GPU with the most free VRAM. None if both busy."""
    gpus = {g["index"]: g for g in query_gpus()}
    candidates = []
    for idx in (0, 1):
        if gpu_busy(idx):
            continue
        used = gpus.get(idx, {}).get("used", 0)
        total = gpus.get(idx, {}).get("total", 20480)
        free = total - used
        candidates.append((free, -used, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def resolve_mode_to_gpu(mode: str) -> tuple[int | None, str]:
    """Return (gpu_index, error_message). error set when job cannot start."""
    if mode == "dual":
        return None, "Dual-GPU FSDP is disabled on this 29 GB host."
    if mode == "gpu0":
        if gpu_busy(0):
            return None, "GPU 0 is already generating a video."
        return 0, ""
    if mode == "gpu1":
        if gpu_busy(1):
            return None, "GPU 1 is already generating a video."
        return 1, ""
    # auto
    gpu = pick_free_gpu()
    if gpu is None:
        return None, "Both GPUs are busy."
    if running_job_count() >= 1:
        avail = host_ram_available_gb()
        if avail < MIN_HOST_RAM_GB_FOR_PARALLEL:
            return None, (
                f"A job is already running and only {avail:.1f} GB host RAM is "
                f"available (need ≥ {MIN_HOST_RAM_GB_FOR_PARALLEL:.0f} GB free "
                "to start a second parallel job)."
            )
    return gpu, ""


def render_gpus() -> None:
    gpus = query_gpus()
    if not gpus:
        st.caption("nvidia-smi did not return GPU stats.")
        return
    for gpu in gpus:
        busy = gpu_busy(gpu["index"])
        tag = " · BUSY" if busy else " · free"
        st.markdown(
            f"**GPU {gpu['index']}** · {gpu['temp']}°C · {gpu['util']}% util{tag}"
        )
        st.progress(min(gpu["used"] / max(gpu["total"], 1), 1.0))
        st.caption(f"{gpu['used']} / {gpu['total']} MiB")
    st.caption(f"Host RAM available: {host_ram_available_gb():.1f} GB")


def latest_percent(text: str) -> int | None:
    matches = re.findall(r"(\d+)%\|", text.replace("\r", "\n"))
    if not matches:
        return None
    return max(0, min(100, int(matches[-1])))


def log_tail(path: Path, limit: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    return "\n".join(lines[-limit:])


def combined_log_tail(limit: int = 100) -> str:
    chunks = []
    for g in (0, 1):
        text = log_tail(GEN_LOGS[g], limit // 2)
        if text.strip():
            chunks.append(f"===== GPU {g} =====\n{text}")
    if not chunks:
        return log_tail(GEN_LOG, limit)
    return "\n\n".join(chunks)


def mark_finished(gpu: int, info: dict) -> dict:
    save = Path(info.get("save_file", ""))
    log = log_tail(GEN_LOGS.get(gpu, GEN_LOG), 120)
    failed = (not save.exists()) or save.stat().st_size < 1024 or "Traceback" in log
    info["status"] = "failed" if failed else "finished"
    info["ended"] = datetime.now().isoformat(timespec="seconds")
    write_lock(gpu, info)
    return info


def build_command(gpu: int, prompt: str, size: str, frames: int, seed: int,
                  steps: int, guidance: float, image: Path | None, save_file: Path,
                  teacache_thresh: float = 0.0, use_sage: bool = False) -> tuple[list[str], dict]:
    common = [
        "--task", "ti2v-5B",
        "--size", size,
        "--frame_num", str(frames),
        "--ckpt_dir", str(CKPT_DIR),
        "--convert_model_dtype",
        "--prompt", prompt,
        "--base_seed", str(seed),
        "--sample_steps", str(steps),
        "--sample_guide_scale", str(guidance),
        "--teacache_thresh", f"{float(teacache_thresh):.4f}",
        "--save_file", str(save_file),
    ]
    if image is not None:
        common.extend(["--image", str(image)])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # One physical GPU per process — DiT samples, then VAE decodes on the
    # same card once DiT has left VRAM. Audio is a second process.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if use_sage:
        env["WAN_USE_SAGEATTN"] = "1"
    else:
        env.pop("WAN_USE_SAGEATTN", None)
    cmd = [
        str(PYTHON),
        str(GENERATE),
        *common,
        "--offload_model", "True",
        "--t5_cpu",
    ]
    return cmd, env


def resolve_audio_gpu(choice: str, video_gpu: int | None) -> int:
    if choice == "GPU 0":
        return 0
    if choice == "GPU 1":
        return 1
    if video_gpu is not None:
        return 1 - int(video_gpu)
    free = pick_free_gpu()
    return 1 if free is None else free


def video_has_audio(path: Path) -> bool:
    """True when the MP4 already has an audio stream."""
    ffprobe = shutil_which_ffprobe()
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True,
        )
        return bool(proc.stdout.strip())
    try:
        import av
        with av.open(str(path)) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


def shutil_which_ffprobe() -> str | None:
    import shutil
    found = shutil.which("ffprobe")
    if found:
        return found
    bundled = ROOT / "bin" / "ffprobe"
    return str(bundled) if bundled.exists() else None


def run_add_audio(video: Path, prompt: str, output: Path, audio_gpu: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = str(audio_gpu)
    cmd = [
        str(PYTHON), str(ADD_AUDIO),
        "--video", str(video),
        "--prompt", prompt,
        "--output", str(output),
        "--device", "cuda:0",
    ]
    return subprocess.run(
        cmd, cwd=str(WAN_ROOT), env=env, capture_output=True, text=True)


def start_job(gpu: int, prompt: str, size: str, frames: int, seed: int,
              steps: int, guidance: float, image: Path | None,
              teacache_thresh: float = 0.0, use_sage: bool = False,
              with_audio: bool = False, audio_prompt: str = "",
              audio_gpu: int = 1) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", prompt).strip("-")[:40] or "clip"
    save_file = OUTPUT_DIR / f"{stamp}_gpu{gpu}_{slug}.mp4"
    silent_file = save_file.with_name(save_file.stem + "_silent.mp4") if with_audio else save_file
    cmd, env = build_command(
        gpu, prompt, size, frames, seed, steps, guidance, image, silent_file,
        teacache_thresh=teacache_thresh, use_sage=use_sage)
    log_path = GEN_LOGS[gpu]
    log_handle = log_path.open("w", encoding="utf-8")
    if with_audio:
        spoken = audio_prompt.strip() or prompt
        audio_cmd = [
            str(PYTHON), str(ADD_AUDIO),
            "--video", str(silent_file),
            "--prompt", spoken,
            "--output", str(save_file),
            "--device", "cuda:0",
        ]
        script = "\n".join([
            "set -euo pipefail",
            "echo STAGE video",
            "export CUDA_VISIBLE_DEVICES=" + str(gpu),
            " ".join(shlex.quote(part) for part in cmd),
            "echo STAGE audio",
            "export CUDA_VISIBLE_DEVICES=" + str(audio_gpu),
            "set +e",
            " ".join(shlex.quote(part) for part in audio_cmd),
            "audio_status=$?",
            "set -e",
            'if [ "$audio_status" -ne 0 ]; then',
            "  echo \"Audio synthesis failed ($audio_status); keeping the silent video.\"",
            f"  cp {shlex.quote(str(silent_file))} {shlex.quote(str(save_file))}",
            "  exit 0",
            "fi",
        ]) + "\n"
        log_handle.write(script + "\n")
        log_handle.flush()
        GEN_LOG.write_text(f"(see {log_path.name})\n" + script)
        # CUDA_VISIBLE_DEVICES is set per stage inside the script.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=str(WAN_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        command = ["bash", "-c", script]
    else:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        GEN_LOG.write_text(f"(see {log_path.name})\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(WAN_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        command = cmd
    write_lock(gpu, {
        "pid": proc.pid,
        "pgid": proc.pid,
        "status": "running",
        "mode": f"gpu{gpu}",
        "gpu": gpu,
        "audio_gpu": audio_gpu if with_audio else None,
        "started": datetime.now().isoformat(timespec="seconds"),
        "save_file": str(save_file),
        "prompt": prompt,
        "audio_prompt": audio_prompt if with_audio else "",
        "size": size,
        "frames": frames,
        "teacache_thresh": teacache_thresh,
        "use_sage": use_sage,
        "with_audio": with_audio,
        "command": command,
        "log_file": str(log_path),
    })


def save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded.name).name
    dest = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    dest.write_bytes(uploaded.getvalue())
    return dest


def show_player(path: Path, key_prefix: str) -> None:
    st.video(str(path))
    st.download_button(
        "Download MP4 Video",
        data=path.read_bytes(),
        file_name=path.name,
        mime="video/mp4",
        type="primary",
        key=f"{key_prefix}-download",
    )


st.set_page_config(page_title="VideoAI", page_icon="🎬", layout="wide")
st.title("VideoAI")
st.caption("Wan 2.2 TI2V-5B · text-to-video and image-to-video")

status = read_status()
if status["ready"]:
    st.success(f"Model Ready · weights {human_size(status['bytes'])}")
else:
    st.warning(
        f"Model weights are still downloading. Generation stays off until the status is DONE.\n\n"
        f"State: {status['state']}\n\n"
        f"Downloaded: {human_size(status['bytes'])}"
        + (f"\n\n{status['total']}" if status["total"] else "")
    )
    if st.button("Refresh Status"):
        st.rerun()

with st.sidebar:
    st.header("Speed")
    fast_mode = st.checkbox(
        "⚡ Fast Sampling Mode (TeaCache + SageAttention)",
        value=True,
        help="TeaCache skips redundant diffusion steps. SageAttention uses "
        "the RTX 4000 Ada's FP8/INT8 tensor cores for full self-attention.",
    )
    teacache_thresh = st.slider(
        "TeaCache threshold",
        min_value=0.0,
        max_value=0.30,
        value=0.16,
        step=0.01,
        disabled=not fast_mode,
        help="0.0 = Off / Exact, 0.15 = Fast Balanced, 0.22 = Ultra Fast. "
        "Fast Sampling Mode applies this threshold and sets WAN_USE_SAGEATTN=1.",
    )
    if not fast_mode:
        teacache_thresh = 0.0
    st.divider()
    st.header("GPUs")
    video_gpu_label = st.radio(
        "Video generation GPU",
        list(VIDEO_GPU_CHOICES),
        index=0,
        help="GPU 0 is the default. Auto Load-Balance picks the freest unlocked "
        "card. VAE decode runs on that same GPU after DiT is unloaded.",
    )
    mode = VIDEO_GPU_CHOICES[video_gpu_label]
    audio_gpu_label = st.radio(
        "Audio generation GPU",
        AUDIO_GPU_CHOICES,
        index=0,
        help="GPU 1 is the default so video sampling on GPU 0 is not interrupted. "
        "Auto Load-Balance uses the other card.",
    )
    st.divider()
    st.header("Audio")
    with_audio = st.checkbox(
        "🔊 Generate Synchronized Audio with MMAudio (Runs on GPU 1 in ~5 sec)",
        value=True,
    )
    audio_prompt = st.text_input(
        "Audio Prompt (leave blank to automatically use the Video Prompt)",
        value="",
    )

    @st.fragment(run_every=2)
    def live_sidebar():
        render_gpus()
        for g in (0, 1):
            info = read_lock(g)
            if not info or info.get("status") != "running":
                continue
            if pid_alive(int(info["pid"])):
                st.info(f"GPU {g} generating (pid {info['pid']}).")
                log = log_tail(GEN_LOGS[g], 200)
                percent = latest_percent(log)
                st.progress((percent or 0) / 100)
                if "STAGE audio" in log or "MMAudio synthesis" in log:
                    st.caption(f"GPU {g}: video ready — MMAudio on the audio GPU.")
                elif "GPU VAE decode finished" in log:
                    st.caption(f"GPU {g}: VAE done — writing MP4…")
                elif "Moving VAE to" in log or "Moving DiT to CPU" in log:
                    st.caption(f"GPU {g}: DiT unloaded — GPU VAE decode running.")
                elif percent is None:
                    st.caption(f"GPU {g}: loading model…")
                else:
                    st.caption(f"GPU {g}: sampling {percent}%")
            else:
                mark_finished(g, info)
                st.rerun(scope="app")
            if info.get("status") == "failed":
                st.error(f"GPU {g}: last generation failed.")
            elif info.get("status") == "killed":
                st.warning(f"GPU {g}: last generation was stopped.")

    live_sidebar()
    kill_cols = st.columns(2)
    if kill_cols[0].button("Kill GPU 0"):
        st.warning(kill_job(0))
        st.rerun()
    if kill_cols[1].button("Kill GPU 1"):
        st.warning(kill_job(1))
        st.rerun()
    if st.button("Force Unlock / Kill All"):
        st.warning(kill_job(None))
        st.rerun()

if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT

st.subheader("Prompt")
preset_cols = st.columns(len(PRESETS))
for col, (label, text) in zip(preset_cols, PRESETS):
    if col.button(label, use_container_width=True):
        st.session_state.prompt = text
        st.rerun()
prompt = st.text_area("Describe the video", key="prompt", height=110)

left, right = st.columns(2)
with left:
    aspect = st.radio(
        "Aspect ratio",
        ["Landscape 720p (1280*704)", "Portrait 720p (704*1280)"],
        horizontal=True,
    )
    size = "1280*704" if aspect.startswith("Landscape") else "704*1280"
with right:
    duration_label = st.selectbox("Duration", list(DURATION_OPTIONS))
    frames = DURATION_OPTIONS[duration_label]
    if frames == 0:
        frames = st.selectbox("Custom frame count (4n+1)", FRAME_CHOICES, index=1)

with st.expander("Advanced settings"):
    seed = st.number_input("Seed (-1 = random)", value=-1, step=1)
    steps = st.slider("Sampling steps", min_value=10, max_value=50, value=20)
    guidance = st.slider("Guidance scale", min_value=1.0, max_value=12.0, value=5.0, step=0.5)

image_file = st.file_uploader(
    "Optional starting image (turns this into image-to-video)",
    type=["jpg", "jpeg", "png", "webp"],
)
if image_file is not None:
    st.image(image_file, caption="This image will be animated.", width=360)
    st.caption("Task: image-to-video")
else:
    st.caption("Task: text-to-video")

target_gpu, block_reason = resolve_mode_to_gpu(mode)
audio_gpu = resolve_audio_gpu(audio_gpu_label, target_gpu)
can_start = status["ready"] and target_gpu is not None
generate = st.button(
    "Generate Video",
    type="primary",
    disabled=not can_start,
)
if target_gpu is not None:
    audio_note = f" Audio on GPU {audio_gpu}." if with_audio else ""
    speed_note = (
        f" TeaCache {teacache_thresh:.2f} + SageAttention."
        if fast_mode else " Exact sampling."
    )
    st.caption(f"Will dispatch video to GPU {target_gpu}.{audio_note}{speed_note}")
elif block_reason:
    st.caption(block_reason)

if generate:
    if not status["ready"]:
        st.error("The model download is not DONE yet.")
    else:
        gpu, err = resolve_mode_to_gpu(mode)
        if err or gpu is None:
            st.error(err or "No free GPU.")
        else:
            image_path = save_upload(image_file) if image_file is not None else None
            try:
                start_job(
                    gpu, prompt.strip(), size, int(frames), int(seed), int(steps),
                    float(guidance), image_path,
                    teacache_thresh=float(teacache_thresh),
                    use_sage=bool(fast_mode),
                    with_audio=bool(with_audio),
                    audio_prompt=audio_prompt.strip(),
                    audio_gpu=resolve_audio_gpu(audio_gpu_label, gpu),
                )
            except OSError as exc:
                st.error(f"Could not start generation: {exc}")
            else:
                st.rerun()

# Refresh finished state for both GPUs
for g in (0, 1):
    info = read_lock(g)
    if info and info.get("status") == "running" and not pid_alive(int(info.get("pid", 0))):
        mark_finished(g, info)

info = read_lock()  # newest / running for result panel
st.subheader("Result")
if info and info.get("status") == "finished":
    result = Path(info["save_file"])
    if result.exists():
        show_player(result, "latest")
    else:
        st.error("The job was marked finished, but the MP4 is missing.")
elif info and info.get("status") == "failed":
    st.error("Generation failed.")
    if info.get("mode") == "dual":
        st.markdown(
            "Dual-GPU is not supported on this 29 GB host. Switch to "
            "**Auto-select**, **GPU 0**, or **GPU 1** and run again."
        )
elif info and info.get("status") == "running":
    st.info("Generation is running. Sampling progress and the log update every couple of seconds.")
else:
    st.caption("A finished video will play here.")

with st.expander("Generation log", expanded=bool(info and info.get("status") in {"running", "failed"})):
    st.code(combined_log_tail(120) or "No generation has been started in this UI yet.", language="text")

st.subheader("Generated videos")
videos = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
if not videos:
    st.caption("No videos yet. They will be saved in /home/rangoondiscount/VideoAI/outputs.")
else:
    gallery_audio_gpu = resolve_audio_gpu(audio_gpu_label, None)
    for video in videos:
        stamp = datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        silent = not video_has_audio(video)
        kind = "silent" if silent else "with audio"
        st.markdown(
            f"**{video.name}** · {stamp} · {human_size(video.stat().st_size)} · {kind}"
        )
        show_player(video, f"gallery-{video.name}")
        if silent:
            gallery_prompt = st.text_input(
                "Audio prompt",
                value="",
                key=f"audio-prompt-{video.name}",
                placeholder="Leave blank to describe ambience from the file name",
            )
            if st.button("🔊 Add Audio (MMAudio)", key=f"add-audio-{video.name}"):
                spoken = gallery_prompt.strip() or video.stem.replace("_", " ")
                out = video.with_name(video.stem + "_mmaudio.mp4")
                if gpu_busy(gallery_audio_gpu) and gpu_busy(1 - gallery_audio_gpu):
                    st.error("Both GPUs are busy. Wait for generation to finish.")
                else:
                    use_gpu = gallery_audio_gpu
                    if gpu_busy(use_gpu):
                        use_gpu = 1 - use_gpu
                    with st.spinner(f"MMAudio on GPU {use_gpu}…"):
                        result = run_add_audio(video, spoken, out, use_gpu)
                    if result.returncode == 0 and out.exists():
                        st.success(f"Saved {out.name}")
                        st.rerun()
                    else:
                        st.error("MMAudio failed.")
                        st.code((result.stderr or result.stdout or "no output")[-2000:])
        if st.button("Delete", key=f"delete-{video.name}"):
            video.unlink(missing_ok=True)
            st.rerun()
        st.divider()
