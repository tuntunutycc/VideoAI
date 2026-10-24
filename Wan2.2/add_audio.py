#!/usr/bin/env python3
"""Synthesize synchronized audio with MMAudio and mux it onto a silent MP4.

Video frames are copied without re-encoding. Audio is AAC at 192 kbps.
The model runs on --device (default cuda:1), falling back to cuda:0 when
that index is not visible.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MMAUDIO_ROOT = Path("/home/rangoondiscount/VideoAI/MMAudio")
VIDEOAI_BIN = Path("/home/rangoondiscount/VideoAI/bin")

if str(MMAUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(MMAUDIO_ROOT))


def _ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = VIDEOAI_BIN / "ffmpeg"
    if bundled.exists():
        return str(bundled)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH, in VideoAI/bin, or via imageio-ffmpeg"
        ) from exc


def probe_duration_sec(video: Path) -> float:
    """Exact video duration via ffprobe, with PyAV as a fallback."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None and (VIDEOAI_BIN / "ffprobe").exists():
        ffprobe = str(VIDEOAI_BIN / "ffprobe")
    if ffprobe is not None:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video),
            ],
            check=True, capture_output=True, text=True)
        return float(proc.stdout.strip())
    import av
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0
    raise RuntimeError(f"Could not read duration of {video}")


def resolve_device(requested: str) -> str:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logging.warning("CUDA is not available; MMAudio will run on CPU")
        return "cpu"
    idx = 0
    if requested.startswith("cuda:"):
        idx = int(requested.split(":", 1)[1])
    if idx < 0 or idx >= torch.cuda.device_count():
        logging.warning(
            "Requested device %s is not visible (%s GPU(s)); using cuda:0",
            requested, torch.cuda.device_count())
        idx = 0
    return f"cuda:{idx}"


def synthesize(video: Path, prompt: str, negative_prompt: str, wav_path: Path,
               device: str, num_steps: int) -> None:
    import torch

    # Weight paths in ModelConfig are relative to the MMAudio repo.
    os.chdir(MMAUDIO_ROOT)
    from mmaudio.eval_utils import all_model_cfg, generate, load_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    video_duration = probe_duration_sec(video)
    logging.info("Video duration %.3fs — synthesizing on %s", video_duration, device)

    model = all_model_cfg["large_44k_v2"]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    dtype = torch.bfloat16
    t0 = time.perf_counter()
    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(
        model.model_path, map_location=device, weights_only=True))
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()
    logging.info("MMAudio weights ready in %.1fs", time.perf_counter() - t0)

    video_info = load_video(video, video_duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    duration = video_info.duration_sec
    seq_cfg.duration = duration
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

    t1 = time.perf_counter()
    # Encoders run under inference_mode and must not be fed into autograd.
    with torch.inference_mode():
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=4.5,
        )
    audio = audios.float().cpu()[0]
    logging.info(
        "MMAudio synthesis finished in %.1fs (44.1 kHz, %s samples)",
        time.perf_counter() - t1, audio.shape[-1])
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf
    waveform = audio.numpy()
    # Model returns channels first. soundfile wants time first, stereo.
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    else:
        waveform = waveform.T
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    target = int(round(video_duration * int(seq_cfg.sampling_rate)))
    if waveform.shape[0] < target:
        pad = np.zeros((target - waveform.shape[0], waveform.shape[1]), dtype=waveform.dtype)
        waveform = np.concatenate([waveform, pad], axis=0)
    elif waveform.shape[0] > target:
        waveform = waveform[:target]
    sf.write(str(wav_path), waveform, int(seq_cfg.sampling_rate))
    logging.info("Wrote %s (%d stereo samples, %.3fs)", wav_path, waveform.shape[0], video_duration)

    del audios, net, feature_utils
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mux(video: Path, wav: Path, output: Path) -> None:
    ffmpeg = _ffmpeg_bin()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(wav),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output),
    ]
    logging.info("Muxing (video copy, no re-encode): %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    logging.info("Wrote %s (%s bytes)", output, output.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add MMAudio to a silent MP4")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="music, speech, noise, distortion",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_steps", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Input video not found: {video}")
    if output == video:
        raise SystemExit("Refusing to overwrite the input video in place")

    device = resolve_device(args.device)
    wav_path = output.with_suffix(".mmaudio.wav")
    t0 = time.perf_counter()
    synthesize(
        video, args.prompt.strip(), args.negative_prompt.strip(),
        wav_path, device, args.num_steps)
    mux(video, wav_path, output)
    try:
        wav_path.unlink()
    except OSError:
        pass
    logging.info(
        "Audio synthesis and mux finished in %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
