# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import free_model, shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        teacache_thresh=0.0,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.checkpoint_dir = checkpoint_dir
        self._use_sp_flag = use_sp
        self._dit_fsdp_flag = dit_fsdp
        self._convert_model_dtype = convert_model_dtype
        self._shard_fn = partial(shard_model, device_id=device_id)

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = self._shard_fn
        # Multi-GPU + t5_cpu: only rank 0 holds T5; embeddings are broadcast.
        self._t5_rank0_only = (
            bool(t5_cpu) and dist.is_initialized() and dist.get_world_size() > 1)
        # Defer T5/VAE/DiT on multi-GPU so we never hold two big weights at once.
        self._dit_deferred = bool(dit_fsdp or use_sp) and dist.is_initialized()
        self._t5_deferred = self._dit_deferred
        self._t5_fsdp = t5_fsdp
        self.text_encoder = None
        if not self._t5_deferred:
            if self._t5_rank0_only and rank != 0:
                self.text_encoder = None
            else:
                self.text_encoder = T5EncoderModel(
                    text_len=config.text_len,
                    dtype=config.t5_dtype,
                    device=torch.device('cpu'),
                    checkpoint_path=os.path.join(
                        checkpoint_dir, config.t5_checkpoint),
                    tokenizer_path=os.path.join(
                        checkpoint_dir, config.t5_tokenizer),
                    shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        # Wan2.2 TI2V VAE z_dim is 48; defer VAE load until encode/decode so
        # DiT sampling keeps the full 20 GB card.
        self.z_dim = 48
        self.vae = None
        self._vae_device = self.device

        self._dit_fsdp = dit_fsdp
        self._use_sp = use_sp
        self.teacache_thresh = float(teacache_thresh or 0.0)
        self.model = None
        if not self._dit_deferred:
            # Single-GPU path: DiT only — VAE loads after DiT is dropped.
            self._load_dit(
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _load_dit(self, use_sp, dit_fsdp, convert_model_dtype):
        """Load DiT; non-zero FSDP ranks build empty weights, then sync from rank 0."""
        logging.info(f"Creating WanModel from {self.checkpoint_dir}")
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp and dist.is_initialized() and self.rank != 0:
            # Meta avoids a second ~10 GB CPU copy (29 GB host). FSDP syncs
            # real weights from rank 0. Requires NCCL_P2P_DISABLE=1 on this box.
            config = WanModel.load_config(self.checkpoint_dir)
            orig_init = WanModel.init_weights
            WanModel.init_weights = lambda self: None
            try:
                with torch.device('meta'):
                    model = WanModel.from_config(config)
            finally:
                WanModel.init_weights = orig_init
            logging.info("Rank %s: meta DiT ready for FSDP sync", self.rank)
        else:
            try:
                model = WanModel.from_pretrained(
                    self.checkpoint_dir,
                    torch_dtype=self.param_dtype,
                    low_cpu_mem_usage=True)
            except TypeError:
                model = WanModel.from_pretrained(self.checkpoint_dir)
            logging.info("Rank %s: DiT weights loaded", self.rank)
        if dist.is_initialized():
            dist.barrier()
        shard_fn = self._shard_fn
        if dit_fsdp and dist.is_initialized() and self.rank != 0:
            # Materialize meta params onto this GPU; FSDP then syncs from rank 0.
            def _meta_param_init(module):
                device = torch.device(f"cuda:{self.device.index if self.device.index is not None else 0}")
                module.to_empty(device=device, recurse=False)

            shard_fn = partial(
                shard_model,
                device_id=self.device.index if self.device.index is not None else 0,
                param_init_fn=_meta_param_init)
        self.model = self._configure_model(
            model=model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        self._bind_teacache(self.model)

    def _ensure_dit_loaded(self):
        if self.model is not None:
            return
        self._load_dit(
            use_sp=self._use_sp_flag,
            dit_fsdp=self._dit_fsdp_flag,
            convert_model_dtype=self._convert_model_dtype)

    def _load_context_file(self, context_file):
        """Load precomputed T5 embeddings (written by encode_context.py).

        Every rank reads the same local file — NCCL cannot broadcast Python
        object lists, which previously hung dual-GPU runs for 10+ minutes.
        """
        logging.info(f"Loading cached text context from {context_file}")
        payload = torch.load(context_file, map_location='cpu', weights_only=False)
        context = [t.contiguous() for t in payload['context']]
        context_null = [t.contiguous() for t in payload['context_null']]
        # Keep embeddings on CPU until after DiT FSDP init — early .cuda() +
        # NCCL barrier races hung dual-GPU jobs for the full NCCL timeout.
        logging.info(
            "Cached context ready on CPU "
            f"(shapes={[tuple(t.shape) for t in context]})")
        return context, context_null

    def _ensure_t5_loaded(self):
        if self._t5_rank0_only and self.rank != 0:
            return
        if self.text_encoder is not None and self.text_encoder.model is not None:
            return
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_checkpoint),
                tokenizer_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_tokenizer),
                shard_fn=self._shard_fn if self._t5_fsdp else None)
        else:
            self.text_encoder.ensure_model_loaded()

    def _ensure_vae_loaded(self, device=None):
        if self.rank != 0:
            return
        if self.vae is not None:
            return
        dev = device if device is not None else self._vae_device
        # bf16 halves activation memory vs fp32 — critical for 720p on 20 GB.
        logging.info(f"Loading bf16 VAE on {dev}")
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(
                self.checkpoint_dir, self.config.vae_checkpoint),
            device=dev,
            dtype=torch.bfloat16)
        self.z_dim = self.vae.model.z_dim

    def _encode_text(self, input_prompt, n_prompt, offload_model):
        """Encode prompts; on multi-GPU + t5_cpu, only rank 0 runs T5."""
        if self._t5_rank0_only:
            # Avoid NCCL object broadcast: rank 0 writes a temp file, all ranks read.
            ctx_path = os.path.join(
                "/tmp", f"wan_t5_ctx_{os.getpid() if self.rank == 0 else 'x'}.pt")
            if dist.is_initialized():
                # Share one path from rank 0 via a tiny CUDA long is awkward; use fixed path.
                ctx_path = f"/tmp/wan_t5_ctx_rank0_{os.environ.get('MASTER_PORT', '29500')}.pt"
            if self.rank == 0:
                self._ensure_t5_loaded()
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                self.text_encoder.unload_model()
                self.text_encoder = None
                gc.collect()
                torch.save({
                    'context': [t.cpu().contiguous() for t in context],
                    'context_null': [t.cpu().contiguous() for t in context_null],
                }, ctx_path)
            if dist.is_initialized():
                dist.barrier()
            payload = torch.load(ctx_path, map_location='cpu', weights_only=False)
            context = [t for t in payload['context']]
            context_null = [t for t in payload['context_null']]
            if self.rank == 0:
                try:
                    os.remove(ctx_path)
                except OSError:
                    pass
            if dist.is_initialized():
                dist.barrier()
            return context, context_null

        self._ensure_t5_loaded()
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        # Free T5 before DiT sampling / VAE so 29 GB hosts survive.
        if offload_model or self._dit_deferred:
            if self._t5_fsdp and self.text_encoder is not None \
                    and self.text_encoder.model is not None \
                    and dist.is_initialized():
                try:
                    free_model(self.text_encoder.model)
                except Exception:
                    self.text_encoder.unload_model()
                else:
                    self.text_encoder.model = None
                    gc.collect()
                    torch.cuda.empty_cache()
            elif self.text_encoder is not None:
                self.text_encoder.unload_model()
        return context, context_null

    def _dit_module(self, model=None):
        """Unwrap FSDP so TeaCache state lives on the WanModel itself."""
        inner = self.model if model is None else model
        seen = set()
        while inner is not None and hasattr(inner, "module") and id(inner) not in seen:
            seen.add(id(inner))
            if hasattr(inner, "reset_teacache"):
                return inner
            inner = inner.module
        return inner

    def _bind_teacache(self, model=None):
        inner = self._dit_module(model)
        if inner is None or not hasattr(inner, "reset_teacache"):
            return
        inner.teacache_thresh = float(self.teacache_thresh or 0.0)
        inner.reset_teacache()

    def _reset_teacache(self, num_steps):
        inner = self._dit_module()
        if inner is None or not hasattr(inner, "reset_teacache"):
            return
        inner.teacache_thresh = float(self.teacache_thresh or 0.0)
        inner.reset_teacache(num_steps)

    def _log_teacache(self):
        inner = self._dit_module()
        if inner is None or not getattr(inner, "teacache_thresh", 0):
            return
        logging.info(
            "TeaCache skipped %s/%s transformer passes (thresh=%.3f)",
            getattr(inner, "teacache_skipped", 0),
            getattr(inner, "teacache_call_count", 0),
            float(inner.teacache_thresh))

    def _park_vae_on_cpu(self):
        """Keep the VAE off the DiT GPU for the whole sampling loop."""
        if self.vae is None:
            return
        try:
            device = next(self.vae.model.parameters()).device
        except StopIteration:
            return
        if device.type == "cpu":
            return
        logging.info("Parking VAE on CPU before DiT sampling")
        self.vae.to_device("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_for_vae_decode(self, offload_model):
        """Unload DiT and T5 from GPU so VAE decode can run there.

        Single-GPU: move DiT to CPU (weights leave VRAM). If the 29 GB host
        cannot accept that copy, delete the module — VRAM is free either way.
        FSDP/SP still frees sharded params in place.
        """
        if self.text_encoder is not None:
            self.text_encoder.unload_model()
            self.text_encoder = None
        if self.model is None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        if self._dit_fsdp or self._use_sp or self.sp_size > 1:
            logging.info("Freeing FSDP/SP DiT before VAE decode")
            try:
                free_model(self.model)
            except Exception:
                del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
            return
        logging.info("Moving DiT to CPU before GPU VAE decode (offload_model=%s)",
                     offload_model)
        # Drop cached residuals (~320 MiB) before the weights move.
        inner = self._dit_module()
        if inner is not None and hasattr(inner, "reset_teacache"):
            inner.prev_residual_cond = None
            inner.prev_residual_uncond = None
            inner.prev_e0_cond = None
        try:
            self.model.to("cpu")
        except RuntimeError as exc:
            logging.warning(
                "DiT .to('cpu') failed (%s); deleting DiT to free VRAM", exc)
            try:
                del self.model
            except Exception:
                pass
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logging.info(
            "DiT unloaded before VAE decode: "
            f"{torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} GiB allocated")

    def _pick_vae_device(self):
        """Prefer a free sibling GPU; otherwise reuse the DiT GPU after unload."""
        override = os.environ.get("WAN_VAE_DEVICE", "").strip()
        if override:
            return torch.device(override)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            dit_idx = self.device.index if self.device.index is not None else 0
            other = 1 if dit_idx == 0 else 0
            logging.info(
                "Using cuda:%s for VAE decode (DiT was on cuda:%s)", other, dit_idx)
            return torch.device(f"cuda:{other}")
        return self.device

    def _decode_latents(self, x0):
        """Decode latents on GPU after DiT unload.

        A full-frame OOM retries with spatial tiles on the same GPU. CPU
        decode is not used — it takes 10–20 minutes on this host.
        """
        zs = [u.detach().contiguous() for u in x0]
        del x0
        gc.collect()
        torch.cuda.empty_cache()

        vae_device = self._pick_vae_device()
        try:
            return self._decode_latents_gpu(zs, vae_device, spatial_tiles=False)
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                "GPU VAE decode OOM on %s; retrying with spatial tiles.",
                vae_device)
            torch.cuda.empty_cache()
            gc.collect()
            try:
                return self._decode_latents_gpu(
                    zs, vae_device, spatial_tiles=True)
            except torch.cuda.OutOfMemoryError:
                if vae_device == self.device:
                    raise
                logging.warning(
                    "Tiled VAE OOM on %s; retrying tiled decode on %s.",
                    vae_device, self.device)
                torch.cuda.empty_cache()
                gc.collect()
                return self._decode_latents_gpu(
                    zs, self.device, spatial_tiles=True)

    def _decode_latents_gpu(self, zs, device, spatial_tiles=False):
        """bf16 VAE on `device`. Weights return to CPU when decode finishes."""
        t0 = time.perf_counter()
        if self.vae is None:
            # Load on CPU first so DiT sampling never shares the card with VAE.
            self._ensure_vae_loaded(device=torch.device("cpu"))
        logging.info(
            "Moving VAE to %s for decode (spatial_tiles=%s); CUDA allocated "
            "%.2f GiB on DiT device",
            device, spatial_tiles,
            torch.cuda.memory_allocated(self.device) / (1024 ** 3))
        self.vae.to_device(device)
        zs_dev = [u.to(device, non_blocking=False) for u in zs]
        try:
            videos = self.vae.decode(zs_dev, spatial_tiles=spatial_tiles)
        finally:
            del zs_dev
            # Park VAE back on CPU so the card is free for the next job.
            try:
                self.vae.to_device("cpu")
            except Exception:
                logging.warning("Could not move VAE back to CPU", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        logging.info("GPU VAE decode finished in %.1fs on %s", elapsed, device)
        return videos

    def _decode_latents_cpu(self, zs):
        """Rebuild VAE on CPU so no CUDA state remains from a failed GPU decode."""
        zs_cpu = [u.cpu().contiguous() for u in zs]
        del zs
        self.vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        logging.info(
            "Loading fresh VAE on CPU for decode; CUDA allocated "
            f"{torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} GiB")
        self._ensure_vae_loaded(device=torch.device('cpu'))
        gc.collect()
        videos = self.vae.decode(zs_cpu)
        del zs_cpu
        gc.collect()
        return videos

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 context_file=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                context_file=context_file)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            context_file=context_file)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            context_file=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if dist.is_initialized():
            dist.barrier()
        if context_file:
            context, context_null = self._load_context_file(context_file)
        else:
            context, context_null = self._encode_text(
                input_prompt, n_prompt, offload_model)
        self._ensure_dit_loaded()
        # Move text embeddings to this rank's GPU after DiT is sharded.
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]
        if dist.is_initialized():
            dist.barrier()

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            self._park_vae_on_cpu()
            self._reset_teacache(len(timesteps))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            t_dit = time.perf_counter()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                del noise_pred_cond, noise_pred_uncond, noise_pred, temp_x0, latent_model_input, timestep, temp_ts
            dit_sec = time.perf_counter() - t_dit
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                if torch.cuda.is_available() else 0.0)
            logging.info(
                "DiT sampling finished in %.1fs (%.2fs/step), peak allocated %.2f GiB",
                dit_sec, dit_sec / max(len(timesteps), 1), peak_gib)
            self._log_teacache()
            x0 = [u.detach() for u in latents]
            del noise, latents, sample_scheduler, context, context_null, arg_c, arg_null, mask1, mask2

        # Decode outside autocast so DiT can fully leave the GPU first.
        self._prepare_for_vae_decode(offload_model)
        videos = self._decode_latents(x0) if self.rank == 0 else None
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            context_file=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if context_file:
            context, context_null = self._load_context_file(context_file)
        else:
            context, context_null = self._encode_text(
                input_prompt, n_prompt, offload_model)

        # Rank 0 encodes the image; other ranks receive the latent.
        if dist.is_initialized() and dist.get_world_size() > 1:
            z_list = [None]
            if self.rank == 0:
                self._ensure_vae_loaded(device=self.device)
                z_list = [[u.cpu() for u in self.vae.encode([img])]]
                # Drop GPU VAE before DiT to free VRAM/RAM on 20 GB cards.
                self.vae = None
                gc.collect()
                torch.cuda.empty_cache()
            dist.broadcast_object_list(z_list, src=0)
            z = [u.to(self.device) for u in z_list[0]]
        else:
            self._ensure_vae_loaded(device=self.device)
            z = self.vae.encode([img])
            # VAE stays on CPU while DiT samples. Decode moves it back later.
            self._park_vae_on_cpu()

        self._ensure_dit_loaded()

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            self._park_vae_on_cpu()
            self._reset_teacache(len(timesteps))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            t_dit = time.perf_counter()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent.detach()]
                del latent_model_input, timestep, noise_pred_cond, noise_pred_uncond, noise_pred, temp_x0

            dit_sec = time.perf_counter() - t_dit
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                if torch.cuda.is_available() else 0.0)
            logging.info(
                "DiT sampling finished in %.1fs (%.2fs/step), peak allocated %.2f GiB",
                dit_sec, dit_sec / max(len(timesteps), 1), peak_gib)
            self._log_teacache()
            del noise, latent, sample_scheduler, context, context_null, arg_c, arg_null

        self._prepare_for_vae_decode(offload_model)
        videos = self._decode_latents(x0) if self.rank == 0 else None
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
