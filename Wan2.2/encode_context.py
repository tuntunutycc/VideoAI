#!/usr/bin/env python3
"""Single-process T5 encode for dual-GPU runs (avoids 2× torchrun RAM peak)."""
import argparse
import logging
import sys

import torch

import wan
from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save_context", required=True)
    parser.add_argument("--n_prompt", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    cfg = WAN_CONFIGS[args.task]
    n_prompt = args.n_prompt or cfg.sample_neg_prompt

    logging.info("Loading T5 on CPU (single process)...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=f"{args.ckpt_dir}/{cfg.t5_checkpoint}",
        tokenizer_path=f"{args.ckpt_dir}/{cfg.t5_tokenizer}",
        shard_fn=None,
    )
    logging.info("Encoding prompts...")
    context = text_encoder([args.prompt], torch.device("cpu"))
    context_null = text_encoder([n_prompt], torch.device("cpu"))
    text_encoder.unload_model()
    del text_encoder
    payload = {
        "context": [t.cpu().contiguous() for t in context],
        "context_null": [t.cpu().contiguous() for t in context_null],
        "prompt": args.prompt,
        "n_prompt": n_prompt,
    }
    torch.save(payload, args.save_context)
    logging.info(f"Wrote context to {args.save_context}")


if __name__ == "__main__":
    main()
