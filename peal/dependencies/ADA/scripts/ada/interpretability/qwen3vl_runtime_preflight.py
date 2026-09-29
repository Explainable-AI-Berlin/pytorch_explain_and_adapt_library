from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight a pinned Qwen3-VL runtime and local snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(args.snapshot, local_files_only=True)
    assert getattr(config, "model_type", "") == "qwen3_vl", getattr(config, "model_type", "")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.snapshot,
        local_files_only=True,
        dtype="auto",
        device_map="auto",
    )
    payload = {
        "snapshot": str(args.snapshot),
        "transformers_version": transformers.__version__,
        "transformers_path": transformers.__file__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_type": getattr(config, "model_type", ""),
        "config_class": type(config).__name__,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "first_parameter_device": str(next(model.parameters()).device),
    }
    out = args.output_dir / "preflight.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (args.output_dir / "COMPLETED").write_text("ok\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
