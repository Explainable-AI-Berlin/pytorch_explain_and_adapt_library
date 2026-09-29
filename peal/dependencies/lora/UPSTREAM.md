<!-- Provenance record for a vendored third-party component of PEAL. -->

# LoRA fine-tuning script (HuggingFace Diffusers example)

**This directory is not part of PEAL.** It holds a single file copied from the
Diffusers example collection; credit belongs to the HuggingFace team.

| | |
|---|---|
| **Original authors** | The HuggingFace Inc. team |
| **Upstream repository** | <https://github.com/huggingface/diffusers> (`examples/text_to_image/train_text_to_image_lora.py`) |
| **Upstream license** | Apache-2.0 (header retained verbatim in the file) |

## Why PEAL vendors it

`train_text_to_image_lora.py` is called as `lora_finetune(...)` from PEAL's
Stable Diffusion and FLUX generators. Diffusers ships it as an example script,
not as part of the installable package, so it cannot be imported from the
`diffusers` distribution.

## How this copy differs from upstream

The script's `main()` was wrapped into an importable `lora_finetune(...)` entry
point so PEAL can call it in-process from a config instead of via the command
line. The Apache-2.0 license header is unchanged.
