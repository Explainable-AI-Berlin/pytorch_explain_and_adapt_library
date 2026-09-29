"""Eval module — re-exports from submodules."""

from .ref_iqa import calculate_psnr, calculate_lpips, calculate_ssim
from .fid import calculate_rfid

try:
    from .clipscore import CLIPScoreEvaluator
except ModuleNotFoundError:
    CLIPScoreEvaluator = None

try:
    from .vqascore import VQAScoreEvaluator
except ModuleNotFoundError:
    VQAScoreEvaluator = None

try:
    from .geneval import GenEvalEvaluator
except ModuleNotFoundError:
    GenEvalEvaluator = None

try:
    from .dpgbench import DPGEvaluator
except ModuleNotFoundError:
    DPGEvaluator = None

from .reconstruction import compute_reconstruction_metrics, evaluate_reconstruction_distributed
from .generation import evaluate_generation_distributed, evaluate_image_set

__all__ = [
    "calculate_psnr", "calculate_lpips", "calculate_ssim",
    "calculate_rfid",
    "CLIPScoreEvaluator", "VQAScoreEvaluator", "GenEvalEvaluator", "DPGEvaluator",
    "compute_reconstruction_metrics", "evaluate_reconstruction_distributed",
    "evaluate_generation_distributed", "evaluate_image_set",
]
