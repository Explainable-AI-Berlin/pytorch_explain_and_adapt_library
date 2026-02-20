"""StableDiffusionAutoencoder — DiDAE with pretrained foundation models.

Uses Stable Diffusion 1.4 (via DDPM inversion) as the pixel-level encoder/decoder
and CLIP ViT-L/14 + SpLICE as the semantic encoder/decomposer. No training required.

Architecture:
    - Semantic encoder: CLIP ViT-L/14 image encoder → 768-dim embedding (z_sem)
    - Stochastic encoder: DDPM inversion forward process → noise path (wT, zs, wts)
    - Sparse dictionary: SpLICE decomposes z_sem into concept weights
    - Editing: DiDAE Algorithms 1 & 2 (reflect/project z_sem along dictionary components)
    - Decoder: DDPM inversion reverse process conditioned on modified z_sem
"""

import os
import copy
import math
from pathlib import Path
from typing import Tuple, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from diffusers import StableDiffusionPipeline, DDIMScheduler

from peal.generators.interfaces import (
    GeneratorConfig,
    InvertibleGenerator,
    EditCapableGenerator,
)
from peal.data.interfaces import DataConfig
from peal.data.dataset_factory import get_datasets
from peal.architectures.interfaces import TaskConfig
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig, SparseDictionary
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
    encode_text,
)
from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_classes import AttentionStore
from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_utils import (
    register_attention_control,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class StableDiffusionAutoencoderConfig(GeneratorConfig):
    """Configuration for the pretrained SD-based DiDAE generator."""
    generator_type: str = "StableDiffusionAutoencoder"
    model_id: str = "CompVis/stable-diffusion-v1-4"
    data: Union[str, DataConfig] = DataConfig()
    task_config: Union[TaskConfig, None] = None
    base_path: str = ""
    encoder_dimensions: int = 768  # CLIP ViT-L/14 dim

    # DDPM inversion parameters
    num_diffusion_steps: int = 100
    cfg_scale_src: float = 3.5
    cfg_scale_tar: float = 15.0
    eta: float = 1.0
    skip: int = 36

    # Sparse dictionary (SpLICE)
    sparse_dictionary: Union[str, SparseDictionaryConfig, None] = None

    # Visualization
    visualizations_per_component: Union[int, None] = 10


# ---------------------------------------------------------------------------
# CLIP ViT-L/14 Encoder wrapper
# ---------------------------------------------------------------------------

class CLIPImageEncoder(nn.Module):
    """Wraps OpenAI CLIP ViT-L/14 as a semantic encoder.

    Resizes input images to 224x224, normalizes them using CLIP's
    preprocessing, and returns 768-dim image embeddings.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to CLIP embeddings.

        Args:
            x: (B, 3, H, W) image tensor in [0, 1] or normalized range.
        Returns:
            (B, 768) L2-normalized CLIP image embeddings.
        """
        # Resize to CLIP's expected 224x224
        x_resized = torchvision.transforms.Resize(
            [224, 224], antialias=True
        )(x)

        # CLIP expects images normalized with specific mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=x.device).view(1, 3, 1, 1)
        x_norm = (x_resized - mean) / std

        with torch.no_grad():
            features = self.clip_model.encode_image(x_norm)

        return F.normalize(features.float(), dim=1)


# ---------------------------------------------------------------------------
# StableDiffusionAutoencoder
# ---------------------------------------------------------------------------

class StableDiffusionAutoencoder(InvertibleGenerator, EditCapableGenerator):
    """DiDAE generator using pretrained SD 1.4 + CLIP ViT-L/14 + SpLICE.

    Treats Stable Diffusion as a diffusion autoencoder where:
      - The semantic code z_sem comes from CLIP's image encoder
      - The stochastic code (noise path) comes from DDPM inversion
      - SpLICE provides disentangled concept decomposition of z_sem
      - Editing reflects z_sem along SpLICE dictionary components (DiDAE Alg. 2)
      - Decoding uses DDPM reverse process conditioned on the modified z_sem
    """

    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

        # --- Load datasets ---
        self.generator_datasets = get_datasets(self.config.data)
        if self.config.task_config is not None:
            for ds in self.generator_datasets:
                if ds is not None:
                    ds.task_config = self.config.task_config

        self.generator_dataset = self.generator_datasets[0] if self.generator_datasets else None

        # --- Load Stable Diffusion pipeline ---
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.config.model_id, subfolder="scheduler"
        )
        self.pipe.scheduler.set_timesteps(self.config.num_diffusion_steps)
        self.pipe.safety_checker = None

        # --- Setup CLIP ViT-L/14 semantic encoder ---
        self.encoder = CLIPImageEncoder(device=self.device)

        # --- Load sparse dictionary (SpLICE) if configured ---
        self.sparse_dictionary = None
        if self.config.sparse_dictionary is not None:
            self.load_sparse_dictionary()
            
            # Check if it was actually loaded from disk (has image_mean)
            # If not, it means the (default) weights file didn't exist yet, so we fit.
            if getattr(self.sparse_dictionary, "image_mean", None) is None:
                self.fit_sparse_dictionary()
                
            # Ensure CLIP weight identity by sharing the model instance
            if self.sparse_dictionary is not None and hasattr(self.sparse_dictionary, 'set_clip_model'):
                self.sparse_dictionary.set_clip_model(self.encoder.clip_model)

    def __setattr__(self, name, value):
        """Override __setattr__ to ensure dictionary consistency.
        
        If a new sparse_dictionary is assigned (e.g., by CFKD), we ensure
        it inherits the correctly resolved weights_path from our config
        if one was already established.
        """
        if name == "sparse_dictionary" and value is not None:
            # If we already have a weights_path resolved in our config,
            # ensure the new dictionary uses it.
            if (
                hasattr(self, "config") 
                and self.config.sparse_dictionary is not None
                and self.config.sparse_dictionary.weights_path
            ):
                if getattr(value.config, 'weights_path', None) is None:
                    value.config.weights_path = self.config.sparse_dictionary.weights_path
                    # If the file exists, load it immediately to avoid re-fitting or using internet mean
                    if os.path.exists(value.config.weights_path):
                        if getattr(value, 'image_mean', None) is None:
                            value.load_from_disk(value.config.weights_path)
            
            # Also ensure weight identity for the new dictionary
            if hasattr(self, 'encoder') and hasattr(value, 'set_clip_model'):
                value.set_clip_model(self.encoder.clip_model)

        super().__setattr__(name, value)

    # -------------------------------------------------------------------
    # Encode / Decode (DiDAE Algorithm 1)
    # -------------------------------------------------------------------

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        """Encode an image into semantic code + stochastic noise path.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            z_sem: (B, 768) CLIP ViT-L/14 image embedding.
            stochastic_code: tuple (wT, zs, wts) from DDPM forward process.
        """
        batch_size = x.shape[0]

        # 1. Semantic encoding via CLIP ViT-L/14
        z_sem = self.encoder(x.to(self.device))

        # 2. Stochastic encoding via DDPM forward process
        x0 = torchvision.transforms.Resize(
            [512, 512], antialias=True
        )(x.clone().to(self.device))

        # VAE encode
        w0 = (self.pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # Balanced Conditioning: Inject z_sem into a 77-token null sequence.
        uncond_embedding = encode_text(self.pipe, [""] * batch_size)
        z_sem_cond = uncond_embedding.clone()
        z_sem_cond[:, 1, :] = z_sem

        # DDPM forward process with empty prompt (faithful encoding)
        # Pass z_sem as encoder_hidden_states for consistency
        wT, zs, wts = inversion_forward_process(
            self.pipe,
            w0,
            etas=self.config.eta,
            prompt=batch_size * [""],
            cfg_scale=self.config.cfg_scale_src,
            prog_bar=False,
            num_inference_steps=self.config.num_diffusion_steps,
            encoder_hidden_states=z_sem_cond,
        )

        return z_sem, (wT, zs, wts)

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        """Decode from semantic code + stochastic noise path.

        Args:
            z: tuple of (z_sem, (wT, zs, wts))
               z_sem: (B, 768) CLIP embedding to condition on.
               wT, zs, wts: noise path from DDPM forward process.

        Returns:
            x_decoded: (B, 3, H, W) reconstructed/edited images.
        """
        z_sem, (wT, zs, wts) = z
        batch_size = z_sem.shape[0]

        # Balanced Conditioning: Inject z_sem into a 77-token null sequence.
        # This prevents attention saturation and stabilizes CFG at high scales.
        uncond_embedding = encode_text(self.pipe, [""] * batch_size)
        z_sem_cond = uncond_embedding.clone()
        z_sem_cond[:, 1, :] = z_sem

        # Setup attention controller
        controller = AttentionStore()
        register_attention_control(self.pipe, controller)

        # Get the starting point for reverse process
        xT = wts[self.config.num_diffusion_steps - self.config.skip]

        # DDPM reverse process conditioned on z_sem
        w0_dec, _ = inversion_reverse_process(
            self.pipe,
            xT=xT,
            etas=self.config.eta,
            prompts=batch_size * [""],
            cfg_scales=[self.config.cfg_scale_src],
            prog_bar=False,
            zs=zs[: (self.config.num_diffusion_steps - self.config.skip)],
            controller=controller,
            encoder_hidden_states=z_sem_cond,
        )

        # VAE decode
        x_decoded = self.pipe.vae.decode(1 / 0.18215 * w0_dec).sample

        if x_decoded.dim() < 4:
            x_decoded = x_decoded.unsqueeze(0)

        return x_decoded.detach()

    def decode_with_modified_embedding(self, z_sem_modified, stochastic_code, original_shape):
        """Decode using a modified semantic embedding for counterfactual generation.

        Uses the modified CLIP embedding as text conditioning for the DDPM
        reverse process. The stochastic noise path preserves structure while
        the modified embedding changes semantics.

        Args:
            z_sem_modified: (B, 768) modified CLIP embedding.
            stochastic_code: tuple (wT, zs, wts) from encode.
            original_shape: target spatial dimensions for output.

        Returns:
            x_counterfactual: (B, 3, H, W) counterfactual images.
        """
        wT, zs, wts = stochastic_code
        batch_size = z_sem_modified.shape[0]

        # Balanced Conditioning for modified embedding
        uncond_embedding = encode_text(self.pipe, [""] * batch_size)
        encoder_hidden_states = uncond_embedding.clone()
        encoder_hidden_states[:, 1, :] = z_sem_modified

        # Setup attention controller
        controller = AttentionStore()
        register_attention_control(self.pipe, controller)

        # Get starting point
        xT = wts[self.config.num_diffusion_steps - self.config.skip]

        # Run reverse process with the modified conditioning
        w0_dec, _ = inversion_reverse_process(
            self.pipe,
            xT=xT,
            etas=self.config.eta,
            prompts=batch_size * [""],
            cfg_scales=[self.config.cfg_scale_tar],
            prog_bar=False,
            zs=zs[: (self.config.num_diffusion_steps - self.config.skip)],
            controller=controller,
            encoder_hidden_states=encoder_hidden_states,
        )

        # VAE decode
        x_decoded = self.pipe.vae.decode(1 / 0.18215 * w0_dec).sample

        if x_decoded.dim() < 4:
            x_decoded = x_decoded.unsqueeze(0)

        # Resize back to original spatial dimensions
        x_counterfactual = torchvision.transforms.Resize(
            original_shape[2:], antialias=True
        )(x_decoded.detach().cpu())

        return x_counterfactual

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def sample_x(self, batch_size=1):
        """Generate random samples from the SD pipeline."""
        images = self.pipe(batch_size * [""]).images
        images_torch = torch.stack(
            [torchvision.transforms.ToTensor()(img) for img in images]
        )
        return images_torch

    def sample_z(self, batch_size=1):
        return torch.randn(batch_size, self.config.encoder_dimensions)

    def log_prob_z(self, z):
        raise NotImplementedError("Log probability not available for SD autoencoder.")

    # -------------------------------------------------------------------
    # Sparse Dictionary (SpLICE)
    # -------------------------------------------------------------------

    def load_sparse_dictionary(self):
        """Load or initialize the SpLICE sparse dictionary."""
        # Ensure dictionary has a base_path if not provided by the user
        # We store it relative to the generator's base_path for organization
        if (
            self.config.sparse_dictionary is not None
            and hasattr(self.config.sparse_dictionary, "base_path")
            and self.config.sparse_dictionary.base_path is None
            and self.config.base_path
        ):
            self.config.sparse_dictionary.base_path = os.path.join(
                self.config.base_path, "sparse_dictionaries", self.config.sparse_dictionary.sparse_dictionaries_type
            )
            # weights_path is usually base_path + weights.ending
            ending = getattr(self.config.sparse_dictionary, 'ending', '.pt')
            self.config.sparse_dictionary.weights_path = os.path.join(
                self.config.sparse_dictionary.base_path, "weights" + ending
            )

        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)

    def fit_sparse_dictionary(self):
        """Fit the SpLICE dictionary by computing dataset-specific image mean."""
        self.load_sparse_dictionary()
        self.sparse_dictionary.fit_from_dataloaders(
            [torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=64)],
            self.encoder,
        )
        # Save results to disk
        if self.config.sparse_dictionary.base_path:
            Path(self.config.sparse_dictionary.base_path).mkdir(parents=True, exist_ok=True)
            self.sparse_dictionary.save_on_disk(self.config.sparse_dictionary.weights_path)
            save_yaml_config(
                self.config.sparse_dictionary,
                os.path.join(self.config.sparse_dictionary.base_path, "config.yaml"),
            )

    # -------------------------------------------------------------------
    # Edit (DiDAE Algorithm 2)
    # -------------------------------------------------------------------

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: dict,
        predictor_datasets: list,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ) -> Tuple[list, list, list, list, list, list]:
        """Generate counterfactuals using DiDAE Algorithms 1 & 2.

        1. Encode input → z_sem + stochastic noise path
        2. Reflect z_sem along sparse dictionary components
        3. Decode reflected z_sem with original noise path → counterfactual

        This follows the same flow as DiffusionAutoencoder.edit() but uses
        SD + CLIP instead of a custom trained diffusion autoencoder.
        """
        param_list = [p for p in predictor.parameters()]
        device = param_list[0].device

        # Compute initial predictions
        pred_original = F.softmax(
            predictor(x_in.to(self.device)), dim=-1
        ).detach().cpu()
        target_confidences = [
            pred_original[i][target_classes[i]] for i in range(len(target_classes))
        ]
        target_confidence_goal = 1 - torch.tensor(target_confidences)

        # Get validation dataset for outlier scoring
        from peal.data.dataloaders import WeightedDataloaderList
        if isinstance(predictor_datasets[1], WeightedDataloaderList):
            validation_dataset = predictor_datasets[1].dataloaders[0].dataset
        else:
            validation_dataset = predictor_datasets[1]

        # Encode
        z_sem, stochastic_code = self.encode(x_in.to(self.device))

        # Get editing direction from gradient predictor
        # Uses the same approach as DiffusionAutoencoder: distill classifier
        # into a linear probe on top of the encoder, then use its weights
        if not hasattr(explainer_config, 'distilled_predictor') or explainer_config.distilled_predictor is None:
            # Direct approach: use the last linear layer weights
            w = list(predictor.children())[-1].weight[0]
        else:
            from peal.adaptors.counterfactual_knowledge_distillation import distill_predictor
            distilled_path = os.path.join(base_path, "explainer", "distilled_predictor", "model.cpl")
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    predictor_distillation=explainer_config.distilled_predictor,
                    base_path=os.path.join(base_path, "explainer"),
                    predictor=lambda x: predictor(x),
                    predictor_datasource=predictor_datasets,
                    predictor_distilled=nn.Sequential(
                        self.encoder,
                        nn.Linear(self.config.encoder_dimensions, 1, bias=False),
                    ),
                    only_last_layer=True,
                    continue_training=True,
                    task_config=TaskConfig(**explainer_config.distilled_predictor["task"]),
                )
            else:
                self.gradient_predictor = torch.load(distilled_path, map_location=self.device)

            w = list(self.gradient_predictor.children())[-1].weight[0]

        # Calculate counterfactual z_sem values (DiDAE Algorithm 2)
        z_sem_before, indices, distances = self._calculate_z_counterfactuals(
            z_sem, w, explainer_config, explainer_config.num_attempts
        )

        # Flatten for batch decoding
        z_sem2 = z_sem_before.reshape([-1, z_sem_before.shape[-1]])

        # Tile stochastic code for all candidates
        wT, zs, wts = stochastic_code
        wT_decoding = wT.unsqueeze(1).unsqueeze(1)
        wT_decoding = wT_decoding.tile(
            1, z_sem_before.shape[1],
            len(explainer_config.linesearch_factors), 1, 1, 1
        )
        wT_decoding = wT_decoding.reshape([-1] + list(wT.shape[1:]))

        # Decode all candidates
        x_counterfactuals_generator = self.decode_with_modified_embedding(
            z_sem2, (wT_decoding, zs, wts), x_in.shape
        )
        x_counterfactuals = x_counterfactuals_generator.detach()

        # Evaluate all candidates with the predictor
        preds = F.softmax(
            predictor(x_counterfactuals.to(device)), dim=-1
        ).detach().cpu()
        y_target_end_confidence = torch.zeros([preds.shape[0]])
        for i in range(preds.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i % target_classes.shape[0]]]

        # Reshape to (batch, num_attempts, linesearch_steps, ...)
        x_counterfactuals = torch.reshape(
            x_counterfactuals,
            list(z_sem_before.shape[:3]) + list(x_counterfactuals.shape[1:]),
        )
        y_target_end_confidence = torch.reshape(
            y_target_end_confidence, z_sem_before.shape[:3]
        )

        # Select best counterfactual per attempt
        x_counterfactuals_out_list = []
        y_target_end_confidence_list = []
        x_out_list = []
        indices_list = []

        for i in range(explainer_config.num_attempts):
            if x_counterfactuals.shape[2] >= 2:
                y_target_diff = y_target_end_confidence.clone()
                for b in range(y_target_end_confidence.shape[0]):
                    outlier_scores = validation_dataset.calculate_outlier_score(
                        x_counterfactuals[b, i]
                    )["relative"].cpu()
                    mask = torch.logical_and(outlier_scores < 1.3, outlier_scores > 0.1)
                    masked_difference = y_target_diff[b, i] * mask
                    y_target_diff[b, i] = torch.abs(
                        masked_difference - target_confidence_goal[b]
                    )
                j = torch.argmin(y_target_diff[:, i, :], dim=-1)
            else:
                j = torch.zeros([x_counterfactuals.shape[0]], dtype=torch.long)

            for k in range(j.shape[0]):
                x_counterfactuals_out_list.append(x_counterfactuals[k, i, j[k], :])
                y_target_end_confidence_list.append(
                    float(y_target_end_confidence[k, i, j[k]])
                )
                indices_list.append(indices[k])

            x_out_list.append(x_in)

        x_counterfactuals = torch.stack(x_counterfactuals_out_list, dim=0)
        x_out = torch.cat(x_out_list, dim=0)
        y_target_end_confidence = torch.tensor(y_target_end_confidence_list)
        indices = torch.cat(indices_list, dim=0)
        x_difference = x_out - x_counterfactuals.cpu()

        return (
            list(x_counterfactuals.cpu()),
            list(x_difference),
            list(y_target_end_confidence),
            list(x_in),
            [],
            list(indices.cpu()),
        )

    def _calculate_z_counterfactuals(
        self,
        z_sem: torch.Tensor,
        w: torch.Tensor,
        explainer_config=None,
        num_attempts=1,
    ):
        """Calculate counterfactual embeddings via reflection (DiDAE Algorithm 2).

        Mirrors DiffusionAutoencoder._calculate_z_counterfactuals exactly.
        Reflects z_sem along sparse dictionary component directions to
        flip the classifier decision.
        """
        if explainer_config is None or explainer_config.num_attempts == 1:
            # Simple reflection along classifier weight direction
            b = w.to(z_sem.dtype)
            a = z_sem
            dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
            dot_bb = torch.sum(b * b)
            proj = dot_ab / dot_bb * b
            reflected = a - 2 * proj
            
            return reflected, None, torch.norm(
                reflected - z_sem, p=2, dim=-1, keepdim=False
            )

        else:
            # Multi-component reflection with linesearch
            
            # Allow targeted edits via predefined component strings/indices
            component_indices = None
            if hasattr(self.sparse_dictionary.config, "component_strings") and self.sparse_dictionary.config.component_strings is not None:
                # Resolve base path and read vocabulary
                base_path = os.environ.get("PEAL_BASE", "")
                if not base_path:
                    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
                vocab_path = os.path.join(base_path, "peal", "dependencies", "SpLiCE", "data", "vocab", "laion.txt")

                if os.path.exists(vocab_path):
                    with open(vocab_path, "r") as f:
                        vocab_list = [line.strip().lower() for line in f.readlines()]
                    
                    component_indices = []
                    for search_term in self.sparse_dictionary.config.component_strings:
                        
                        # Find exact match first
                        try:
                            found_idx = vocab_list.index(search_term)
                            component_indices.append(found_idx)
                        except ValueError:
                            # Fallback to substring matching
                            found = False
                            for i, v in enumerate(vocab_list):
                                if search_term in v:
                                    component_indices.append(i)
                                    found = True
                                    break
                            if not found:
                                print(f"Warning: Could not find match for {search_term} in SpLICE vocab")
                else:
                    print(f"Warning: SpLICE vocab not found at {vocab_path}")

            if component_indices is None or len(component_indices) == 0:
                if hasattr(explainer_config, "component_indices") and explainer_config.component_indices is not None:
                    component_indices = explainer_config.component_indices
                else:
                    component_indices = range(num_attempts)
                
            W_all = self.sparse_dictionary.get_components()

            if component_indices is not None:
                W = W_all[:, component_indices].to(z_sem.device).to(z_sem.dtype)
                if explainer_config is not None and hasattr(explainer_config, "num_attempts"):
                    explainer_config.num_attempts = min(len(component_indices), W.shape[1])
            else:
                W = W_all.to(z_sem.device).to(z_sem.dtype)

            # Cross-projection targeting classifier flip
            w_cast = w.to(z_sem.dtype)
            dot_zw = torch.sum(z_sem * w_cast, dim=-1, keepdim=True)
            dot_uw = torch.matmul(w_cast, W)

            eps = 1e-6
            dot_uw_safe = dot_uw.clone()
            dot_uw_safe[torch.abs(dot_uw_safe) < eps] = eps

            proj_factors = dot_zw / dot_uw_safe.unsqueeze(0)
            projections = proj_factors.unsqueeze(-1) * W.permute(1, 0).unsqueeze(0)

            # Linesearch
            line_search_factors = torch.tensor(
                explainer_config.linesearch_factors
            ).to(z_sem.device)
            z_base = z_sem.unsqueeze(1).unsqueeze(1)
            proj_expanded = projections.unsqueeze(2)
            factors_expanded = line_search_factors.view(1, 1, -1, 1)

            z_reflected = z_base - factors_expanded * proj_expanded
            
            # Distances and sorting
            distances = torch.norm(z_base - z_reflected, p=2, dim=-1)
            sorted_indices = torch.argsort(distances, dim=1)
            component_indices = torch.arange(sorted_indices.shape[1]).unsqueeze(0).tile(
                [sorted_indices.shape[0], 1]
            )

            return z_reflected, component_indices, distances

    # -------------------------------------------------------------------
    # Component Explanation (mirroring DiffusionAutoencoder)
    # -------------------------------------------------------------------

    def explain_all_components(self, sparse_dictionary=None):
        """Visualize all sparse dictionary components."""
        if self.sparse_dictionary is None or sparse_dictionary is not None:
            if isinstance(sparse_dictionary, SparseDictionary):
                self.sparse_dictionary = sparse_dictionary
                self.config.sparse_dictionary = copy.deepcopy(sparse_dictionary.config)
            else:
                if isinstance(sparse_dictionary, SparseDictionaryConfig):
                    self.config.sparse_dictionary = sparse_dictionary
                self.config.sparse_dictionary.act_size = self.config.encoder_dimensions
                self.load_sparse_dictionary()
                if self.sparse_dictionary is None:
                    self.fit_sparse_dictionary()

        explanation_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
        )
        Path(explanation_path).mkdir(parents=True, exist_ok=True)

        # Cache paths
        y_list_path = os.path.join(explanation_path, "y_list.pt")
        c_list_path = os.path.join(explanation_path, "c_list.pt")
        z_list_path = os.path.join(explanation_path, "z_list.pt")

        # Compute or load component correlations
        if os.path.exists(y_list_path) and os.path.exists(c_list_path) and os.path.exists(z_list_path):
            print(f"Loading cached component correlations from {explanation_path}...")
            y_list = torch.load(y_list_path)
            c_list = torch.load(c_list_path)
            z_list = torch.load(z_list_path)
        else:
            print("Calculating component correlations (this may take a while)...")
            y_list, c_list, z_list = [], [], []
            task_config_buffer = (
                self.generator_datasets[1].task_config
                if hasattr(self.generator_datasets[1], "task_config")
                else None
            )
            self.generator_datasets[1].task_config = None

            for idx, batch in enumerate(
                torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10)
            ):
                print(f"{10 * idx}/{len(self.generator_datasets[1])}")
                x, y = batch
                z, _ = self.encode(x.to(self.device))
                c = z @ self.sparse_dictionary.get_components().to(self.device).to(z.dtype)
                y_list.append(y)
                c_list.append(c.detach().cpu())
                z_list.append(z.detach().cpu())

            self.generator_datasets[1].task_config = task_config_buffer
            
            print(f"Saving computed correlations to {explanation_path}...")
            torch.save(y_list, y_list_path)
            torch.save(c_list, c_list_path)
            torch.save(z_list, z_list_path)

        # Visualize individual components
        # Default to all components if n_components is not specified or <= 0
        n_comps = self.config.sparse_dictionary.n_components
        if n_comps is None or n_comps <= 0:
            n_comps = self.sparse_dictionary.get_components().shape[1]
            
        for component_idx in range(
            min(
                n_comps,
                self.sparse_dictionary.get_components().shape[1],
            )
        ):
            self.explain_sparse_component(
                torch.utils.data.DataLoader(
                    self.generator_datasets[1],
                    batch_size=self.config.visualizations_per_component or 10,
                ),
                component_idx,
            )

    def explain_sparse_component(self, dataloader, component_idx):
        """Visualize a single sparse dictionary component."""
        x_factual_list = []
        x_counterfactual_list = []
        start_idx = 0
        current_base_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
            str(component_idx),
        )
        Path(current_base_path).mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(dataloader):
            if (
                self.config.visualizations_per_component is not None
                and start_idx >= self.config.visualizations_per_component
            ):
                break

            x_factual_list.extend(list(batch[0]))
            x_factual = batch[0].to(self.device)
            x_counterfactual, (dot_before, dot_after) = (
                self.explain_sparse_component_batch(x_factual, component_idx)
            )
            x_counterfactual_list.extend(list(x_counterfactual.cpu()))
            start_idx += len(x_factual)

        return x_factual_list, x_counterfactual_list

    def explain_sparse_component_batch(self, x_generator, component_idx):
        """Generate counterfactuals for a single sparse component."""
        z_sem, stochastic_code = self.encode(x_generator.to(self.device))

        w_raw = self.sparse_dictionary.get_components()[:, component_idx].to(self.device)
        w = w_raw / torch.norm(w_raw, p=2)

        mu = (
            self.sparse_dictionary.mu.to(self.device)
            if hasattr(self.sparse_dictionary, 'mu') and self.sparse_dictionary.mu is not None
            else torch.zeros_like(z_sem[0])
        )

        proj_factors = (z_sem - mu) @ w
        z_sem_after = z_sem - 2 * proj_factors.unsqueeze(1) * w
        proj_factors_after = (z_sem_after - mu) @ w

        x_counterfactuals = self.decode_with_modified_embedding(
            z_sem_after, stochastic_code, x_generator.shape
        )

        return x_counterfactuals.cpu(), (
            proj_factors.cpu(),
            proj_factors_after.cpu(),
        )
