import os
import types
import shutil
import copy
from pathlib import Path

import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.data.dataset_factory import get_datasets
from peal.data.datasets import Image2MixedDataset
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config, embed_numberstring
from peal.dependencies.time.generate_ce import (
    generate_time_counterfactuals,
)
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import training


class StableDiffusion(EditCapableGenerator):
    def __init__(self, config, classifier_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.classifier_dataset = copy.deepcopy(classifier_dataset)
        self.generator_dataset = None

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")

    def sample_x(self, batch_size=1):
        return self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.classifier_dataset.config.input_size
        )

    def initialize(self, classifier, base_path):
        class_predictions_path = os.path.join(base_path, "explainer", "predictions.csv")
        generator_dataset = get_datasets(self.config.data)[0]
        generator_dataset.task_config = self.classifier_dataset.task_config
        import pdb; pdb.set_trace()
        Path(os.path.join(base_path, "explainer")).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            generator_dataset.enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=generator_dataset,
                classifier=classifier,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
            )
            get_predictions(prediction_args)

        generator_dataset_config = copy.deepcopy(self.config.data)
        generator_dataset_config.split = [1.0, 1.0]
        self.generator_dataset = get_datasets(
            config=generator_dataset_config, data_dir=class_predictions_path
        )[0]
        context_embedding_path = os.path.join(
            base_path, "explainer", "context_embedding"
        )
        if not os.path.exists(context_embedding_path):
            train_context_embedding_args = types.SimpleNamespace(
                sd_model="CompVis/stable-diffusion-v1-4",
                embedding_files=[],
                output_path=context_embedding_path,
                dataset=self.generator_dataset,
                label_query=0,
                training_label=-1,
                custom_tokens=["|<C*1>|", "|<C*2>|", "|<C*3>|"],
                custom_tokens_init=["centered", "realistic", "celebrity"],
                phase="context",
                mini_batch_size=1,
                enable_xformers_memory_efficient_attention=False,
                gpu="0",
                use_fp16=False,
                lr=1e-4,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-9,
                weight_decay=1e-4,
                iterations=3000,
                batch_size=64,
                image_size=128,
                partition="train",
                seed=99999999,
            )
            training(train_context_embedding_args)

        # TODO how to extend this for multiclass??
        for class_idx in range(2):
            class_token_path = os.path.join(base_path, "explainer", "class_token" + str(class_idx))
            if not os.path.exists(class_token_path):
                class_related_bias_embedding_args = types.SimpleNamespace(
                    sd_model="CompVis/stable-diffusion-v1-4",
                    embedding_files=[context_embedding_path],
                    output_path=class_token_path,
                    dataset=self.generator_dataset,
                    label_query=0,
                    training_label=class_idx,
                    custom_tokens=[
                        "|<A*" + str(class_idx) + "1>|",
                        "|<A*" + str(class_idx) + "2>|",
                        "|<A*" + str(class_idx) + "3>|",
                    ],
                    custom_tokens_init=["centered", "realistic", "celebrity"],
                    phase="class",
                    mini_batch_size=1,
                    base_prompt="A |<C*1>| |<C*2>| |<C*3>| photo",
                    enable_xformers_memory_efficient_attention=False,
                    gpu="0",
                    use_fp16=False,
                    lr=1e-4,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    adam_epsilon=1e-9,
                    weight_decay=1e-4,
                    iterations=3000,
                    batch_size=64,
                    image_size=128,
                    partition="train",
                    seed=99999999,
                )
                training(class_related_bias_embedding_args)

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        explainer_config,
        classifier_dataset,
        pbar=None,
        mode="",
        base_path="",
    ):
        if self.generator_dataset is None:
            self.initialize(classifier, base_path)

        dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        ce_generation_args = types.SimpleNamespace(
            embedding_files=[
                os.path.join(base_path, "explainer", "context_embedding"),
                os.path.join(base_path, "explainer", "class_token0"),
                os.path.join(base_path, "explainer", "class_token1"),
            ],
            use_negative_guidance_denoise=True,
            use_negative_guidance_inverse=True,
            guidance_scale_denoising=[4,4,4,4],
            guidance_scale_invertion=[4,4,4,4],
            num_inference_steps=[15,20,25,35],
            output_path=os.path.join(base_path, "explainer", "outputs"),
            exp_name="time",
            label_target=-1,
            label_query=31,
            neg_custom_token="|<A*01>| |<A*02>| |<A*03>|",
            pos_custom_token="|<A*11>| |<A*12>| |<A*13>|",
            base_prompt="A |<C*1>| |<C*2>| |<C*3>| photo",
            chunks=1,
            chunk=0,
            enable_xformers_memory_efficient_attention=False,
            partition="val",
            dataset=dataset,
            classifier=classifier,
            use_fp16=False,
            sd_model="CompVis/stable-diffusion-v1-4",
            sd_image_size=128,
            custom_obj_token="|<C*>|",
            p=0.93,
            l2=0.0,
            batch_size=1,
            classifier_image_size=128,
            recover=False,
            num_samples=9999999999999999,
            merge_chunks=False,
            postprocess=lambda x, size: self.generator_dataset.project_to_pytorch_default(x),
            generic_custom_tokens=["|<C*1>|", "|<C*2>|", "|<C*3>|"],
        )
        x_counterfactuals = generate_time_counterfactuals(ce_generation_args)

        """dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        args = copy.deepcopy(self.config)
        args.dataset = dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = time_main(args=args)"""

        x_counterfactuals = torch.cat(x_counterfactuals, dim=0)
        device = [p for p in classifier.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(
            classifier(x_counterfactuals.to(device)).detach().cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_in),
        )
