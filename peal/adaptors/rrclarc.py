import copy
import os
import pathlib
from collections.abc import Iterable
from typing import Union
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.architectures.predictors import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.data.datasets import DataConfig
from peal.explainers.lrp_explainer import LRPExplainer
from peal.global_utils import embed_numberstring
from peal.teachers.virelay_teacher import VirelayTeacher
from peal.training.trainers import TrainingConfig


class RRClArCConfig(AdaptorConfig):
    """
    The config template for a running the RR-ClArC adaptor.
    """

    """
    The config template for an adaptor.
    """
    category: str = "adaptor"

    adaptor_type: str = "RRClArC"

    model_path: str

    compute: str = "l2_mean"

    criterion: str = "allrand"

    eval_acc_every_epoch: bool = False

    img_size: int = 224

    lamb: float = 1.0

    layer_name: str = "last_conv"

    loss: str = "cross_entropy"

    classes: Iterable[int] = [0]

    data: DataConfig

    training: TrainingConfig

    task: TaskConfig

    base_dir: str

    task = None

    __name__: str = "peal.AdaptorConfig"

    def __init__(self,
                 model_path: str = None,
                 compute: str = None,
                 criterion: str = None,
                 eval_acc_every_epoch: bool = None,
                 img_size: int = None,
                 lamb: float = None,
                 layer_name: str = None,
                 loss: str = None,
                 classes: Iterable[int] = None,
                 data: Union[dict, DataConfig] = None,
                 training: Union[dict, TrainingConfig] = None,
                 task: Union[dict, TaskConfig] = None,
                 base_dir: str = None,
                 **kwargs):

        self.model_path = model_path
        self.base_dir = base_dir

        if isinstance(data, DataConfig):
            self.data = data
        else:
            self.data = DataConfig(**data)

        if isinstance(training, TrainingConfig):
            self.training = training
        else:
            self.training = TrainingConfig(**training)

        if isinstance(task, TaskConfig):
            self.task = task
        elif not task is None:
            self.task = TaskConfig(**task)

        if not compute is None:
            self.compute = compute
        if not criterion is None:
            self.criterion = criterion
        if not eval_acc_every_epoch is None:
            self.eval_acc_every_epoch = eval_acc_every_epoch
        if not img_size is None:
            self.img_size = img_size
        if not lamb is None:
            self.lamb = lamb
        if not layer_name is None:
            self.layer_name = layer_name
        if not loss is None:
            self.loss = loss
        if not classes is None:
            self.classes = classes

        self.kwargs = kwargs


class RRClArC(Adaptor):
    def __init__(self, adaptor_config: RRClArCConfig):
        self.adaptor_config = adaptor_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.orginal_model = torch.load(self.adaptor_config.model_path, map_location=self.device)
        self.model = copy.deepcopy(self.orginal_model)
        self.explainer = LRPExplainer(downstream_model=self.model, num_classes=2, explainer_config_return_namespace=False)
        self.teacher = VirelayTeacher(2)

    def extract_all_children(self, model):
        '''
        Extracts all children of a model
        '''
        children = []
        for child in model.children():
            if isinstance(child, torch.nn.Sequential):
                children.extend(self.extract_all_children(child))

            else:
                children.append(child)

        return children

    def split_model_at_penultima(self, model):
        '''
        Splits a model at the penultima layer
        '''
        children_list = self.extract_all_children(model)
        feature_extractor = torch.nn.Sequential(*children_list[:-2])
        downstream_head = torch.nn.Sequential(*children_list[-2:])
        return feature_extractor, downstream_head

    def generate_lrp_heatmaps(self, dataloader, max_samples = 10):
        images = []
        heatmaps = []
        overlays = []
        gt_classes = []
        predictions = []
        hints = []
        samples = 0
        filenames = []
        max_samples = min(len(dataloader.dataset), max_samples)
        with tqdm(enumerate(dataloader)) as pbar:
            for it, batch in pbar:
                X = batch["x"]
                y = batch["y"]
                filenames.append(batch["url"])

                heatmap_batch, overlay_batch, prediction_batch = self.explainer.explain_batch(X, y)

                predictions.append(prediction_batch)
                images.append(X)
                overlays.append(overlay_batch)
                heatmaps.append(heatmap_batch)
                gt_classes.append(y)
                # TODO implement hints
                hints.append(heatmap_batch)
                samples += X.shape[0]
                pbar.set_description('Generating LRP maps: ' + str(samples) + ' / ' + str(max_samples))
                if samples > max_samples:
                    break

        images = list(torch.cat(images))
        heatmaps = list(torch.cat(heatmaps))
        overlays = list(torch.cat(overlays))
        gt_classes = list(torch.cat(gt_classes))
        predictions = list(torch.cat(predictions))
        hints = list(torch.cat(hints))
        # TODO implement attributions in intermediate layers
        attributions = heatmaps

        collages_dir = os.path.join(self.adaptor_config.base_dir, "collages")
        pathlib.Path(collages_dir).mkdir(parents=True, exist_ok=True)

        collage_paths = []
        for idx in range(len(images)):
            collage = np.array(255 * torch.cat([images[idx], heatmaps[idx], overlays[idx]], 2).numpy(), dtype=np.uint8).transpose(1,2,0)
            collage_img = Image.fromarray(collage)
            collage_path = os.path.join(collages_dir, embed_numberstring(str(idx)) + '.png')
            collage_img.save(collage_path)
            collage_paths.append(collage_path)

        return collage_paths, heatmaps, gt_classes, hints, attributions, images, predictions



    def run(self, *args, **kwargs):
        self.model.eval()

        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
        train_dataloader.dataset.enable_idx()
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True

        # preprocessing: collect relevance scores (LRP) and activations
        #self.generate_lrp_heatmaps(train_dataloader, max_samples=20)
        max_samples = 64 # len(train_dataloader.dataset)
        collage_paths, heatmaps, gt_classes, hints, attributions, images, predictions\
            = self.generate_lrp_heatmaps(train_dataloader, max_samples=max_samples)

        feedback = self.teacher.get_feedback(
            images = images,
            heatmaps = heatmaps,
            collage_paths = collage_paths,
            gt_classes = gt_classes,
            attributions=attributions,
            base_dir = self.adaptor_config.base_dir,
            source_classes=predictions
        )

        print("feedback:\n")
        print(feedback)

        feature_extractor, downstream_head = self.split_model_at_penultima(self.model)
        feature_extractor.eval()

        uncorrect_decision_strategy = []
        correct_decision_strategy = []
        with tqdm(range(len(feedback))) as pbar:
            for it in pbar:
                X = images[it].unsqueeze(0).to(self.device)
                features = feature_extractor(X).detach()[0]
                if feedback[it] == 'true':
                    correct_decision_strategy.append(features)
                else:
                    uncorrect_decision_strategy.append(features)

                pbar.set_description(f'Extracting features: {it}/{len(feedback)}')

        # for class_id in self.adaptor_config.classes:
        #     self.collect_relevances_and_activations(class_id)

# def collect_relevances_and_activations(self, class_id: int):
#     self.model.eval()
#
#     # TODO: only load data with corresponding class_id
#     train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
#     train_dataloader.dataset.enable_idx()
#     train_dataloader.dataset.return_dict = True
#     train_dataloader.dataset.url_enabled = True
#
#     assert isinstance(train_dataloader.dataset, SquareDataset), "Only SquareDataset supported"
#
#
#
#     # attribution = CondAttribution(self.model)
#     # canonizers = get_canonizer('resnet18')  # TODO: currently only supports resnet18 predictors
#     # composite = EpsilonPlusFlat(canonizers)
#     # cc = ChannelConcept()
#
#     layer_names = ["last_conv"]
#     conv_layers = ["last_conv"]
#     crvs = dict(zip(layer_names, [[] for _ in layer_names]))
#     relevances_all = dict(zip(layer_names, [[] for _ in layer_names]))
#     cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
#     cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
#     smpls = []
#     output = []
#
#     i=0
#     for batch_idx, batch in enumerate(tqdm(train_dataloader)):
#
#         print(batch["url"])
#         i += 1
#         if i > 100:
#             exit()
#         continue
#
#         # TODO: check if this is enough or if real indices of samples are needed
#         sample_idxs = batch["idx"]
#         x = batch["x"].to(self.device).requires_grad_()
#         out = self.model(x).detach()
#         condition = [{"y": c_id} for c_id in out.argmax(1)]
#         print(batch["y"])
#         ## print( batch["x"])
#         print(out)
#         print("\n~~~~~~~~~~~\n")
#         assert out.argmax(1).item() == batch["y"].item(), "Model prediction does not match ground truth label"
#
#         attr = attribution(x, condition, composite, record_layer=layer_names, init_rel=1)
#
#         non_zero = ((attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0) * (out.argmax(1) == class_id)).numpy()
#         samples_nz = sample_idxs[non_zero]
#         output.append(out[non_zero])
#
#         layer_names_ = [l for l in layer_names if l in attr.relevances.keys()]
#         conv_layers_ = [l for l in conv_layers if l in attr.relevances.keys()]
#
#         if samples_nz.size:
#             smpls += [s for s in samples_nz]
#             rels = [cc.attribute(attr.relevances[layer][non_zero], abs_norm=True) for layer in layer_names_]
#             acts_max = [
#                 attr.activations[layer][non_zero].flatten(start_dim=2).max(2)[0] if layer in conv_layers_ else
#                 attr.activations[layer][non_zero] for layer in layer_names_]
#             acts_mean = [attr.activations[layer][non_zero].mean((2, 3)) if layer in conv_layers_ else
#                          attr.activations[layer][non_zero] for layer in layer_names_]
#             for l, r, amax, amean in zip(layer_names_, rels, acts_max, acts_mean):
#                 crvs[l] += r.detach().cpu()
#                 cavs_max[l] += amax.detach().cpu()
#                 cavs_mean[l] += amean.detach().cpu()
#
#     path = self.adaptor_config.base_dir
#     os.makedirs(path, exist_ok=True)
#
#     print("saving as", f"{path}/class_{class_id}.pth")
#
#     str_class_id = 'all' if class_id is None else class_id
#     torch.save({"samples": smpls,
#                 "output": output,
#                 "crvs": crvs,
#                 "relevances_all": relevances_all,
#                 "cavs_max": cavs_max,
#                 "cavs_mean": cavs_mean},
#                f"{path}/class_{str_class_id}.pth")
#     for layer in layer_names:
#         torch.save({"samples": smpls,
#                     "output": output,
#                     "crvs": crvs[layer],
#                     "relevances_all": relevances_all[layer],
#                     "cavs_max": cavs_max[layer],
#                     "cavs_mean": cavs_mean[layer]},
#                    f"{path}/{layer}_class_{str_class_id}.pth")