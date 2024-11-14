import copy
import os
import pathlib
from collections.abc import Iterable
from typing import Union
import numpy as np
import torchvision.models
from PIL import Image

import torch
from tqdm import tqdm
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
from zennit.image import imsave
from zennit.torchvision import ResNetCanonizer

from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.architectures.predictors import TaskConfig, TorchvisionModel
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.data.datasets import DataConfig
from peal.dependencies.ace.core.metrics import accuracy
from peal.explainers.lrp_explainer import LRPExplainer
from peal.teachers.virelay_teacher import VirelayTeacher
from peal.training.trainers import TrainingConfig

grad_in_fc = None
grad_out_avgpool = None

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
        self.explainer = LRPExplainer(downstream_model=self.model, num_classes=2,
                                      explainer_config_return_namespace=False)
        self.teacher = VirelayTeacher(2)

        self.activation = torch.tensor([0])
        self.hook_handle = self.register_feature_extractor()

    def register_feature_extractor(self):

        model = self.model
        if isinstance(model, TorchvisionModel):
            assert len(list(model.children())) == 1
            model = next(model.children())

        assert isinstance(model, torchvision.models.ResNet)

        def forward_hook(module, x_in, x_out):
            self.activation = x_out
            return x_out.clone()

        return model.avgpool.register_forward_hook(forward_hook)

    def explain_prediction(self, x, y, filenames, collages_dir):
        canonizer = ResNetCanonizer()
        # low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))
        # composite = EpsilonGammaBox(low=0, high=1, canonizers=[canonizer])
        composite = EpsilonPlusFlat(canonizers=[canonizer])

        with Gradient(model=self.model, composite=composite) as attributor:
            # compute the model output and attribution
            output, attribution = attributor(x, torch.eye(2)[y].to(self.device))

        print("accuracy: ", torch.mean((output.argmax(1) == y).float()).item())

        relevance = attribution.sum(1)

        for i in range(len(relevance)):
            ori_file_path = os.path.join(collages_dir, filenames[i])
            rel_file_path = os.path.join(collages_dir, "_heatmap.".join(filenames[i].split(".")))
            imsave(ori_file_path, x[i])
            imsave(rel_file_path, relevance[i], symmetric=True, cmap='bwr')

    def generate_lrp_heatmaps(self, dataloader, max_samples=10):

        collages_dir = os.path.join(self.adaptor_config.base_dir, "collages_unpoisoned_gradient")
        pathlib.Path(collages_dir).mkdir(parents=True, exist_ok=True)

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
                y = batch["y"][:, 0].to(torch.int)
                # filenames.extend(batch["url"])
                filenames = batch["url"]

                self.explain_prediction(X, y, batch["url"], collages_dir)

                samples += X.shape[0]
                if samples > max_samples:
                    break
                continue

                heatmap_batch, overlay_batch, prediction_batch = self.explainer.explain_batch(X, y)

                for idx in range(len(heatmap_batch)):
                    collage = np.array(255 * torch.cat([X[idx], heatmap_batch[idx], overlay_batch[idx]], 2).numpy(),
                                       dtype=np.uint8).transpose(1, 2, 0)
                    collage_img = Image.fromarray(collage)
                    collage_path = os.path.join(collages_dir, filenames[idx])
                    collage_img.save(collage_path)

                predictions.append(prediction_batch)
                images.append(X)
                overlays.append(overlay_batch)
                heatmaps.append(heatmap_batch)
                gt_classes.append(y)
                # TODO implement hints
                hints.append(heatmap_batch)
                pbar.set_description('Generating LRP maps: ' + str(samples) + ' / ' + str(max_samples))

        exit()
        images = list(torch.cat(images))
        heatmaps = list(torch.cat(heatmaps))
        overlays = list(torch.cat(overlays))
        gt_classes = list(torch.cat(gt_classes))
        predictions = list(torch.cat(predictions))
        hints = list(torch.cat(hints))
        # TODO implement attributions in intermediate layers
        attributions = heatmaps

        collage_paths = []
        for idx in range(len(images)):
            collage = np.array(255 * torch.cat([images[idx], heatmaps[idx], overlays[idx]], 2).numpy(),
                               dtype=np.uint8).transpose(1, 2, 0)
            collage_img = Image.fromarray(collage)
            collage_path = os.path.join(collages_dir, filenames[idx])
            collage_img.save(collage_path)
            collage_paths.append(collage_path)

        return collage_paths, heatmaps, gt_classes, hints, attributions, images, predictions

    def calculate_annotations_and_activations(self, dataloader, feature_extractor):
        # preprocessing: collect relevance scores (LRP) and activations
        max_samples = 200  # len(dataloader.dataset)
        collage_paths, heatmaps, gt_classes, hints, attributions, images, predictions \
            = self.generate_lrp_heatmaps(dataloader, max_samples=max_samples)

        feedback = self.teacher.get_feedback(
            images=images,
            heatmaps=heatmaps,
            collage_paths=collage_paths,
            gt_classes=gt_classes,
            attributions=attributions,
            base_dir=self.adaptor_config.base_dir,
            source_classes=predictions
        )

        print("feedback:\n")
        print(feedback)

        activations = []
        annotations = []
        targets = []
        with tqdm(range(len(feedback))) as pbar:
            for it in pbar:
                X = images[it].unsqueeze(0).to(self.device)
                features = feature_extractor(X).detach()[0]
                activations.append(features)
                annotations.append(-1 if feedback[it] == 'true' else 1)

                pbar.set_description(f'Extracting features: {it}/{len(feedback)}')

        activations = torch.stack(activations, 0).to(self.device).detach().cpu().numpy()
        annotations = torch.tensor(annotations).to(self.device).detach().cpu().numpy()

        return activations, annotations, targets

    def annotations_and_activations_simplified(self, dataloader):
        activations = []
        targets = []
        annotations = []

        with tqdm(enumerate(dataloader)) as pbar:
            for it, batch in pbar:
                pbar.set_description(
                    f'calculating activations and annotations: {it * dataloader.batch_size}/{len(dataloader.dataset)}')
                X = batch["x"].to(self.device)
                self.model(X)
                targets.append(batch["y"][:, 0].detach())
                annotations.append(batch["y"][:, 1].detach())
                activations.append(torch.flatten(self.activation.detach(), 1))
                # if it > 25:
                #     break

        activations = torch.cat(activations, dim=0)
        targets = torch.cat(targets, 0).to(self.device)
        assert torch.all(targets == 0), "Only class 0 supported"
        annotations = torch.cat(annotations, dim=0)
        annotations[annotations == 0] = -1

        return activations, annotations, targets

    def calculate_cav(self, actvs, t):
        print("actvs: ", actvs.shape, actvs.mean(dim=0).shape)
        print("t: ", t.shape)
        t_centered = t - t.mean()
        actvs_centered = actvs - actvs.mean(dim=0)[None]
        covar = (actvs_centered * t_centered[:, None]).sum(dim=0) / (t.shape[0] - 1)
        vary = torch.sum(t_centered ** 2, dim=0) / (t.shape[0] - 1)
        w = (covar / vary)[None]

        cav = w / torch.sqrt((w ** 2).sum())
        return cav

    def finetune(self, dataloader, cav):
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.model.fc.parameters():
            param.requires_grad = True

        classifier_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(classifier_params) == 2, "something went wrong when freezing parameters for fine tuning"
        optimizer = torch.optim.SGD(classifier_params, lr=0.00001)

        for epoch in range(10):
            with tqdm(enumerate(dataloader)) as pbar:
                for it, batch in pbar:
                    optimizer.zero_grad()
                    x = batch["x"].to(self.device)
                    x.requires_grad = True
                    y = batch["y"][:,0].to(self.device)

                    prediction = self.model(x)

                    #y_hat = (prediction * torch.sign(0.5 - torch.rand_like(prediction))).sum(1)
                    y_hat = prediction.sum(1)
                    grad = torch.autograd.grad(outputs=y_hat,
                                               inputs=self.activation,
                                               create_graph=True,
                                               retain_graph=True,
                                               grad_outputs=torch.ones_like(y_hat))[0]
                    grad = grad.flatten(start_dim=1)

                    rr_loss = ((grad * cav).sum(1) ** 2).mean(0)
                    ce_loss = torch.nn.functional.cross_entropy(prediction, y.to(torch.long))
                    accuracy = torch.mean((prediction.argmax(1) == y).float()).item()

                    pbar.set_description(f'epoch {epoch}: accuracy={accuracy}, rr_loss={rr_loss}, ce_loss={ce_loss}')
                    loss = self.adaptor_config.lamb * rr_loss + ce_loss

                    loss.backward()
                    optimizer.step()

    def run(self, *args, **kwargs):
        self.model.eval()

        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
        train_dataloader.dataset.enable_idx()
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True
        train_dataloader.dataset.enable_class_restriction(0)

        # targets = []
        # predictions = []
        # for batch in train_dataloader:
        #     x = batch["x"].to(self.device)
        #     y = batch["y"][:,0].to(self.device)
        #     targets.append(y)
        #
        #     prediction = self.model(x)
        #     predictions.append(prediction.argmax(1))
        #
        # targets = torch.cat(targets, 0)
        # predictions = torch.cat(predictions, 0)
        # print("accuracy: ", torch.mean((predictions == targets).float()).item())
        # exit()
        test_dataloader.dataset.enable_idx()
        test_dataloader.dataset.return_dict = True
        test_dataloader.dataset.url_enabled = True

        # train_dataloader.dataset.visualize_decision_boundary(self.model, 32, self.device, os.path.join(self.adaptor_config.base_dir, "decision_boundary_corrected.png"))


        activations, annotations, targets = self.calculate_annotations_and_activations(test_dataloader, self.model)
        activations, annotations, targets = self.annotations_and_activations_simplified(train_dataloader)
        cav = self.calculate_cav(activations, annotations)

        train_dataloader.dataset.disable_class_restriction()
        self.finetune(train_dataloader, cav)
        self.hook_handle.remove()

        corrected_model_path = os.path.join(self.adaptor_config.base_dir, "corrected_model.cpl")
        print("saving corrected model to: " + corrected_model_path)
        torch.save(self.model.to("cpu"), corrected_model_path)


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
