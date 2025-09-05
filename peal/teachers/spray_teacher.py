import json
import multiprocessing
import os
import pathlib
import time
from typing import Union

import h5py
import torch
import numpy as np
import csv
import yaml
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans, AgglomerativeClustering, HDBSCAN, DBSCAN
from corelay.processor.embedding import EigenDecomposition, TSNEEmbedding, UMAPEmbedding
from corelay.processor.flow import Sequential, Parallel
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from virelay.model import Workspace
from virelay.server import Server
from zennit.attribution import Gradient, IntegratedGradients, SmoothGrad
from zennit.composites import EpsilonPlusFlat, EpsilonGammaBox
from zennit.image import imgify, imsave
from zennit.torchvision import ResNetCanonizer

from peal.adaptors.clarc import get_layer_name
from peal.architectures.interfaces import TaskConfig
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import save_yaml_config
from peal.teachers.interfaces import TeacherInterface


class SprayConfig(BaseModel):
    config_name: str = "SprayConfig"
    base_dir: str
    data: DataConfig
    dataset_name: str = None
    model: str
    port: int = 8080
    use_relative_concept_importance: bool = False
    sum_attribution_channels: bool = False
    normalize_attribtions: bool = True
    attribution_layer: int = -2
    classes_total: int
    num_clusters_max: int = 8
    num_eigval: int = 32
    tsne_perplexity: Union[float, list[float]] = 64.
    umap_neighbors: Union[int, list[int]] = 32
    task: TaskConfig
    skip_wrongly_classified_samples: bool = False
    max_samples: int = 10000
    split_dataset: bool = False
    conditioning_layer: int = None
    conditioning_channels: list[int] = None
    include_testsplit: bool = False


class LogitDifference(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        return y[0] - y[1]


class Spray(TeacherInterface):
    def __init__(self, config: Union[dict, SprayConfig], device: str = None):

        if not isinstance(config, SprayConfig):
            config = SprayConfig(**config)
        self.config = config
        if isinstance(self.config.tsne_perplexity, float):
            self.config.tsne_perplexity = [self.config.tsne_perplexity]
        if isinstance(self.config.umap_neighbors, int):
            self.config.umap_neighbors = [self.config.umap_neighbors]

        self.base_dir = config.base_dir
        pathlib.Path(self.base_dir).mkdir(exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print("running on device: ", self.device)

        self.model = torch.load(config.model, map_location=self.device)
        if hasattr(self.model, 'model'):
            self.model = self.model.model
        self.model.eval()

        self.attribution_layer = self.config.attribution_layer
        self.dataset_class = config.dataset_name or config.data.dataset_class

        self.analysis_dir = None
        self.attribution_db_path = None
        self.heatmaps_db_path = None
        self.concept_importance_db_path = None
        self.input_db_path = None
        self.analysis_db_path = None
        self.label_map_file_path = None
        self.virelay_project_path = None

        self.train_data, self.val_data, self.test_data = get_datasets(self.config.data, task_config=self.config.task, return_dict=True)
        self.train_data.url_enabled = True
        self.val_data.url_enabled = True
        self.test_data.url_enabled = True
        self.train_data.disable_class_restriction()
        self.val_data.disable_class_restriction()
        self.test_data.disable_class_restriction()
        self.train_data.enable_groups()
        self.val_data.enable_groups()
        self.test_data.enable_groups()

    def get_feedback(self, **args) -> dict[str, int]:

        layer_name = "input" if self.attribution_layer == 0 else get_layer_name(self.model, self.attribution_layer)
        self.analysis_dir = os.path.join(self.base_dir, f"attrbs-layer-{self.attribution_layer}_analysis")
        pathlib.Path(self.analysis_dir).mkdir(exist_ok=True)
        save_yaml_config(self.config, os.path.join(self.analysis_dir, "config.yaml"))

        self._compute_attributions(layer_name)
        self.analysis_db_path = self._spectral_clustering()
        self.label_map_file_path = self._create_label_map()
        self.virelay_project_path = self._create_project(self.config.data.input_size)
        confounder_ids, non_confounder_ids = self._run_virelay()

        with h5py.File(self.input_db_path, 'r') as input_db:
            filenames = input_db['filenames'].asstr()[:]

        group_label_map = {}
        for id in confounder_ids:
            group_label_map[filenames[id]] = 1
        for id in non_confounder_ids:
            group_label_map[filenames[id]] = 0

        true_labels = np.array(list(group_label_map.values()))
        print(f"found {len(true_labels[true_labels == 1])} confounders and {len(true_labels[true_labels == 0])} non-confounders")

        new_data_file = []
        with open(os.path.join(self.config.data.dataset_path, "data.csv"), "r") as f:
            data_file = csv.reader(f, delimiter=",")
            header = next(data_file)
            drop_last = header[len(header) - 1] == ""
            if drop_last:
                header.pop()
            header.append("SprayLabel")
            new_data_file.append(header)

            for line in data_file:
                if drop_last:
                    line.pop()
                line.append(group_label_map.get(line[0], -1))
                new_data_file.append(line)

        with open(os.path.join(self.analysis_dir, "data_spray_labels.csv"), "w", newline="") as f:
            csv.writer(f).writerows(new_data_file)

        if self.config.include_testsplit:
            dataloader = DataLoader(ConcatDataset([self.train_data, self.val_data, self.test_data]), batch_size=64, shuffle=False)
        else:
            dataloader = DataLoader(ConcatDataset([self.train_data, self.val_data]), batch_size=64, shuffle=False)

        label_acc = [[[], []], [[], []]]
        for batch in dataloader:
            true_labels = batch["has_confounder"]
            targets = batch["y"].squeeze().int()
            for i, filename in enumerate(batch['url']):
                if filename in group_label_map:
                    spray_label = group_label_map[filename]
                    label_acc[targets[i].item()][spray_label].append(true_labels[i].item() == spray_label)

        result_summary = ""
        for class_idx in range(len(label_acc)):
            for confounder_idx in range(len(label_acc[class_idx])):
                result_summary += f"class {class_idx} and {'confounder' if confounder_idx == 1 else 'non-confounder'}: n={len(label_acc[class_idx][confounder_idx])}, label_acc={np.mean(label_acc[class_idx][confounder_idx])}\n"
        with open(os.path.join(self.analysis_dir, "result_summary.txt"), "w") as f:
            f.write(result_summary)

        return group_label_map


    def _compute_attributions(self, attribution_layer: str):

        self.attribution_db_path = os.path.join(self.analysis_dir, "attribution.h5")
        self.heatmaps_db_path = os.path.join(self.analysis_dir, "heatmaps.h5")
        self.concept_importance_db_path = os.path.join(self.analysis_dir, "concept_importance.h5")
        self.input_db_path = os.path.join(self.analysis_dir, "input.h5")

        if pathlib.Path(self.input_db_path).is_file():
            return

        attribution_db_file = None
        input_db_file = None
        heatmaps_db_file = None
        concept_importance_db_file = None

        attributor = CondAttribution(self.model, no_param_grad=True)
        # composite = EpsilonPlusFlat([ResNetCanonizer()])
        composite = EpsilonGammaBox(
            low = -3.0, high = 3.0,
            epsilon=1e-5,
            gamma = .1,
            stabilizer = .15,
            canonizers=[ResNetCanonizer()]
        )
        heatmap_generator = SmoothGrad(self.model, composite)
        # heatmap_generator = Gradient(self.model, composite)
        cc = ChannelConcept()

        concept_layer = None
        if self.config.conditioning_layer is not None:
            concept_layer = get_layer_name(self.model, self.config.conditioning_layer)

        train_data = self.train_data
        if len(self.train_data) > self.config.max_samples:
            idxs = np.random.choice(np.arange(len(self.train_data)), size=self.config.max_samples, replace=False)
            train_data = Subset(self.train_data, idxs)

        if self.config.include_testsplit:
            len_data = len(train_data) + len(self.val_data) + len(self.test_data)
            datasplits = [train_data, self.val_data, self.test_data]
        else:
            len_data = len(train_data) + len(self.val_data)
            datasplits = [train_data, self.val_data]

        if self.config.split_dataset:
            data = [(i, DataLoader(split, batch_size=64, shuffle=False)) for i, split in enumerate(datasplits)]
        else:
            data = [(-1, DataLoader(ConcatDataset(datasplits), batch_size=64, shuffle=False))]

        number_samples_processed = 0
        for split, dataloader in data:

            with tqdm(dataloader) as pbar:
                pbar.set_description(f"computing relevances in dataset split {split} at layer {attribution_layer}...")
                for batch in pbar:
                    x = batch["x"].to(self.device).requires_grad_()
                    y = batch["y"].squeeze().to(torch.int64)
                    # out = self.model(x)

                    def one_hot_max(output):
                        # return torch.eye(output.shape[1], device=self.device)[y]
                        logits = output[torch.arange(output.shape[0]), y]
                        return logits[:, None] * torch.eye(output.shape[1], device=self.device)[y]

                    out, heatmaps = heatmap_generator(x, one_hot_max)
                    heatmaps = heatmaps.sum(1)

                    if concept_layer is None:
                        # condition = [{"y": [c_id]} for c_id in out.argmax(1)]
                        condition = [{"y": [target]} for target in y]
                    else:
                        # condition = [{"y": [c_id], concept_layer: self.config.conditioning_channels} for c_id in out.argmax(1)]
                        condition = [{"y": [target], concept_layer: self.config.conditioning_channels} for target in y]

                    record_layers = [] if attribution_layer == "input" else [attribution_layer]
                    attr = attributor(x, condition, composite, record_layer=record_layers)
                    if self.config.skip_wrongly_classified_samples:
                        non_zero = ((attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0) * (out.argmax(1).detach().cpu() == y)).numpy()
                    else:
                        non_zero = (attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0).numpy()

                    if non_zero.sum() == 0:
                        continue

                    heatmaps = heatmaps[non_zero]
                    attributions = heatmaps if attribution_layer == "input" else attr.relevances[attribution_layer][non_zero]
                    # heatmaps = attributions.mean(dim=1)
                    heatmaps = heatmaps[:, None]
                    concept_importance = cc.attribute(attributions, abs_norm=True)[:, None]
                    y = y[non_zero]
                    x = x[non_zero]

                    if number_samples_processed == 0:
                        print("attributions shape: ", attributions.shape)
                        print("heatmaps shape: ", heatmaps.shape)
                        attribution_db_file = create_attribution_database(self.attribution_db_path, attributions[0].shape, self.config.classes_total, len_data)
                        heatmaps_db_file = create_attribution_database(self.heatmaps_db_path, heatmaps[0].shape, self.config.classes_total, len_data)
                        concept_importance_db_file = create_attribution_database(self.concept_importance_db_path, concept_importance[0].shape, self.config.classes_total, len_data)
                        input_db_file = create_input_database(self.input_db_path, x[0].shape, len_data)

                    dataset_split = (np.ones(len(y), dtype=np.uint16) * split)
                    append_attributions(attribution_db_file, number_samples_processed, attributions, attr.prediction[non_zero], y, dataset_split)
                    append_attributions(heatmaps_db_file, number_samples_processed, heatmaps, attr.prediction[non_zero], y, dataset_split)
                    append_attributions(concept_importance_db_file, number_samples_processed, concept_importance, attr.prediction[non_zero], y, dataset_split)
                    append_inputs(input_db_file, number_samples_processed, x, y, np.asarray(batch["url"])[non_zero])
                    number_samples_processed += len(y)
                    pbar.set_description(f"computing relevances ({number_samples_processed}/{len_data}) in dataset split {split} at layer {attribution_layer}...")

        print(f"{number_samples_processed} of {len_data} processed")
        resize_input_db(input_db_file, number_samples_processed)
        resize_attribution_db(attribution_db_file, number_samples_processed)
        resize_attribution_db(heatmaps_db_file, number_samples_processed)
        resize_attribution_db(concept_importance_db_file, number_samples_processed)
        attribution_db_file.close()
        heatmaps_db_file.close()
        concept_importance_db_file.close()
        input_db_file.close()

    def _spectral_clustering(self):

        analysis_db_path = os.path.join(self.analysis_dir, f"analysis.h5")
        if pathlib.Path(analysis_db_path).is_file():
            return analysis_db_path

        number_of_clusters_list = list(range(2, self.config.num_clusters_max + 1))
        pipeline = SpectralClustering(
            preprocessing=Sequential(self._get_preprocessing_pipeline()),
            embedding=EigenDecomposition(n_eigval=self.config.num_eigval, is_output=True),
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=number_of_clusters, kwargs={"max_iter": 700}) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                Parallel([
                    AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                DBSCAN(),
                HDBSCAN(),
                Parallel([
                    TSNEEmbedding(perplexity=perplexity, kwargs={"learning_rate": "auto", "init": "pca"}) for perplexity in self.config.tsne_perplexity
                ], broadcast=True),
                Parallel([
                    UMAPEmbedding(n_neighbors=n_neighbors) for n_neighbors in self.config.umap_neighbors
                ], broadcast=True)
            ], broadcast=True, is_output=True)
        )

        attribution_path = self.concept_importance_db_path if self.config.use_relative_concept_importance else self.attribution_db_path
        with h5py.File(attribution_path, 'r') as attributions_file:
            labels = attributions_file['label'][:]
            splits = attributions_file['dataset_split'][:]
            assert len(labels) == len(splits)
            print("total number of samples to be processed:", len(labels))

        if self.config.split_dataset:
            split_names = [(0, 'train'), (1, 'val')]
            if self.config.include_testsplit:
                split_names += [(2, 'test')]
        else:
            split_names = [(-1, 'full')]

        for split, split_name in split_names:
            for class_idx in range(self.config.classes_total):

                with h5py.File(attribution_path, 'r') as attributions_file:
                    indices_of_samples, = np.nonzero((labels == class_idx) & (splits == split))
                    print(f"number of samples in class {class_idx}: {(labels == class_idx).sum()}")
                    print(f"number of samples in split {split}: {(splits == split).sum()}")
                    print(f"process {len(indices_of_samples)} samples (class {class_idx}, split {split})")
                    attribution_data = attributions_file['attribution'][indices_of_samples, :]
                    print("attribution data shape:", attribution_data.shape)

                print("running spray pipeline...")
                (eigenvalues, embedding), (kmeans, ac, dbscan, hdbscan, tsne, umap) = pipeline(attribution_data)

                with h5py.File(analysis_db_path, 'a') as analysis_file:

                    analysis_name = CLASS_NAMES[self.dataset_class][class_idx] + f' ({split_name})'

                    analysis_group = analysis_file.require_group(analysis_name)
                    analysis_group['index'] = indices_of_samples.astype('uint32')

                    embedding_group = analysis_group.require_group('embedding')
                    embedding_group['spectral'] = embedding.astype(np.float32)
                    embedding_group['spectral'].attrs['eigenvalue'] = eigenvalues.astype(np.float32)

                    for perplexity, tsne_embedding in zip(self.config.tsne_perplexity, tsne):
                        embedding_name = f'tsne-{perplexity:0>5.1f}'
                        embedding_group[embedding_name] = tsne_embedding.astype(np.float32)
                        embedding_group[embedding_name].attrs['embedding'] = 'spectral'
                        embedding_group[embedding_name].attrs['index'] = np.array([0, 1])

                    for n_neighbors, umap_embedding in zip(self.config.umap_neighbors, umap):
                        embedding_name = f'umap-{n_neighbors:0>3d}'
                        embedding_group[embedding_name] = umap_embedding.astype(np.float32)
                        embedding_group[embedding_name].attrs['embedding'] = 'spectral'
                        embedding_group[embedding_name].attrs['index'] = np.array([0, 1])

                    cluster_group = analysis_group.require_group('cluster')
                    for number_of_clusters, clustering in zip(number_of_clusters_list, kmeans):
                        clustering_dataset_name = f'kmeans-{number_of_clusters:02d}'
                        cluster_group[clustering_dataset_name] = clustering
                        cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
                        cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
                        cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                            embedding.shape[1],
                            dtype=np.uint32
                        )

                    for number_of_clusters, clustering in zip(number_of_clusters_list, ac):
                        clustering_dataset_name = f'AgglomerativeClustering-{number_of_clusters:02d}'
                        cluster_group[clustering_dataset_name] = clustering
                        cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
                        cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
                        cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                            embedding.shape[1],
                            dtype=np.uint32
                        )

                    cluster_group['DBSCAN'] = dbscan
                    cluster_group['DBSCAN'].attrs['embedding'] = 'spectral'
                    cluster_group['DBSCAN'].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)

                    cluster_group['HDBSCAN'] = hdbscan
                    cluster_group['HDBSCAN'].attrs['embedding'] = 'spectral'
                    cluster_group['HDBSCAN'].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)

        print("spray analysis finished!")
        return analysis_db_path

    def _create_label_map(self) -> str:
        label_map_file_path = os.path.join(self.analysis_dir, "label_map.json")
        if pathlib.Path(label_map_file_path).is_file():
            return label_map_file_path

        class_names = CLASS_NAMES[self.dataset_class]
        label_map = [{
            'index': idx,
            'name': class_names[idx],
            'word_net_id': str(idx)
        } for idx in range(self.config.classes_total)]
        with open(label_map_file_path, 'w', encoding='utf-8') as label_map_file:
            json.dump(label_map, label_map_file)

        return label_map_file_path

    def _create_project(self, input_size: tuple[int]) -> str:

        output_file_path = os.path.join(self.analysis_dir, "virelay_project.yml")
        if pathlib.Path(output_file_path).is_file():
            return output_file_path

        project = {
            'project': {
                'name': "Finding Confounding Factors",
                'model': "PyTorch ResNet-18",
                'label_map': os.path.split(self.label_map_file_path)[1],
                'dataset': {
                    'name': self.dataset_class,
                    'type': 'hdf5',
                    'path': os.path.split(self.input_db_path)[1],
                    'input_width': input_size[1],
                    'input_height': input_size[2],
                    'down_sampling_method': None,
                    'up_sampling_method': None,
                },
                'attributions': {
                    'attribution_method': "Conditional Attribution EpsilonPlusFlat",
                    'attribution_strategy': 'true_label',
                    'sources': [os.path.split(self.heatmaps_db_path)[1]],
                },
                'analyses': [
                    {
                        'analysis_method': "Spectral",
                        'sources': [os.path.split(self.analysis_db_path)[1]],
                    }
                ]
            }
        }

        with open(output_file_path, 'w', encoding='utf-8') as project_file:
            yaml.dump(project, project_file, default_flow_style=False)
        return output_file_path

    def _run_virelay(self) -> tuple[list[int], list[int]]:

        group_label_dir = os.path.join(self.analysis_dir, "group-labels")

        if not os.path.exists(group_label_dir):
            host_name = "0.0.0.0"
            workspace = Workspace()
            workspace.add_project(self.virelay_project_path)
            app = Server(workspace)
            proc = multiprocessing.Process(
                target=lambda: app.run(host=host_name, port=self.config.port), args=())
            proc.start()
            print(f'ViReLay GUI is active on {host_name}:{self.config.port}')

            print(f'For each class, select the samples containing the confounding feature and export, '
                  f'then do the same for the samples without the confounding feature. Rename the files to '
                  f'"<cls-idx>-confounders.json" and "<cls-idx>-nonconfounders.json", respectively, and place '
                  f'them in the directory {group_label_dir} (leaving out some datapoints completely is generally '
                  f'ok, but might make it harder to correctly assess avg/worst group accuracy later, especially '
                  f'if some groups only have very few samples).')

            while not os.path.exists(group_label_dir):
                time.sleep(4.0)

            proc.terminate()
            print("Re-run script after placing the files in order to process results")
            exit()

        confounder_ids = []
        non_confounder_ids = []
        for file in os.listdir(group_label_dir):
            file = os.path.join(group_label_dir, file)
            if "non" in file:
                non_confounder_ids.extend(json.load(open(file, 'r'))['selectedDataPointIndices'])
            else:
                confounder_ids.extend(json.load(open(file, 'r'))['selectedDataPointIndices'])

        return confounder_ids, non_confounder_ids

    def _get_preprocessing_pipeline(self) -> list[Processor]:
        preprocessing = []
        if self.config.sum_attribution_channels:
            preprocessing.append(SumChannel())
        if self.config.normalize_attribtions:
            preprocessing.append(Normalize())
        preprocessing.append(Flatten())
        return preprocessing

def create_input_database(dataset_file_path, samples_shape, number_of_samples):

    dataset_file = h5py.File(dataset_file_path, 'w')
    dataset_file.create_dataset(
        'data',
        shape=(number_of_samples,) + tuple(samples_shape),
        maxshape=(None,) + tuple(samples_shape),
        dtype='float32'
    )
    dataset_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype='uint16'
    )
    dataset_file.create_dataset(
        'filenames',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype=h5py.string_dtype()
    )
    return dataset_file

def append_inputs(dataset_file, index, sample, label, filenames):
    dataset_file['data'][index:sample.shape[0] + index] = sample.detach().cpu().numpy()
    dataset_file['label'][index:label.shape[0] + index] = label.detach().cpu().numpy()
    dataset_file['filenames'][index:len(filenames) + index] = filenames

def resize_input_db(database_file, new_size):
    database_file['data'].resize(new_size, axis=0)
    database_file['label'].resize(new_size, axis=0)
    database_file['filenames'].resize(new_size, axis=0)

def create_attribution_database(attribution_database_file_path,
                                attribution_shape,
                                number_of_classes,
                                number_of_samples):

    attribution_database_file = h5py.File(attribution_database_file_path, 'w')
    attribution_database_file.create_dataset(
        'attribution',
        shape=(number_of_samples,) + tuple(attribution_shape),
        maxshape=(None,) + tuple(attribution_shape),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'prediction',
        shape=(number_of_samples, number_of_classes),
        maxshape=(None, number_of_classes),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype='uint16'
    )
    attribution_database_file.create_dataset(
        'dataset_split',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype='int16'
    )
    return attribution_database_file

def append_attributions(attribution_database_file,
                        index: int,
                        attributions: torch.Tensor,
                        predictions: torch.Tensor,
                        labels: torch.Tensor,
                        dataset_split: np.ndarray):

    attribution_database_file['attribution'][index:attributions.shape[0] + index] = attributions.detach().cpu().numpy()
    attribution_database_file['prediction'][index:predictions.shape[0] + index] = predictions.detach().cpu().numpy()
    attribution_database_file['label'][index:labels.shape[0] + index] = labels.detach().cpu().numpy()
    attribution_database_file['dataset_split'][index:dataset_split.shape[0] + index] = dataset_split

def resize_attribution_db(database_file, new_size):
    database_file['attribution'].resize(new_size, axis=0)
    database_file['prediction'].resize(new_size, axis=0)
    database_file['label'].resize(new_size, axis=0)
    database_file['dataset_split'].resize(new_size, axis=0)

CLASS_NAMES = {
    "SquareDataset": ['0 - Dark Foreground', '1 - Bright Foreground'],
    "ColoredMnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "Follicles": ['0', '1'],
    "CelebaMale": ["0 - Female", "1 - Male"],
    "CelebaSmiling": ["0 - Not Smiling", "1 - Smiling"],
    "CelebaBlond": ["0 - Not Blond", "1 - Blond"],
    "WaterbirdsDataset": ["0 - Landbird", "1 - Waterbird"],
    "Camelyon17AugmentedDataset": ["0 - No Tumor", "1 - Tumor"]
}

class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class Normalize(Processor):
    def function(self, data):
        data = data / data.sum(tuple(range(1, len(data.shape))), keepdims=True)
        return data


class SumChannel(Processor):
    def function(self, data):
        return data.sum(1)
